#include <iostream>
#include <unordered_map>
#include <boost/program_options.hpp>
#include "logging.h"
#include "trainer_utils.h"
#include "sst_corpus.h"
#include "sst_model.h"
#include "dynet/init.h"
#if _MSC_VER
#include <process.h>
#endif
namespace po = boost::program_options;

void init_command_line(int argc, char* argv[], po::variables_map& conf) {
  po::options_description general_opt("General options");
  general_opt.add_options()
    ("system", po::value<std::string>()->default_value("cons"), "The type of tree lstm")
    ("policy", po::value<std::string>()->default_value("sample"), "The policy name.")
    ("training_data,T", po::value<std::string>(), "The path to the training data.")
    ("embedding,w", po::value<std::string>(), "The path to the embedding file.")
    ("devel_data,d", po::value<std::string>(), "The path to the development data.")
    ("test_data,t", po::value<std::string>(), "The path to the test data.")
    ("word_dim", po::value<unsigned>()->default_value(100), "The dimension of embedding.")
    ("hidden_dim", po::value<unsigned>()->default_value(100), "The hidden dimension.")
    ("max_iter", po::value<unsigned>()->default_value(10), "The number of iteration.")
    ("report_stops", po::value<unsigned>()->default_value(1000), "The reporting stops")
    ("evaluate_stops", po::value<unsigned>()->default_value(5000), "The evaluation stops")
    ("help,h", "Show help information.")
    ;
  po::options_description optimizer_opt = get_optimizer_options();
  po::options_description cmd("An implementation of learning to compose word into sentence for SNLI dataset");
  cmd.add(general_opt).add(optimizer_opt);

  po::store(po::parse_command_line(argc, argv, cmd), conf);
  if (conf.count("help")) {
    std::cerr << cmd << std::endl;
    exit(1);
  }
  init_boost_log(conf.count("verbose") > 0);
  if (!conf.count("training_data")) {
    std::cerr << "Please specify --training_data (-T), even in test" << std::endl;
    exit(1);
  }
}

float evaluate(SSTModel & engine,
               SSTCorpus & corpus,
               bool devel) {
  unsigned n = devel ? corpus.n_devel : corpus.n_test;
  unsigned n_corr = 0;
  for (unsigned i = 0; i < n; ++i) {
    SSTInstance & inst = (devel ? corpus.devel_instances[i] : corpus.test_instances[i]);
    unsigned result = engine.predict(inst);
    if (result == inst.label) { n_corr++; }
  }

  return float(n_corr) / n;
}

int portable_getpid() {
#ifdef _MSC_VER
  return _getpid();
#else
  return getpid();
#endif
}


int main(int argc, char* argv[]) {
  dynet::initialize(argc, argv, false);
  std::cerr << "command:";
  for (int i = 0; i < argc; ++i) { std::cerr << ' ' << argv[i]; }
  std::cerr << std::endl;

  po::variables_map conf;
  init_command_line(argc, argv, conf);

  SSTCorpus corpus;
  Embeddings embeddings;
  load_word_embeddings(conf["embedding"].as<std::string>(),
                       conf["word_dim"].as<unsigned>(),
                       embeddings, corpus);
  _INFO << "Main:: loaded " << embeddings.size() << " pretrained word embeddings.";

  corpus.load_training_data(conf["training_data"].as<std::string>());
  corpus.load_devel_data(conf["devel_data"].as<std::string>());
  corpus.load_test_data(conf["test_data"].as<std::string>());
  _INFO << "Main:: " << corpus.n_classes << "-way classification.";

  std::vector<unsigned> orders(corpus.n_train);
  for (unsigned i = 0; i < corpus.n_train; ++i) { orders[i] = i; }

  dynet::Model model;
  dynet::Trainer* trainer = get_trainer(conf, model);
  std::string policy_name = conf["policy"].as<std::string>();
  std::string system_name = conf["system"].as<std::string>();
  TransitionSystem * system = get_system(system_name);
  TreeLSTMStateBuilder * state_builder = get_state_builder(system_name,
                                                           model,
                                                           conf["word_dim"].as<unsigned>());
  SSTModel engine(corpus.word_map.size(),
                  conf["word_dim"].as<unsigned>(),
                  conf["hidden_dim"].as<unsigned>(),
                  corpus.n_classes,
                  *system,
                  *state_builder,
                  embeddings,
                  policy_name);

  std::string name = "l2c.model.sst." + system_name + "." + policy_name + "." +
    boost::lexical_cast<std::string>(portable_getpid());

  unsigned max_iter = conf["max_iter"].as<unsigned>();
  float llh = 0.f, llh_in_batch = 0.f;
  float best_dev_p = 0.f, test_p = 0.f;
  unsigned logc = 0;
  for (unsigned iter = 0; iter < max_iter; ++iter) {
    llh = 0.f;
    _INFO << "Main:: start of iteration #" << iter;
    std::shuffle(orders.begin(), orders.end(), (*dynet::rndeng));
    for (unsigned sid : orders) {
      SSTInstance & inst = corpus.training_instances[sid];
      {
        dynet::ComputationGraph cg;
        dynet::expr::Expression l = engine.objective(cg, inst);
        float lp = dynet::as_scalar(cg.forward(l));
        cg.backward(l);
        trainer->update();
        llh += lp;
        llh_in_batch += lp;
      }

      ++logc;
      if (logc % conf["report_stops"].as<unsigned>() == 0) {
        float epoch = (float(logc) / corpus.n_train);
        _INFO << "Main:: iter #" << iter << " (epoch " << epoch << ") loss " << llh_in_batch;
        llh_in_batch = 0.f;
      }
      if (logc % conf["evaluate_stops"].as<unsigned>() == 0) {
        float p = evaluate(engine, corpus, true);
        float epoch = (float(logc) / corpus.n_train);
        _INFO << "Main:: iter #" << iter << " (epoch " << epoch << ") dev p: " << p;
        if (p > best_dev_p) {
          best_dev_p = p;
          test_p = evaluate(engine, corpus, false);
          _INFO << "Main:: new BEST dev: " << best_dev_p << ", test: " << test_p;
          dynet::save_dynet_model(name, (&model));
        }
      }
    }
    float p = evaluate(engine, corpus, true);
    _INFO << "Main:: end of iter #" << iter << " loss: " << llh << " dev p: " << p;
    if (p > best_dev_p) {
      best_dev_p = p;
      test_p = evaluate(engine, corpus, false);
      _INFO << "Main:: new BEST dev: " << best_dev_p << ", test: " << test_p;
      dynet::save_dynet_model(name, (&model));
    }
    update_trainer(conf, trainer);
  }
  _INFO << "Main:: best dev: " << best_dev_p << ", test: " << test_p;
  return 0;
}