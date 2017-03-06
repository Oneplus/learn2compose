#include <iostream>
#include <unordered_map>
#include <boost/program_options.hpp>
#include "logging.h"
#include "trainer_utils.h"
#include "snli_corpus.h"
#include "snli_model.h"
#include "dynet/init.h"
#if _MSC_VER
#include <process.h>
#endif
namespace po = boost::program_options;

void init_command_line(int argc, char* argv[], po::variables_map& conf) {
  po::options_description general_opt("An implementation of learning to compose word into sentence for SNLI dataset.");
  general_opt.add_options()
    ("system", po::value<std::string>()->default_value("cons"), "The type of tree lstm")
    ("policy", po::value<std::string>()->default_value("sample"), "The policy name.")
    ("training_data,T", po::value<std::string>(), "The path to the training data.")
    ("embedding", po::value<std::string>(), "The path to the embedding file.")
    ("devel_data,d", po::value<std::string>(), "The path to the development data.")
    ("test_data,t", po::value<std::string>(), "The path to the test data.")
    ("word_dim", po::value<unsigned>()->default_value(100), "The dimension of embedding.")
    ("max_iter", po::value<unsigned>()->default_value(10), "The number of iteration.")
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

int main(int argc, char* argv[]) {
  dynet::initialize(argc, argv, false);
  std::cerr << "command:";
  for (int i = 0; i < argc; ++i) { std::cerr << ' ' << argv[i]; }
  std::cerr << std::endl;

  po::variables_map conf;
  init_command_line(argc, argv, conf);
  
  SNLICorpus corpus;
  std::unordered_map<unsigned, std::vector<float>> embeddings;
  load_word_embeddings(conf["embedding"].as<std::string>(),
                       conf["word_dim"].as<unsigned>(),
                       embeddings, corpus);
  _INFO << "Main:: loaded " << embeddings.size() << " embeddings.";

  corpus.load_training_data(conf["training_data"].as<std::string>());
  corpus.load_devel_data(conf["devel_data"].as<std::string>());
  corpus.load_test_data(conf["test_data"].as<std::string>());

  std::vector<unsigned> orders(corpus.n_train);
  for (unsigned i = 0; i < corpus.n_train; ++i) { orders[i] = i; }

  // SNLIModel engine;
  unsigned max_iter = conf["max_iter"].as<unsigned>();
  for (unsigned i = 0; i < max_iter; ++i) {
    unsigned sid = orders[i];
    // SNLIInstance & inst = corpus.training_instances[sid];

    // engine.reinforce(inst);
  }
  return 0;
}