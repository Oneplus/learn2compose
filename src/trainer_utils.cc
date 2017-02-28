#include "trainer_utils.h"
#include "logging.h"
#include <fstream>
#include <sstream>
#include <boost/algorithm/string.hpp>

po::options_description get_optimizer_options() {
  po::options_description cmd("Optimizer options");
  cmd.add_options()
    ("optimizer", po::value<std::string>()->default_value("simple_sgd"), "The choice of optimizer [simple_sgd, momentum_sgd, adagrad, adadelta, adam].")
    ("optimizer_eta", po::value<float>(), "The initial value of learning rate (eta).")
    ("optimizer_final_eta", po::value<float>()->default_value(0.), "The final value of eta.")
    ("optimizer_enable_eta_decay", po::value<bool>()->required(), "Specify to update eta at the end of each epoch.")
    ("optimizer_eta_decay", po::value<float>(), "The decay rate of eta.")
    ("optimizer_enable_clipping", po::value<bool>()->required(), "Enable clipping.")
    ("optimizer_adam_beta1", po::value<float>()->default_value(0.9f), "The beta1 hyper-parameter of adam")
    ("optimizer_adam_beta2", po::value<float>()->default_value(0.999f), "The beta2 hyper-parameter of adam.")
    ;

  return cmd;
}

dynet::Trainer* get_trainer(const po::variables_map& conf, dynet::Model& model) {
  dynet::Trainer* trainer = nullptr;
  if (!conf.count("optimizer") || conf["optimizer"].as<std::string>() == "simple_sgd") {
    float eta0 = (conf.count("optimizer_eta") ? conf["optimizer_eta"].as<float>() : 0.1);
    trainer = new dynet::SimpleSGDTrainer(model, eta0);
    trainer->eta_decay = 0.08f;
  } else if (conf["optimizer"].as<std::string>() == "momentum_sgd") {
    trainer = new dynet::MomentumSGDTrainer(model);
    trainer->eta_decay = 0.08f;
  } else if (conf["optimizer"].as<std::string>() == "adagrad") {
    trainer = new dynet::AdagradTrainer(model);
  } else if (conf["optimizer"].as<std::string>() == "adadelta") {
    trainer = new dynet::AdadeltaTrainer(model);
  } else if (conf["optimizer"].as<std::string>() == "rmsprop") {
    trainer = new dynet::RmsPropTrainer(model);
  } else if (conf["optimizer"].as<std::string>() == "adam") {
    // default setting is same with Kingma and Ba (2015). 
    float eta0 = (conf.count("optimizer_eta") ? conf["optimizer_eta"].as<float>() : 0.001f);
    float beta1 = conf["optimizer_adam_beta1"].as<float>();
    float beta2 = conf["optimizer_adam_beta2"].as<float>();
    trainer = new dynet::AdamTrainer(model, eta0, beta1, beta2);
  } else {
    _ERROR << "Trainier:: unknown optimizer: " << conf["optimizer"].as<std::string>();
    exit(1);
  }
  _INFO << "Trainer:: using " << conf["optimizer"].as<std::string>() << " optimizer";
  _INFO << "Trainer:: eta = " << trainer->eta;

  if (conf["optimizer_enable_clipping"].as<bool>()) {
    trainer->clipping_enabled = true;
    _INFO << "Trainer:: gradient clipping = enabled";
  } else {
    trainer->clipping_enabled = false;
    _INFO << "Trainer:: gradient clipping = false";
  }

  if (conf["optimizer_enable_eta_decay"].as<bool>()) {
    _INFO << "Trainer:: eta decay = enabled";
    if (conf.count("optimizer_eta_decay")) {
      trainer->eta_decay = conf["optimizer_eta_decay"].as<float>();
      _INFO << "Trainer:: eta decay rate = " << trainer->eta_decay;
    } else {
      _INFO << "Trainer:: eta decay rate not set, use default = " << trainer->eta_decay;
    }
  } else {
    _INFO << "Trainer:: eta decay = disabled";
  }
  return trainer;
}

void get_objective_sequence(std::string expr,
                            unsigned max_iter,
                            std::vector<Model::Param> & seq) {
  // supported expressions:
  //  - "right_reward_1(sample_policy_1,sample_reward_1)+"
  //  - "right_reward_1"
  //  - "(sample_policy_1,sample_reward_1)+"
  auto p = expr.find('(');
  if (p != std::string::npos) {
    if (p == 0) {
      get_objective_sequence_parse_loop(expr, max_iter, seq);
    } else {
      std::string pretrain_str = expr.substr(0, p);
      unsigned time;
      Model::POLICY_TYPE policy_type = Model::kSample;
      Model::OBJECTIVE_TYPE objective_type = Model::kBothPolicyAndReward;
      get_objective_sequence_parse_one_param(pretrain_str, time, policy_type, objective_type);
      for (unsigned t = 0; t < time; ++t) {
        seq.push_back(std::make_pair(policy_type, objective_type));
      }
      std::string loop_str = expr.substr(p + 1);
      get_objective_sequence_parse_loop(loop_str, max_iter, seq);
    }
  } else {
    unsigned time;
    Model::POLICY_TYPE policy_type = Model::kSample;
    Model::OBJECTIVE_TYPE objective_type = Model::kBothPolicyAndReward;
    get_objective_sequence_parse_one_param(expr, time, policy_type, objective_type);
    while (seq.size() < max_iter) {
      seq.push_back(std::make_pair(policy_type, objective_type));
    }
  }
}

void get_objective_sequence_parse_loop(const std::string & expr,
                                       unsigned max_iter,
                                       std::vector<Model::Param>& seq) {
  auto p2 = expr.find(')');
  std::string loop_body_str = expr.substr(1, p2 - 1);
  std::string loop_time_str = expr.substr(p2);
  std::vector<std::string> tokens;
  boost::algorithm::split(tokens, loop_body_str, boost::is_any_of(","));
  unsigned i = 0;
  while (seq.size() < max_iter) {
    unsigned time;
    Model::POLICY_TYPE policy_type = Model::kSample;
    Model::OBJECTIVE_TYPE objective_type = Model::kBothPolicyAndReward;
    get_objective_sequence_parse_one_param(tokens[i], time, policy_type, objective_type);
    for (unsigned t = 0; t < time; ++t) {
      seq.push_back(std::make_pair(policy_type, objective_type));
    }
    i++;
    if (i == tokens.size()) { i = 0; }
  }
}

void get_objective_sequence_parse_one_param(const std::string & expr,
                                            unsigned & time, 
                                            Model::POLICY_TYPE & policy_type,
                                            Model::OBJECTIVE_TYPE & objective_type) {
  std::vector<std::string> params;
  boost::algorithm::split(params, expr, boost::is_any_of("_"));
  assert(params.size() == 3);
  time = boost::lexical_cast<unsigned>(params[2]);
  if (boost::algorithm::to_lower_copy(params[0]) == "right") {
    policy_type = Model::kRight;
  } else if (boost::algorithm::to_lower_copy(params[0]) == "left") {
    policy_type = Model::kLeft;
  } else {
    policy_type = Model::kSample;
  }
  if (boost::algorithm::to_lower_copy(params[1]) == "policy") {
    objective_type = Model::kPolicyOnly;
  } else if (boost::algorithm::to_lower_copy(params[1]) == "reward") {
    objective_type = Model::kRewardOnly;
  } else {
    objective_type = Model::kBothPolicyAndReward;
  }
  if ((policy_type == Model::kLeft || policy_type == Model::kRight) &&
    (objective_type == Model::kPolicyOnly || objective_type == Model::kBothPolicyAndReward)) {
    _WARN << "fix policy (code=" << policy_type
      << "), but objective (code=" << objective_type << ") contains policy network part";
  }
}

void update_trainer(const po::variables_map& conf, dynet::Trainer* trainer) {
  if (conf.count("optimizer_enable_eta_decay")) {
    float final_eta = conf["optimizer_final_eta"].as<float>();
    if (trainer->eta > final_eta) {
      trainer->update_epoch();
      trainer->status();
      _INFO << "Trainer:: trainer updated.";
    } else {
      trainer->eta = final_eta;
      _INFO << "Trainer:: eta reach the final value " << final_eta;
    }
  }
}
