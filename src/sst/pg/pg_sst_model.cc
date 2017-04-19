#include "pg_sst_model.h"
#include "dynet/globals.h"
#include "logging.h"
#include "math_utils.h"

PolicyGradientSSTModel::PolicyGradientSSTModel(unsigned word_size,
                                               unsigned word_dim,
                                               unsigned hidden_dim,
                                               unsigned n_classes,
                                               TransitionSystem & system,
                                               TreeLSTMStateBuilder & state_builder,
                                               float dropout,
                                               bool tune_embedding,
                                               const Embeddings & embeddings,
                                               const std::string & policy_name) :
  SSTModelI(system, state_builder, policy_name),
  policy_projector(state_builder.m, state_builder.state_repr_dim(), hidden_dim),
  policy_scorer(state_builder.m, hidden_dim, system.num_actions()),
  classifier_scorer(state_builder.m, state_builder.final_repr_dim(), n_classes),
  word_emb(state_builder.m, word_size, word_dim, tune_embedding),
  dropout(dropout),
  n_actions(system.num_actions()),
  n_classes(n_classes),
  word_dim(word_dim) {

  for (const auto & p : embeddings) {
    word_emb.p_labels.initialize(p.first, p.second);
  }
}

void PolicyGradientSSTModel::new_graph(dynet::ComputationGraph & cg) {
  policy_projector.new_graph(cg);
  policy_scorer.new_graph(cg);
  classifier_scorer.new_graph(cg);
  word_emb.new_graph(cg);
}

void PolicyGradientSSTModel::discount_rewards(std::vector<float>& rewards) {
  for (int i = rewards.size() - 2; i >= 0; --i) {
    // rewards[i] = rewards[i + 1] * 0.99;
    rewards[i] = rewards[i + 1] * 1.;
  }
}

unsigned PolicyGradientSSTModel::predict(const SSTInstance & inst, State & state) {
  dynet::ComputationGraph cg;
  new_graph(cg);
  unsigned len = inst.sentence.size();
  std::vector<dynet::expr::Expression> input(len);
  for (unsigned i = 0; i < len; ++i) { input[i] = word_emb.embed(inst.sentence[i]); }

  TreeLSTMState * machine = state_builder.build();
  machine->new_graph(cg);
  machine->initialize(input);

  while (!state.is_terminated()) {
    std::vector<unsigned> valid_actions;
    system.get_valid_actions(state, valid_actions);
    dynet::expr::Expression logits = get_policy_logits(machine, state, false);
    std::vector<float> scores = dynet::as_vector(cg.get_value(logits));
    std::vector<float> valid_scores;
    for (unsigned action : valid_actions) { valid_scores.push_back(scores[action]); }
    unsigned action = valid_actions[std::max_element(valid_scores.begin(), valid_scores.end()) - valid_scores.begin()];

    system.perform_action(state, action);
    machine->perform_action(action);
  }
  dynet::expr::Expression pred_expr = get_classifier_logits(machine->final_repr(state), false);
  std::vector<float> pred = dynet::as_vector(cg.get_value(pred_expr));
  unsigned ret = std::max_element(pred.begin(), pred.end()) - pred.begin();

  delete machine;
  return ret;
}

dynet::expr::Expression PolicyGradientSSTModel::get_policy_logits(TreeLSTMState * machine,
                                                                  const State & state,
                                                                  bool train) {
  dynet::expr::Expression ret = policy_projector.get_output(machine->state_repr(state));
  if (train && dropout > 0.f) {
    ret = dynet::expr::dropout(ret, dropout);
  }
  return policy_scorer.get_output(ret);
}

dynet::expr::Expression PolicyGradientSSTModel::objective(dynet::ComputationGraph & cg,
                                                          const SSTInstance & inst) {
  new_graph(cg);
  unsigned len = inst.sentence.size();
  std::vector<dynet::expr::Expression> input(len);
  for (unsigned i = 0; i < len; ++i) { input[i] = word_emb.embed(inst.sentence[i]); }
  std::vector<dynet::expr::Expression> loss;
  std::vector<float> rewards;

  State state(len);
  TreeLSTMState * machine = state_builder.build();
  machine->new_graph(cg);
  machine->initialize(input);

  while (!state.is_terminated()) {
    std::vector<unsigned> valid_actions;
    system.get_valid_actions(state, valid_actions);
    dynet::expr::Expression logits = get_policy_logits(machine, state, true);
    std::vector<float> scores = dynet::as_vector(cg.get_value(logits));
    std::vector<float> valid_prob;
    for (unsigned action : valid_actions) { valid_prob.push_back(scores[action]); }
    softmax_inplace(valid_prob);
    std::discrete_distribution<unsigned> distrib(valid_prob.begin(), valid_prob.end());
    unsigned action = valid_actions[distrib(*(dynet::rndeng))];

    system.perform_action(state, action);
    machine->perform_action(action);
    rewards.push_back(0.);
    loss.push_back(dynet::expr::pickneglogsoftmax(logits, action));
  }
  dynet::expr::Expression logits = get_classifier_logits(machine->final_repr(state), true);
  dynet::expr::Expression pred_expr = dynet::expr::softmax(logits);
  std::vector<float> pred = dynet::as_vector(cg.get_value(pred_expr));
  std::discrete_distribution<unsigned> distrib(pred.begin(), pred.end());
  unsigned action = distrib(*(dynet::rndeng));
  loss.push_back(dynet::expr::pickneglogsoftmax(logits, action));
  rewards.push_back(action == inst.label ? 1.0f : -1.0f);

  delete machine;
  discount_rewards(rewards);
  // float mean, std;
  // mean_and_stddev(rewards, mean, std);
  // for (auto & reward : rewards) { reward = (reward - mean) / std; }
  for (unsigned i = 0; i < loss.size(); ++i) { loss[i] = loss[i] * rewards[i]; }
  return dynet::sum(loss);
}

dynet::expr::Expression PolicyGradientSSTModel::get_classifier_logits(dynet::expr::Expression repr,
                                                                      bool train) {
  dynet::expr::Expression ret = repr;
  if (train && dropout > 0.f) {
    ret = dynet::expr::dropout(ret, dropout);
  }
  return classifier_scorer.get_output(ret);
}
