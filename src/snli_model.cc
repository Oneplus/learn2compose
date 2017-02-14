#include "snli_model.h"
#include "system.h"

SNLIModel::SNLIModel(dynet::Model & m,
                     unsigned word_size,
                     unsigned word_dim,
                     unsigned hidden_dim,
                     const std::unordered_map<unsigned, std::vector<float>>& embeddings)
  : Reinforce(m, word_dim),
  policy_merger(m, word_dim, word_dim, word_dim, hidden_dim),
  policy_scorer(m, hidden_dim, TransitionSystem::n_actions),
  classifier_merger(m, word_dim, word_dim, word_dim, word_dim, hidden_dim),
  classifier_scorer(m, hidden_dim, n_classes),
  word_emb(m, word_size, word_dim, false),
  p_sigma_guard_j(m.add_parameters({ word_dim, 1})),
  p_sigma_guard_i(m.add_parameters({word_dim, 1})),
  p_beta_guard(m.add_parameters({word_dim, 1})),
  word_dim(word_dim) {
}

void SNLIModel::new_graph_impl(dynet::ComputationGraph & cg) {
  policy_merger.new_graph(cg);
  policy_scorer.new_graph(cg);
  classifier_merger.new_graph(cg);
  classifier_scorer.new_graph(cg);
  word_emb.new_graph(cg);

  sigma_guard_j = dynet::expr::parameter(cg, p_sigma_guard_j);
  sigma_guard_i = dynet::expr::parameter(cg, p_sigma_guard_i);
  beta_guard = dynet::expr::parameter(cg, p_beta_guard);
  zero_padding = dynet::expr::zeroes(cg, { word_dim });
}

dynet::expr::Expression SNLIModel::reinforce(dynet::ComputationGraph & cg,
                                             const SNLIInstance & inst) {
  dynet::expr::Expression s1;
  dynet::expr::Expression s2;
  dynet::expr::Expression u = dynet::expr::square(s1 - s2);
  dynet::expr::Expression v = dynet::expr::cmult(s1, s2);

  dynet::expr::Expression q = classifier_merger.get_output(u, v, s1, s2);
  dynet::expr::Expression loss =
    dynet::expr::pickneglogsoftmax(classifier_scorer.get_output(q), inst.label);
  return loss;
}
