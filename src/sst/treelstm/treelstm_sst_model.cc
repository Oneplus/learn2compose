#include "treelstm_sst_model.h"

TreeLSTMSSTModel::TreeLSTMSSTModel(unsigned word_size,
                                   unsigned word_dim,
                                   unsigned hidden_dim,
                                   unsigned n_classes,
                                   TransitionSystem & system,
                                   TreeLSTMStateBuilder & state_builder,
                                   float dropout,
                                   float l2,
                                   bool tune_embedding,
                                   const Embeddings & embeddings,
                                   const std::string & policy_name) :
  SSTModelI(system, state_builder, policy_name),
  classifier_scorer(state_builder.m, state_builder.final_repr_dim(), n_classes),
  word_emb(state_builder.m, word_size, word_dim, 0.05, tune_embedding),
  dropout(dropout),
  l2(l2),
  n_actions(system.num_actions()),
  n_classes(n_classes),
  word_dim(word_dim) {

  for (const auto& p : embeddings) {
    word_emb.p_labels.initialize(p.first, p.second);
  }
}

void TreeLSTMSSTModel::new_graph(dynet::ComputationGraph & cg) {
  classifier_scorer.new_graph(cg);
  word_emb.new_graph(cg);
}

dynet::expr::Expression TreeLSTMSSTModel::objective(dynet::ComputationGraph & cg,
                                                    const SSTInstanceWithTree & inst) {
  new_graph(cg);
  unsigned len = inst.sentence.size();
  std::vector<dynet::expr::Expression> input(len);
  for (unsigned i = 0; i < len; ++i) { input[i] = word_emb.embed(inst.sentence[i]); }
  std::vector<unsigned> actions;
  system.get_oracle_actions(inst.parents, actions);

  TreeLSTMState * machine = state_builder.build();
  State state(len);
  machine->new_graph(cg);
  machine->initialize(input);
  unsigned n_step = 0;
  std::vector<dynet::expr::Expression> loss;
  while (!state.is_terminated()) {
    unsigned action = actions[n_step];
    system.perform_action(state, action);
    machine->perform_action(action);
    n_step++;

    unsigned label = inst.labels[state.sigma.back()];
    dynet::expr::Expression final_repr = machine->final_repr(state);
    dynet::expr::Expression pred_logits = get_classifier_logits(final_repr, true);
    loss.push_back(dynet::expr::pickneglogsoftmax(pred_logits, label));
  }

  std::vector<dynet::expr::Expression> all_params = state_builder.get_params();
  for (auto & e : classifier_scorer.get_params()) { all_params.push_back(e); }
  std::vector<dynet::expr::Expression> reg;
  for (auto & e : all_params) { reg.push_back(dynet::expr::squared_norm(e)); }
  return dynet::expr::sum(loss) + (0.5 * l2) * dynet::expr::sum(reg);
}

unsigned TreeLSTMSSTModel::predict(const SSTInstance & inst,
                                   State & state) {
  assert(false);
  return 0;
}

unsigned TreeLSTMSSTModel::predict(const SSTInstanceWithTree & inst) {
  dynet::ComputationGraph cg;
  new_graph(cg);
  unsigned len = inst.sentence.size();
  std::vector<dynet::expr::Expression> input(len);
  for (unsigned i = 0; i < len; ++i) { input[i] = word_emb.embed(inst.sentence[i]); }

  std::vector<unsigned> actions;
  system.get_oracle_actions(inst.parents, actions);
  dynet::expr::Expression final_repr = execute(cg, input, actions, false);
  dynet::expr::Expression pred_expr = get_classifier_logits(final_repr, false);
  std::vector<float> pred_score = dynet::as_vector(cg.get_value(pred_expr));
  return std::max_element(pred_score.begin(), pred_score.end()) - pred_score.begin();
}

dynet::expr::Expression TreeLSTMSSTModel::get_policy_logits(TreeLSTMState * machine,
                                                            const State & state,
                                                            bool train) {
  assert(false);
  return dynet::expr::Expression();
}

dynet::expr::Expression TreeLSTMSSTModel::get_classifier_logits(dynet::expr::Expression repr,
                                                                bool train) {
  dynet::expr::Expression ret = repr;
  if (train && dropout > 0.f) {
    ret = dynet::expr::dropout(ret, dropout);
  }
  return classifier_scorer.get_output(ret);
}
