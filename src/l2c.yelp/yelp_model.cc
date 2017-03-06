#include "yelp_model.h"

YelpAvgPipeL2CModel::YelpAvgPipeL2CModel(unsigned word_size,
                                         unsigned word_dim,
                                         unsigned hidden_dim,
                                         unsigned n_classes,
                                         TransitionSystem & system,
                                         TreeLSTMStateBuilder & state_builder,
                                         const Embeddings & embeddings,
                                         const std::string & policy_name) :
  Model(system, state_builder, policy_name),
  policy_projector(state_builder.m, state_builder.state_repr_dim(), hidden_dim),
  policy_scorer(state_builder.m, hidden_dim, system.num_actions()),
  classifier_projector(state_builder.m, state_builder.final_repr_dim(), hidden_dim),
  classifier_scorer(state_builder.m, hidden_dim, n_classes),
  word_emb(state_builder.m, word_size, word_dim, false),
  n_actions(system.num_actions()),
  n_classes(n_classes),
  word_dim(word_dim) {

  for (const auto& p : embeddings) {
    word_emb.p_labels.initialize(p.first, p.second);
  }
}

void YelpAvgPipeL2CModel::new_graph(dynet::ComputationGraph & cg) {
  policy_projector.new_graph(cg);
  policy_scorer.new_graph(cg);
  classifier_projector.new_graph(cg);
  classifier_scorer.new_graph(cg);
  word_emb.new_graph(cg);
}

dynet::expr::Expression YelpAvgPipeL2CModel::objective(dynet::ComputationGraph & cg,
                                                       const YelpInstance & inst,
                                                       OBJECTIVE_TYPE objective_type) {
  new_graph(cg);
  unsigned len = inst.document.size();
  std::vector<dynet::expr::Expression> input(len);
  for (unsigned i = 0; i < len; ++i) { input[i] = sentence_expr(inst, i); }
  std::vector<dynet::expr::Expression> transition_probs;
  dynet::expr::Expression final_repr;
  if (policy_type == kSample && objective_type == kRewardOnly) {
    final_repr = decode(cg, input);
  } else if (policy_type == kLeft) {
    final_repr = left(cg, input);
  } else if (policy_type == kRight) {
    final_repr = right(cg, input);
  } else {
    final_repr = reinforce(cg, input, transition_probs);
  }

  // neg -> minimize
  dynet::expr::Expression reward =
    dynet::expr::pickneglogsoftmax(get_classifier_logits(final_repr), inst.label);

  if (policy_type == kRight || policy_type == kLeft || objective_type == kRewardOnly) {
    return reward;
  } else {  
    std::vector<dynet::expr::Expression> loss;
    if (objective_type == kPolicyOnly) {
      float rwd = dynet::as_scalar(cg.get_value(reward));
      for (unsigned i = 0; i < transition_probs.size(); ++i) {
        loss.push_back(transition_probs[i] * rwd);
      }
    } else if (objective_type == kBothPolicyAndReward) {
      for (unsigned i = 0; i < transition_probs.size(); ++i) {
        loss.push_back(transition_probs[i] * reward);
      }
    }
    return dynet::expr::sum(loss);
  }
}

unsigned YelpAvgPipeL2CModel::predict(const YelpInstance & inst) {
  dynet::ComputationGraph cg;
  new_graph(cg);
  unsigned len = inst.document.size();
  std::vector<dynet::expr::Expression> input(len);
  for (unsigned i = 0; i < len; ++i) { input[i] = sentence_expr(inst, i); }

  dynet::expr::Expression final_repr;
  if (policy_type == kSample) {
    final_repr = decode(cg, input);
  } else if (policy_type == kLeft) {
    final_repr = left(cg, input);
  } else {
    final_repr = right(cg, input);
  }

  dynet::expr::Expression pred_expr = get_classifier_logits(final_repr);
  std::vector<float> pred_score = dynet::as_vector(cg.get_value(pred_expr));
  return std::max_element(pred_score.begin(), pred_score.end()) - pred_score.begin();
}

dynet::expr::Expression YelpAvgPipeL2CModel::get_policy_logits(TreeLSTMState * machine) {
  return policy_scorer.get_output(dynet::expr::rectify(
    policy_projector.get_output(machine->state_repr()))
  );
}

dynet::expr::Expression YelpAvgPipeL2CModel::get_classifier_logits(dynet::expr::Expression repr) {
  return classifier_scorer.get_output(dynet::expr::rectify(
    classifier_projector.get_output(repr))
  );
}

dynet::expr::Expression YelpAvgPipeL2CModel::sentence_expr(const YelpInstance & inst, unsigned sid) {
  std::vector<dynet::expr::Expression> sentence_expr;
  const std::vector<unsigned> & sentence = inst.document[sid];
  for (const auto & word : sentence) { sentence_expr.push_back(word_emb.embed(word)); }
  return dynet::expr::average(sentence_expr);
}

YelpBiGRUPipeL2CModel::YelpBiGRUPipeL2CModel(unsigned word_size,
                                             unsigned word_dim,
                                             unsigned hidden_dim,
                                             unsigned n_classes,
                                             TransitionSystem & system,
                                             TreeLSTMStateBuilder & state_builder,
                                             const Embeddings & embeddings,
                                             const std::string & policy_name) :
  YelpAvgPipeL2CModel(word_size, word_dim, hidden_dim, n_classes, system,
                      state_builder, embeddings, policy_name),
  fwd_gru(1, word_dim, word_dim, state_builder.m),
  bwd_gru(1, word_dim, word_dim, state_builder.m),
  p_fwd_guard(state_builder.m.add_parameters({ word_dim })),
  p_bwd_guard(state_builder.m.add_parameters({ word_dim })) {
}

void YelpBiGRUPipeL2CModel::new_graph(dynet::ComputationGraph & cg) {
  YelpAvgPipeL2CModel::new_graph(cg);
  fwd_gru.new_graph(cg);
  bwd_gru.new_graph(cg);
  fwd_guard = dynet::expr::parameter(cg, p_fwd_guard);
  bwd_guard = dynet::expr::parameter(cg, p_bwd_guard);
}

dynet::expr::Expression YelpBiGRUPipeL2CModel::sentence_expr(const YelpInstance & inst,
                                                             unsigned sid) {
  fwd_gru.start_new_sequence();
  bwd_gru.start_new_sequence();
  const std::vector<unsigned> & sentence = inst.document[sid];
  fwd_gru.add_input(fwd_guard);
  bwd_gru.add_input(bwd_guard);
  unsigned len = sentence.size();
  for (unsigned i = 0; i < sentence.size(); ++i) {
    fwd_gru.add_input(word_emb.embed(sentence[i]));
    bwd_gru.add_input(word_emb.embed(sentence[len - i - 1]));
  }
  return dynet::expr::concatenate({fwd_gru.back(), bwd_gru.back()});
}
