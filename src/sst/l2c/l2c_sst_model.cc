#include "l2c_sst_model.h"

Learn2ComposeSSTModel::Learn2ComposeSSTModel(unsigned word_size,
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

  for (const auto& p : embeddings) {
    word_emb.p_labels.initialize(p.first, p.second);
  }
}

void Learn2ComposeSSTModel::new_graph(dynet::ComputationGraph & cg) {
  policy_projector.new_graph(cg);
  policy_scorer.new_graph(cg);
  classifier_scorer.new_graph(cg);
  word_emb.new_graph(cg);
}

dynet::expr::Expression Learn2ComposeSSTModel::objective(dynet::ComputationGraph & cg,
                                                         const SSTInstance & inst,
                                                         OBJECTIVE_TYPE objective_type) {
  new_graph(cg);
  unsigned len = inst.sentence.size();
  std::vector<dynet::expr::Expression> input(len);
  for (unsigned i = 0; i < len; ++i) { input[i] = word_emb.embed(inst.sentence[i]); }
  std::vector<dynet::expr::Expression> trans_logits;
  std::vector<unsigned> actions;
  dynet::expr::Expression final_repr;
  if (policy_type == kSample) {
    if (objective_type == kRewardOnly) {
      final_repr = decode(cg, input, true);
    } else {
      final_repr = reinforce(cg, input, trans_logits, actions, true);
    }
  } else if (policy_type == kLeft) {
    final_repr = left(cg, input);
  } else {
    final_repr = right(cg, input);
  }
  dynet::expr::Expression pred_logits = get_classifier_logits(final_repr, true);

  if (policy_type == kRight || policy_type == kLeft) {
    return dynet::expr::pickneglogsoftmax(pred_logits, inst.label);
  } else {
    std::vector<dynet::expr::Expression> loss;
    dynet::expr::Expression trans_loss;
    dynet::expr::Expression class_loss;
    if (objective_type == kRewardOnly || objective_type == kBothPolicyAndReward) {
      float prob = 1.;
      for (unsigned i = 0; i < trans_logits.size(); ++i) {
        prob *= dynet::as_vector(cg.get_value(dynet::expr::softmax(trans_logits[i])))[actions[i]];
      }
      class_loss = dynet::expr::pickneglogsoftmax(pred_logits, inst.label) * prob;
    }
    if (objective_type == kPolicyOnly || objective_type == kBothPolicyAndReward) {
      float rwd = dynet::as_scalar(cg.get_value(dynet::expr::pickneglogsoftmax(pred_logits, inst.label)));
      if (rwd < -50.) { rwd = -50.; }
      for (int i = trans_logits.size() - 1; i >= 0; --i) {
        loss.push_back(dynet::expr::pickneglogsoftmax(trans_logits[i], actions[i]) * rwd);
        // rwd *= 0.99;
      }
      trans_loss = dynet::expr::sum(loss);
    }
    if (objective_type == kPolicyOnly) {
      return trans_loss;
    } else if (objective_type == kRewardOnly) {
      return class_loss;
    } else {
      return trans_loss + class_loss;
    }
  }
}

unsigned Learn2ComposeSSTModel::predict(const SSTInstance & inst,
                                        State & state) {
  dynet::ComputationGraph cg;
  new_graph(cg);
  unsigned len = inst.sentence.size();
  std::vector<dynet::expr::Expression> input(len);
  for (unsigned i = 0; i < len; ++i) { input[i] = word_emb.embed(inst.sentence[i]); }

  dynet::expr::Expression final_repr;
  if (policy_type == kSample) {
    final_repr = decode(cg, input, state, false);
  } else if (policy_type == kLeft) {
    final_repr = left(cg, input, state);
  } else {
    final_repr = right(cg, input, state);
  }

  dynet::expr::Expression pred_expr = get_classifier_logits(final_repr, false);
  std::vector<float> pred_score = dynet::as_vector(cg.get_value(pred_expr));
  return std::max_element(pred_score.begin(), pred_score.end()) - pred_score.begin();
}

dynet::expr::Expression Learn2ComposeSSTModel::get_policy_logits(TreeLSTMState * machine,
                                                                 const State & state,
                                                                 bool train) {
  dynet::expr::Expression ret = policy_projector.get_output(machine->state_repr(state));
  if (train && dropout > 0.f) {
    ret = dynet::expr::dropout(ret, dropout);
  }
  return policy_scorer.get_output(ret);
}

dynet::expr::Expression Learn2ComposeSSTModel::get_classifier_logits(dynet::expr::Expression repr,
                                                                     bool train) {
  dynet::expr::Expression ret = repr;
  if (train && dropout > 0.f) {
    ret = dynet::expr::dropout(ret, dropout);
  }
  return classifier_scorer.get_output(ret);
}
