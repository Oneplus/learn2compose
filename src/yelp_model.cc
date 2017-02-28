#include "yelp_model.h"

YelpModel::YelpModel(unsigned word_size,
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
  classifier_projector(state_builder.m, word_dim, hidden_dim),
  classifier_scorer(state_builder.m, hidden_dim, n_classes),
  word_emb(state_builder.m, word_size, word_dim, false),
  n_actions(system.num_actions()),
  n_classes(n_classes),
  word_dim(word_dim) {

  for (const auto& p : embeddings) {
    word_emb.p_labels.initialize(p.first, p.second);
  }
}

void YelpModel::new_graph(dynet::ComputationGraph & cg) {
  policy_projector.new_graph(cg);
  policy_scorer.new_graph(cg);
  classifier_projector.new_graph(cg);
  classifier_scorer.new_graph(cg);
  word_emb.new_graph(cg);
}

dynet::expr::Expression YelpModel::objective(dynet::ComputationGraph & cg,
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

unsigned YelpModel::predict(const YelpInstance & inst) {
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

dynet::expr::Expression YelpModel::get_policy_logits(TreeLSTMState * machine) {
  return policy_scorer.get_output(dynet::expr::rectify(
    policy_projector.get_output(machine->state_repr()))
  );
}

dynet::expr::Expression YelpModel::get_classifier_logits(dynet::expr::Expression repr) {
  return classifier_scorer.get_output(dynet::expr::rectify(
    classifier_projector.get_output(repr))
  );
}

dynet::expr::Expression YelpModel::sentence_expr(const YelpInstance & inst, unsigned sid) {
  std::vector<dynet::expr::Expression> sentence_expr;
  const std::vector<unsigned> & sentence = inst.document[sid];
  for (const auto & word : sentence) { sentence_expr.push_back(word_emb.embed(word)); }
  return dynet::expr::average(sentence_expr);
}
