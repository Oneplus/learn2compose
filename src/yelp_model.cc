#include "yelp_model.h"
#include <random>

YelpModel::YelpModel(dynet::Model & m,
                     unsigned word_size,
                     unsigned word_dim,
                     unsigned hidden_dim,
                     unsigned n_classes,
                     const std::unordered_map<unsigned, std::vector<float>> & embeddings)
  : Reinforce(m, word_dim),
  policy_merger(m, word_dim, word_dim, word_dim, hidden_dim),
  policy_scorer(m, hidden_dim, TransitionSystem::n_actions),
  classifier_projector(m, word_dim, hidden_dim),
  classifier_scorer(m, hidden_dim, n_classes),
  word_emb(m, word_size, word_dim, false),
  p_sigma_guard_j(m.add_parameters({ word_dim, 1 })),
  p_sigma_guard_i(m.add_parameters({ word_dim, 1 })),
  p_beta_guard(m.add_parameters({ word_dim, 1 })),
  n_classes(n_classes),
  word_dim(word_dim) {

  for (const auto& p : embeddings) {
    word_emb.p_labels.initialize(p.first, p.second);
  }
}

void YelpModel::new_graph_impl(dynet::ComputationGraph & cg) {
  policy_merger.new_graph(cg);
  policy_scorer.new_graph(cg);
  classifier_projector.new_graph(cg);
  classifier_scorer.new_graph(cg);
  word_emb.new_graph(cg);

  sigma_guard_j = dynet::expr::parameter(cg, p_sigma_guard_j);
  sigma_guard_i = dynet::expr::parameter(cg, p_sigma_guard_i);
  beta_guard = dynet::expr::parameter(cg, p_beta_guard);
  zero_padding = dynet::expr::zeroes(cg, { word_dim });
}

dynet::expr::Expression YelpModel::reinforce(dynet::ComputationGraph & cg,
                                             const YelpInstance & inst) {
  unsigned len = inst.document.size();
  new_graph(cg);

  State state(len);
  stack.clear();
  std::vector<dynet::expr::Expression> transition_probs;
  while (!TransitionSystem::is_terminated(state)) {
    bool shift_valid = TransitionSystem::is_valid(state, TransitionSystem::get_shift_id());
    bool reduce_valid = TransitionSystem::is_valid(state, TransitionSystem::get_reduce_id());
    unsigned action;

    dynet::expr::Expression logits = get_policy_logits(state, inst);
    if (shift_valid && reduce_valid) {
      dynet::expr::Expression prob_expr = dynet::expr::softmax(logits);
      std::vector<float> prob = dynet::as_vector(cg.get_value(prob_expr));
      std::discrete_distribution<unsigned> distrib(prob.begin(), prob.end());
      action = distrib(*(dynet::rndeng));
    } else if (shift_valid) {
      action = TransitionSystem::get_shift_id();
    } else {
      action = TransitionSystem::get_reduce_id();
    }

    transition_probs.push_back(
      dynet::expr::pick(dynet::expr::softmax(logits), action));

    if (TransitionSystem::is_shift(action)) {
      shift_function(stack, sentence_expr(inst, state.beta), zero_padding);
      TransitionSystem::shift(state);
    } else {
      reduce_function(stack);
      TransitionSystem::reduce(state);
    }
  }

  // neg -> minimize
  dynet::expr::Expression reward = dynet::expr::pickneglogsoftmax(
    classifier_scorer.get_output(
    dynet::expr::rectify(classifier_projector.get_output(stack[0].first))),
    inst.label
  );
  // return reward;

  std::vector<dynet::expr::Expression> loss;
  for (unsigned i = 0; i < transition_probs.size(); ++i) {
    loss.push_back(transition_probs[i] * reward);
  }
  return dynet::expr::sum(loss);
}

unsigned YelpModel::predict(const YelpInstance & inst) {
  unsigned len = inst.document.size();
  dynet::ComputationGraph cg;
  new_graph(cg);
  stack.clear();
  State state(len);
  while (!TransitionSystem::is_terminated(state)) {
    bool shift_valid = TransitionSystem::is_valid(state, TransitionSystem::get_shift_id());
    bool reduce_valid = TransitionSystem::is_valid(state, TransitionSystem::get_reduce_id());
    unsigned action;

    if (shift_valid && reduce_valid) {
      dynet::expr::Expression logits = get_policy_logits(state, inst);
      std::vector<float> score = dynet::as_vector(
        cg.get_value(dynet::expr::softmax(logits)));
      action = (score[0] > score[1] ? 0 : 1);
    } else if (shift_valid) {
      action = TransitionSystem::get_shift_id();
    } else {
      action = TransitionSystem::get_reduce_id();
    }

    if (TransitionSystem::is_shift(action)) {
      shift_function(stack, sentence_expr(inst, state.beta), zero_padding);
      TransitionSystem::shift(state);
    } else {
      reduce_function(stack);
      TransitionSystem::reduce(state);
    }
  }

  dynet::expr::Expression pred_expr = classifier_scorer.get_output(
    dynet::expr::rectify(classifier_projector.get_output(stack[0].first)));
  std::vector<float> pred_score = dynet::as_vector(cg.get_value(pred_expr));
  return std::max_element(pred_score.begin(), pred_score.end()) - pred_score.begin();
}

dynet::expr::Expression YelpModel::get_policy_logits(const State & state,
                                                     const YelpInstance & inst) {
  dynet::expr::Expression h_j = (stack.size() > 0 ? stack.back().first : sigma_guard_j);
  dynet::expr::Expression h_i = (stack.size() > 1 ? stack[stack.size() - 2].first : sigma_guard_i);
  dynet::expr::Expression p = (state.beta < state.n) ? sentence_expr(inst, state.beta) : beta_guard;
  return policy_scorer.get_output(dynet::expr::rectify(policy_merger.get_output(h_i, h_j, p)));
}

dynet::expr::Expression YelpModel::sentence_expr(const YelpInstance & inst, unsigned sid) {
  std::vector<dynet::expr::Expression> sentence_expr;
  const std::vector<unsigned> & sentence = inst.document[sid];
  for (const auto & word : sentence) { sentence_expr.push_back(word_emb.embed(word)); }
  return dynet::expr::average(sentence_expr);
}
