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

  for (const auto& p : embeddings) {
    word_emb.p_labels.initialize(p.first, p.second);
  }
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
  new_graph(cg);

  std::vector<dynet::expr::Expression> probs1;
  std::vector<dynet::expr::Expression> probs2;
  dynet::expr::Expression s1 = rollin(cg, inst.sentence1, probs1);
  dynet::expr::Expression s2 = rollin(cg, inst.sentence2, probs2);

  dynet::expr::Expression reward = dynet::expr::pickneglogsoftmax(
    get_score_logits(s1, s2), inst.label);

  std::vector<dynet::expr::Expression> loss;
  for (unsigned i = 0; i < probs1.size(); ++i) { loss.push_back(probs1[i] * reward); }
  for (unsigned i = 0; i < probs2.size(); ++i) { loss.push_back(probs2[i] * reward); }
  return dynet::expr::sum(loss);
}

dynet::expr::Expression SNLIModel::rollin(dynet::ComputationGraph & cg,
                                          const std::vector<unsigned>& sentence,
                                          std::vector<dynet::expr::Expression> & probs) {
  unsigned len = sentence.size();
  return dynet::expr::Expression();
  std::vector<Reinforce::TreeLSTMCell> stack;
  State state(len);
  stack.clear();
  while (!TransitionSystem::is_terminated(state)) {
    bool shift_valid = TransitionSystem::is_valid(state, TransitionSystem::get_shift_id());
    bool reduce_valid = TransitionSystem::is_valid(state, TransitionSystem::get_reduce_id());
    unsigned action;

    dynet::expr::Expression logits = get_policy_logits(state, sentence, stack);
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

    probs.push_back(dynet::expr::pick(dynet::expr::softmax(logits), action));
    if (TransitionSystem::is_shift(action)) {
      shift_function(stack, word_emb.embed(sentence[state.beta]), zero_padding);
      TransitionSystem::shift(state);
    } else {
      reduce_function(stack);
      TransitionSystem::reduce(state);
    }
  }
  return stack[0].first;
}

dynet::expr::Expression SNLIModel::decode(dynet::ComputationGraph & cg,
                                          const std::vector<unsigned>& sentence) {
  unsigned len = sentence.size();
  return dynet::expr::Expression();
  std::vector<Reinforce::TreeLSTMCell> stack;
  State state(len);
  stack.clear();
  while (!TransitionSystem::is_terminated(state)) {
    bool shift_valid = TransitionSystem::is_valid(state, TransitionSystem::get_shift_id());
    bool reduce_valid = TransitionSystem::is_valid(state, TransitionSystem::get_reduce_id());
    unsigned action;

    dynet::expr::Expression logits = get_policy_logits(state, sentence, stack);
    if (shift_valid && reduce_valid) {
      dynet::expr::Expression prob_expr = dynet::expr::softmax(logits);
      std::vector<float> prob = dynet::as_vector(cg.get_value(prob_expr));
      std::discrete_distribution<unsigned> distrib(prob.begin(), prob.end());
      action = std::max_element(prob.begin(), prob.end()) - prob.begin();
    } else if (shift_valid) {
      action = TransitionSystem::get_shift_id();
    } else {
      action = TransitionSystem::get_reduce_id();
    }

    if (TransitionSystem::is_shift(action)) {
      auto p = word_emb.embed(sentence[state.beta]);
      shift_function(stack, p, zero_padding);
      TransitionSystem::shift(state);
    } else {
      reduce_function(stack);
      TransitionSystem::reduce(state);
    }
  }
  return stack[0].first;
}

unsigned SNLIModel::predict(const SNLIInstance & inst) {
  dynet::ComputationGraph cg;
  new_graph(cg);

  dynet::expr::Expression s1 = decode(cg, inst.sentence1);
  dynet::expr::Expression s2 = decode(cg, inst.sentence2);
  dynet::expr::Expression pred_expr = get_score_logits(s1, s2);

  std::vector<float> pred_score = dynet::as_vector(cg.get_value(pred_expr));
  return std::max_element(pred_score.begin(), pred_score.end()) - pred_score.begin();
}

dynet::expr::Expression SNLIModel::get_score_logits(dynet::expr::Expression & s1, dynet::expr::Expression & s2) {
  dynet::expr::Expression u = dynet::expr::square(s1 - s2);
  dynet::expr::Expression v = dynet::expr::cmult(s1, s2);

  return classifier_scorer.get_output(
    dynet::expr::rectify(classifier_merger.get_output(s1, s2, u, v)));
}

dynet::expr::Expression SNLIModel::get_policy_logits(const State & state,
                                                     const std::vector<unsigned> & sentence,
                                                     const std::vector<Reinforce::TreeLSTMCell> & stack) {
  dynet::expr::Expression h_j = (stack.size() > 0 ? stack.back().first : sigma_guard_j);
  dynet::expr::Expression h_i = (stack.size() > 1 ? stack[stack.size() - 2].first : sigma_guard_i);
  dynet::expr::Expression p = (state.beta < state.n) ? word_emb.embed(sentence[state.beta]) : beta_guard;
  return policy_scorer.get_output(dynet::expr::rectify(policy_merger.get_output(h_i, h_j, p)));
}
