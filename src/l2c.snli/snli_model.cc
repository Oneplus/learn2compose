#include "snli_model.h"
#include "system.h"
#include "dynet/globals.h"

SNLIModel::SNLIModel(unsigned word_size,
                     unsigned word_dim,
                     unsigned hidden_dim,
                     unsigned n_classes,
                     TransitionSystem & system,
                     TreeLSTMStateBuilder & state_builder,
                     const Embeddings & embeddings,
                     const std::string & policy_name) :
  state_builder(state_builder),
  system(system),
  policy_projector(state_builder.m, state_builder.state_repr_dim(), hidden_dim),
  policy_scorer(state_builder.m, hidden_dim, system.num_actions()),
  classifier_merger(state_builder.m,
                    state_builder.final_repr_dim(), 
                    state_builder.final_repr_dim(),
                    state_builder.final_repr_dim(),
                    state_builder.final_repr_dim(),
                    hidden_dim),
  classifier_scorer(state_builder.m, hidden_dim, n_classes),
  word_emb(state_builder.m, word_size, word_dim, false),
  n_actions(system.num_actions()),
  n_classes(n_classes),
  word_dim(word_dim) {

  for (const auto& p : embeddings) {
    word_emb.p_labels.initialize(p.first, p.second);
  }
}

void SNLIModel::new_graph(dynet::ComputationGraph & cg) {
  policy_projector.new_graph(cg);
  policy_scorer.new_graph(cg);
  classifier_merger.new_graph(cg);
  classifier_scorer.new_graph(cg);
  word_emb.new_graph(cg);
}

dynet::expr::Expression SNLIModel::reinforce(dynet::ComputationGraph & cg,
                                             const SNLIInstance & inst) {
  new_graph(cg);

  std::vector<dynet::expr::Expression> probs1;
  std::vector<dynet::expr::Expression> probs2;
  dynet::expr::Expression s1 = rollin(cg, inst.sentence1, probs1);
  dynet::expr::Expression s2 = rollin(cg, inst.sentence2, probs2);

  dynet::expr::Expression reward = dynet::expr::pickneglogsoftmax(get_classifier_logits(s1, s2), inst.label);

  std::vector<dynet::expr::Expression> loss;
  for (unsigned i = 0; i < probs1.size(); ++i) { loss.push_back(probs1[i] * reward); }
  for (unsigned i = 0; i < probs2.size(); ++i) { loss.push_back(probs2[i] * reward); }
  return dynet::expr::sum(loss);
}

dynet::expr::Expression SNLIModel::rollin(dynet::ComputationGraph & cg,
                                          const std::vector<unsigned>& sentence,
                                          std::vector<dynet::expr::Expression> & probs) {
  unsigned len = sentence.size();
  new_graph(cg);

  State state(len);
  TreeLSTMState * machine = state_builder.build();
  machine->new_graph(cg);

  std::vector<dynet::expr::Expression> input(len);
  for (unsigned i = 0; i < len; ++i) { input[i] = word_emb.embed(sentence[i]); }
  machine->initialize(input);

  std::vector<dynet::expr::Expression> transition_probs;
  while (!state.is_terminated()) {
    std::vector<unsigned> valid_actions;
    system.get_valid_actions(state, valid_actions);
    dynet::expr::Expression logits = get_policy_logits(machine, state);
    dynet::expr::Expression prob_expr = dynet::expr::softmax(logits);
    unsigned action = 0;
    if (policy_type == kSample) {
      if (valid_actions.size() == 1) {
        action = valid_actions[0];
      } else {
        std::vector<float> prob = dynet::as_vector(cg.get_value(prob_expr));
        std::vector<float> valid_prob;
        for (unsigned action : valid_actions) { valid_prob.push_back(prob[action]); }
        std::discrete_distribution<unsigned> distrib(valid_prob.begin(), valid_prob.end());
        action = valid_actions[distrib(*(dynet::rndeng))];
      }
    } else {
      action = system.get_reduce();
      if (!system.is_valid(state, action)) { action = system.get_shift(); }
    }

    system.perform_action(state, action);
    machine->perform_action(action);
    transition_probs.push_back(dynet::expr::pick(prob_expr, action));
  }
  dynet::expr::Expression ret = machine->final_repr(state);
  delete machine;
  return ret;
}

dynet::expr::Expression SNLIModel::decode(dynet::ComputationGraph & cg,
                                          const std::vector<unsigned>& sentence) {
  unsigned len = sentence.size();
  new_graph(cg);

  State state(len);
  TreeLSTMState * machine = state_builder.build();
  machine->new_graph(cg);

  std::vector<dynet::expr::Expression> input(len);
  for (unsigned i = 0; i < len; ++i) { input[i] = word_emb.embed(sentence[i]); }
  machine->initialize(input);

  while (!state.is_terminated()) {
    std::vector<unsigned> valid_actions;
    system.get_valid_actions(state, valid_actions);
    dynet::expr::Expression logits = get_policy_logits(machine, state);
    unsigned action = 0;
    if (policy_type == kSample) {
      std::vector<float> prob = dynet::as_vector(cg.get_value(logits));
      std::vector<float> valid_prob;
      for (unsigned action : valid_actions) { valid_prob.push_back(prob[action]); }
      action = valid_actions[std::max_element(valid_prob.begin(), valid_prob.end()) - valid_prob.begin()];
    } else {
      action = system.get_reduce();
      if (!system.is_valid(state, action)) {
        action = system.get_shift();
      }
    }

    system.perform_action(state, action);
    machine->perform_action(action);
  }
  dynet::expr::Expression ret = machine->final_repr(state);
  delete machine;
  return ret;
}

unsigned SNLIModel::predict(const SNLIInstance & inst) {
  dynet::ComputationGraph cg;
  new_graph(cg);

  dynet::expr::Expression s1 = decode(cg, inst.sentence1);
  dynet::expr::Expression s2 = decode(cg, inst.sentence2);
  dynet::expr::Expression pred_expr = get_classifier_logits(s1, s2);

  std::vector<float> pred_score = dynet::as_vector(cg.get_value(pred_expr));
  return std::max_element(pred_score.begin(), pred_score.end()) - pred_score.begin();
}

dynet::expr::Expression SNLIModel::get_classifier_logits(dynet::expr::Expression & s1, dynet::expr::Expression & s2) {
  dynet::expr::Expression u = dynet::expr::square(s1 - s2);
  dynet::expr::Expression v = dynet::expr::cmult(s1, s2);

  return classifier_scorer.get_output(
    dynet::expr::rectify(classifier_merger.get_output(s1, s2, u, v)));
}

dynet::expr::Expression SNLIModel::get_policy_logits(TreeLSTMState * machine,
                                                     const State & state) {
  return policy_scorer.get_output(dynet::expr::rectify(
    policy_projector.get_output(machine->state_repr(state)))
  );
}
