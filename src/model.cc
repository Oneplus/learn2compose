#include "model.h"
#include "math_utils.h"
#include "dynet/globals.h"

Model::Model(TransitionSystem & system,
             TreeLSTMStateBuilder & state_builder,
             const std::string & policy_name) :
  state_builder(state_builder),
  system(system) {

  policy_type = kSample;
  if (policy_name == "left") {
    policy_type = kLeft;
  } else if (policy_name == "right") {
    policy_type = kRight;
  }
}

dynet::expr::Expression Model::reinforce(dynet::ComputationGraph & cg,
                                         const std::vector<dynet::expr::Expression>& input,
                                         std::vector<dynet::expr::Expression> & trans_logits,
                                         std::vector<unsigned> & actions,
                                         bool train) {
  unsigned len = input.size();
  State state(len);
  TreeLSTMState * machine = state_builder.build();
  machine->new_graph(cg);
  machine->initialize(input);

  while (!state.is_terminated()) {
    std::vector<unsigned> valid_actions;
    system.get_valid_actions(state, valid_actions);
    dynet::expr::Expression logits = get_policy_logits(machine, state, train);
    unsigned action = 0;
    if (valid_actions.size() == 1) {
      action = valid_actions[0];
    } else {
      std::vector<float> score = dynet::as_vector(cg.get_value(logits));
      std::vector<float> valid_prob;
      for (unsigned action : valid_actions) { valid_prob.push_back(score[action]); }
      softmax_inplace(valid_prob);
      std::discrete_distribution<unsigned> distrib(valid_prob.begin(), valid_prob.end());
      action = valid_actions[distrib(*(dynet::rndeng))];
    }

    system.perform_action(state, action);
    machine->perform_action(action);
    actions.push_back(action);
    trans_logits.push_back(logits);
  }

  dynet::expr::Expression ret = machine->final_repr(state);
  delete machine;
  return ret;
}

dynet::expr::Expression Model::decode(dynet::ComputationGraph & cg,
                                      const std::vector<dynet::expr::Expression>& input,
                                      std::vector<dynet::expr::Expression>& trans_logits,
                                      std::vector<unsigned> & actions,
                                      State & state,
                                      bool train) {
  TreeLSTMState * machine = state_builder.build();
  machine->new_graph(cg);
  machine->initialize(input);

  while (!state.is_terminated()) {
    std::vector<unsigned> valid_actions;
    system.get_valid_actions(state, valid_actions);
    dynet::expr::Expression logits = get_policy_logits(machine, state, train);

    unsigned action = 0;
    if (valid_actions.size() == 1) {
      action = valid_actions[0];
    } else {
      std::vector<float> score = dynet::as_vector(cg.get_value(logits));
      std::vector<float> valid_prob;
      for (unsigned action : valid_actions) { valid_prob.push_back(score[action]); }
      action = valid_actions[std::max_element(valid_prob.begin(), valid_prob.end()) - valid_prob.begin()];
    }

    system.perform_action(state, action);
    machine->perform_action(action);

    actions.push_back(action);
    trans_logits.push_back(logits);
  }

  dynet::expr::Expression ret = machine->final_repr(state);
  delete machine;
  return ret;
}

dynet::expr::Expression Model::decode(dynet::ComputationGraph & cg,
                                      const std::vector<dynet::expr::Expression>& input,
                                      State & state,
                                      bool train) {
  TreeLSTMState * machine = state_builder.build();
  machine->new_graph(cg);
  machine->initialize(input);

  while (!state.is_terminated()) {
    std::vector<unsigned> valid_actions;
    system.get_valid_actions(state, valid_actions);
    dynet::expr::Expression logits = get_policy_logits(machine, state, train);
    unsigned action = 0;
    if (valid_actions.size() == 1) {
      action = valid_actions[0];
    } else {
      std::vector<float> score = dynet::as_vector(cg.get_value(logits));
      std::vector<float> valid_score;
      for (unsigned action : valid_actions) { valid_score.push_back(score[action]); }
      action = valid_actions[std::max_element(valid_score.begin(), valid_score.end()) - valid_score.begin()];
    }

    system.perform_action(state, action);
    machine->perform_action(action);
  }

  dynet::expr::Expression ret = machine->final_repr(state);
  delete machine;
  return ret;
}

dynet::expr::Expression Model::decode(dynet::ComputationGraph & cg,
                                      const std::vector<dynet::expr::Expression>& input,
                                      std::vector<dynet::expr::Expression>& trans_logits,
                                      std::vector<unsigned> & actions,
                                      bool train) {
  unsigned len = input.size();
  State state(len);
  return decode(cg, input, trans_logits, actions, state, train);
}

dynet::expr::Expression Model::decode(dynet::ComputationGraph & cg,
                                      const std::vector<dynet::expr::Expression>& input,
                                      bool train) {
  unsigned len = input.size();
  State state(len);
  return decode(cg, input, state, train);
}

dynet::expr::Expression Model::left(dynet::ComputationGraph & cg,
                                    const std::vector<dynet::expr::Expression>& input,
                                    State & state) {
  TreeLSTMState * machine = state_builder.build();
  machine->new_graph(cg);
  machine->initialize(input);

  while (!state.is_terminated()) {
    unsigned action = system.get_shift();
    if (!system.is_valid(state, action)) { action = system.get_reduce(); }
    system.perform_action(state, action);
    machine->perform_action(action);
  }

  dynet::expr::Expression ret = machine->final_repr(state);
  delete machine;
  return ret;
}

dynet::expr::Expression Model::left(dynet::ComputationGraph & cg,
                                    const std::vector<dynet::expr::Expression>& input,
                                    std::vector<dynet::expr::Expression>& trans_logits,
                                    std::vector<unsigned> & actions,
                                    State & state,
                                    bool train) {
  TreeLSTMState * machine = state_builder.build();
  machine->new_graph(cg);
  machine->initialize(input);

  while (!state.is_terminated()) {
    dynet::expr::Expression logits = get_policy_logits(machine, state, train);
    dynet::expr::Expression prob_expr = dynet::expr::softmax(logits);

    unsigned action = system.get_shift();
    if (!system.is_valid(state, action)) { action = system.get_reduce(); }
    system.perform_action(state, action);
    machine->perform_action(action);
    actions.push_back(action);
    trans_logits.push_back(logits);
  }

  dynet::expr::Expression ret = machine->final_repr(state);
  delete machine;
  return ret;
}

dynet::expr::Expression Model::left(dynet::ComputationGraph & cg, 
                                    const std::vector<dynet::expr::Expression>& input) {
  unsigned len = input.size();
  State state(len);
  return left(cg, input, state);
}

dynet::expr::Expression Model::left(dynet::ComputationGraph & cg,
                                    const std::vector<dynet::expr::Expression>& input,
                                    std::vector<dynet::expr::Expression> & trans_probs,
                                    std::vector<unsigned> & actions,
                                    bool train) {
  unsigned len = input.size();
  State state(len);
  return left(cg, input, trans_probs, actions, state, train);
}

dynet::expr::Expression Model::right(dynet::ComputationGraph & cg, 
                                     const std::vector<dynet::expr::Expression>& input,
                                     State & state) {
  TreeLSTMState * machine = state_builder.build();
  machine->new_graph(cg);
  machine->initialize(input);

  while (!state.is_terminated()) {
    unsigned action = system.get_reduce();
    if (!system.is_valid(state, action)) { action = system.get_shift(); }

    system.perform_action(state, action);
    machine->perform_action(action);
  }

  dynet::expr::Expression ret = machine->final_repr(state);
  delete machine;
  return ret;
}

dynet::expr::Expression Model::right(dynet::ComputationGraph & cg,
                                     const std::vector<dynet::expr::Expression>& input,
                                     std::vector<dynet::expr::Expression>& trans_logits,
                                     std::vector<unsigned> & actions,
                                     State & state,
                                     bool train) {
  TreeLSTMState * machine = state_builder.build();
  machine->new_graph(cg);
  machine->initialize(input);

  while (!state.is_terminated()) {
    dynet::expr::Expression logits = get_policy_logits(machine, state, train);
    unsigned action = system.get_reduce();
    if (!system.is_valid(state, action)) { action = system.get_shift(); }

    system.perform_action(state, action);
    machine->perform_action(action);
    actions.push_back(action);
    trans_logits.push_back(logits);
  }

  dynet::expr::Expression ret = machine->final_repr(state);
  delete machine;
  return ret;
}

dynet::expr::Expression Model::right(dynet::ComputationGraph & cg,
                                     const std::vector<dynet::expr::Expression>& input) {
  unsigned len = input.size();
  State state(len);
  return right(cg, input, state);
}

dynet::expr::Expression Model::right(dynet::ComputationGraph & cg,
                                     const std::vector<dynet::expr::Expression>& input,
                                     std::vector<dynet::expr::Expression> & trans_probs,
                                     std::vector<unsigned> & actions,
                                     bool train) {
  unsigned len = input.size();
  State state(len);
  return right(cg, input, trans_probs, actions, state, train);
}

dynet::expr::Expression Model::execute(dynet::ComputationGraph & cg,
                                       const std::vector<dynet::expr::Expression>& input,
                                       const std::vector<unsigned>& actions, 
                                       std::vector<dynet::expr::Expression>& trans_logits,
                                       State & state,
                                       bool train) {
  TreeLSTMState * machine = state_builder.build();
  machine->new_graph(cg);
  machine->initialize(input);

  unsigned n_step = 0;
  while (!state.is_terminated()) {
    dynet::expr::Expression logits = get_policy_logits(machine, state, train);
    unsigned action = actions[n_step];
    system.perform_action(state, action);
    machine->perform_action(action);
    trans_logits.push_back(logits);
    n_step++;
  }

  dynet::expr::Expression ret = machine->final_repr(state);
  delete machine;
  return ret;
}

dynet::expr::Expression Model::execute(dynet::ComputationGraph & cg, 
                                       const std::vector<dynet::expr::Expression>& input, 
                                       const std::vector<unsigned>& actions,
                                       State & state,
                                       bool train) {
  TreeLSTMState * machine = state_builder.build();
  machine->new_graph(cg);
  machine->initialize(input);
  unsigned n_step = 0;
  while (!state.is_terminated()) {
    unsigned action = actions[n_step];
    system.perform_action(state, action);
    machine->perform_action(action);
    n_step++;
  }

  dynet::expr::Expression ret = machine->final_repr(state);
  delete machine;
  return ret;
}

dynet::expr::Expression Model::execute(dynet::ComputationGraph & cg,
                                       const std::vector<dynet::expr::Expression>& input,
                                       const std::vector<unsigned>& actions, 
                                       std::vector<dynet::expr::Expression>& trans_logits,
                                       bool train) {
  unsigned len = input.size();
  State state(len);
  return execute(cg, input, actions, trans_logits, state, train);
}

dynet::expr::Expression Model::execute(dynet::ComputationGraph & cg,
                                       const std::vector<dynet::expr::Expression>& input,
                                       const std::vector<unsigned>& actions,
                                       bool train) {
  unsigned len = input.size();
  State state(len);
  return execute(cg, input, actions, state, train);
}

void Model::set_policy(const std::string & policy_name) {
  if (policy_name == "left") {
    policy_type = kLeft;
  } else if (policy_name == "right") {
    policy_type = kRight;
  } else {
    policy_type = kSample;
  }
}

void Model::set_policy(const POLICY_TYPE & policy_type_) {
  policy_type = policy_type_;
}
