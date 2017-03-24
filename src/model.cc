#include "model.h"
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
                                         std::vector<dynet::expr::Expression> & probs) {
  unsigned len = input.size();
  State state(len);
  TreeLSTMState * machine = state_builder.build();
  machine->new_graph(cg);
  machine->initialize(input);

  while (!state.is_terminated()) {
    std::vector<unsigned> valid_actions;
    system.get_valid_actions(state, valid_actions);
    dynet::expr::Expression logits = get_policy_logits(machine, state);
    dynet::expr::Expression prob_expr = dynet::expr::softmax(logits);
    unsigned action = 0;
    if (valid_actions.size() == 1) {
      action = valid_actions[0];
    } else {
      std::vector<float> prob = dynet::as_vector(cg.get_value(prob_expr));
      std::vector<float> valid_prob;
      for (unsigned action : valid_actions) { valid_prob.push_back(prob[action]); }
      std::discrete_distribution<unsigned> distrib(valid_prob.begin(), valid_prob.end());
      action = valid_actions[distrib(*(dynet::rndeng))];
    }

    system.perform_action(state, action);
    machine->perform_action(action);
    probs.push_back(dynet::expr::pick(prob_expr, action));
  }

  dynet::expr::Expression ret = machine->final_repr(state);
  delete machine;
  return ret;
}

dynet::expr::Expression Model::decode(dynet::ComputationGraph & cg,
                                      const std::vector<dynet::expr::Expression>& input,
                                      std::vector<dynet::expr::Expression>& probs,
                                      State & state) {
  TreeLSTMState * machine = state_builder.build();
  machine->new_graph(cg);
  machine->initialize(input);

  while (!state.is_terminated()) {
    std::vector<unsigned> valid_actions;
    system.get_valid_actions(state, valid_actions);
    dynet::expr::Expression logits = get_policy_logits(machine, state);
    dynet::expr::Expression prob_expr = dynet::expr::softmax(logits);
    unsigned action = 0;
    if (valid_actions.size() == 1) {
      action = valid_actions[0];
    } else {
      std::vector<float> prob = dynet::as_vector(cg.get_value(prob_expr));
      std::vector<float> valid_prob;
      for (unsigned action : valid_actions) { valid_prob.push_back(prob[action]); }
      action = valid_actions[std::max_element(valid_prob.begin(), valid_prob.end()) - valid_prob.begin()];
    }

    system.perform_action(state, action);
    machine->perform_action(action);
    probs.push_back(dynet::expr::pick(prob_expr, action));
  }

  dynet::expr::Expression ret = machine->final_repr(state);
  delete machine;
  return ret;
}

dynet::expr::Expression Model::decode(dynet::ComputationGraph & cg,
                                      const std::vector<dynet::expr::Expression>& input,
                                      State & state) {
  TreeLSTMState * machine = state_builder.build();
  machine->new_graph(cg);
  machine->initialize(input);

  while (!state.is_terminated()) {
    std::vector<unsigned> valid_actions;
    system.get_valid_actions(state, valid_actions);
    dynet::expr::Expression logits = get_policy_logits(machine, state);
    unsigned action = 0;
    if (valid_actions.size() == 1) {
      action = valid_actions[0];
    } else {
      dynet::expr::Expression prob_expr = dynet::expr::softmax(logits);
      std::vector<float> prob = dynet::as_vector(cg.get_value(prob_expr));
      std::vector<float> valid_prob;
      for (unsigned action : valid_actions) { valid_prob.push_back(prob[action]); }
      action = valid_actions[std::max_element(valid_prob.begin(), valid_prob.end()) - valid_prob.begin()];
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
                                      std::vector<dynet::expr::Expression>& probs) {
  unsigned len = input.size();
  State state(len);
  return decode(cg, input, probs, state);
}

dynet::expr::Expression Model::decode(dynet::ComputationGraph & cg,
                                      const std::vector<dynet::expr::Expression>& input) {
  unsigned len = input.size();
  State state(len);
  return decode(cg, input, state);
}

dynet::expr::Expression Model::left(dynet::ComputationGraph & cg,
                                    const std::vector<dynet::expr::Expression>& input,
                                    State & state) {
  TreeLSTMState * machine = state_builder.build();
  machine->new_graph(cg);
  machine->initialize(input);

  while (!state.is_terminated()) {
    std::vector<unsigned> valid_actions;
    system.get_valid_actions(state, valid_actions);
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
                                    std::vector<dynet::expr::Expression>& probs,
                                    State & state) {
  TreeLSTMState * machine = state_builder.build();
  machine->new_graph(cg);
  machine->initialize(input);

  while (!state.is_terminated()) {
    std::vector<unsigned> valid_actions;
    system.get_valid_actions(state, valid_actions);
    dynet::expr::Expression logits = get_policy_logits(machine, state);
    dynet::expr::Expression prob_expr = dynet::expr::softmax(logits);

    unsigned action = system.get_shift();
    if (!system.is_valid(state, action)) { action = system.get_reduce(); }
    system.perform_action(state, action);
    machine->perform_action(action);
    probs.push_back(dynet::expr::pick(prob_expr, action));
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
                                    std::vector<dynet::expr::Expression> & probs) {
  unsigned len = input.size();
  State state(len);
  return left(cg, input, probs, state);
}

dynet::expr::Expression Model::right(dynet::ComputationGraph & cg, 
                                     const std::vector<dynet::expr::Expression>& input,
                                     State & state) {
  TreeLSTMState * machine = state_builder.build();
  machine->new_graph(cg);
  machine->initialize(input);

  while (!state.is_terminated()) {
    std::vector<unsigned> valid_actions;
    system.get_valid_actions(state, valid_actions);
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
                                     std::vector<dynet::expr::Expression>& probs,
                                     State & state) {
  TreeLSTMState * machine = state_builder.build();
  machine->new_graph(cg);
  machine->initialize(input);

  while (!state.is_terminated()) {
    std::vector<unsigned> valid_actions;
    system.get_valid_actions(state, valid_actions);
    dynet::expr::Expression logits = get_policy_logits(machine, state);
    dynet::expr::Expression prob_expr = dynet::expr::softmax(logits);
    unsigned action = system.get_reduce();
    if (!system.is_valid(state, action)) { action = system.get_shift(); }

    system.perform_action(state, action);
    machine->perform_action(action);
    probs.push_back(dynet::expr::pick(prob_expr, action));
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
                                     std::vector<dynet::expr::Expression> & probs) {
  unsigned len = input.size();
  State state(len);
  return right(cg, input, probs, state);
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
