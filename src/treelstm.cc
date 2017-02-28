#include "treelstm.h"
#include "system.h"
#include "logging.h"

TreeLSTMState::TreeLSTMCell2 TreeLSTMState::compose2(TreeLSTMCell2 & _i,
                                                     TreeLSTMCell2 & _j,
                                                     Merge2Layer & input_gate,
                                                     Merge2Layer & output_gate,
                                                     Merge2Layer & left_forget_gate,
                                                     Merge2Layer & right_forget_gate,
                                                     Merge2Layer & rnn_cell) {
  dynet::expr::Expression & h_i = _i.first;
  dynet::expr::Expression & h_j = _j.first;
  dynet::expr::Expression & c_i = _i.second;
  dynet::expr::Expression & c_j = _j.second;

  dynet::expr::Expression i = dynet::expr::logistic(input_gate.get_output(h_i, h_j));
  dynet::expr::Expression o = dynet::expr::logistic(output_gate.get_output(h_i, h_j));
  dynet::expr::Expression f_i = dynet::expr::logistic(left_forget_gate.get_output(h_i, h_j));
  dynet::expr::Expression f_j = dynet::expr::logistic(right_forget_gate.get_output(h_i, h_j));
  dynet::expr::Expression g = dynet::expr::tanh(rnn_cell.get_output(h_i, h_j));
  dynet::expr::Expression c = dynet::expr::cmult(f_i, c_i) + dynet::expr::cmult(f_j, c_j) + dynet::expr::cmult(i, g);
  dynet::expr::Expression h = dynet::expr::cmult(o, c);

  return std::make_pair(h, c);
}

ConstituentTreeLSTMModel::ConstituentTreeLSTMModel(dynet::Model & m,
                                                   unsigned word_dim) :
  input_gate(m, word_dim, word_dim, word_dim),
  output_gate(m, word_dim, word_dim, word_dim),
  left_forget_gate(m, word_dim, word_dim, word_dim),
  right_forget_gate(m, word_dim, word_dim, word_dim),
  rnn_cell(m, word_dim, word_dim, word_dim),
  p_sigma_guard_j(m.add_parameters({ word_dim })),
  p_sigma_guard_i(m.add_parameters({ word_dim })),
  p_beta_guard(m.add_parameters({ word_dim })),
  word_dim(word_dim) {
}

void ConstituentTreeLSTMModel::new_graph(dynet::ComputationGraph & cg) {
  input_gate.new_graph(cg);
  output_gate.new_graph(cg);
  left_forget_gate.new_graph(cg);
  right_forget_gate.new_graph(cg);
  rnn_cell.new_graph(cg);
  sigma_guard_j = dynet::expr::parameter(cg, p_sigma_guard_j);
  sigma_guard_i = dynet::expr::parameter(cg, p_sigma_guard_i);
  beta_guard = dynet::expr::parameter(cg, p_beta_guard);
  zero_padding = dynet::expr::zeroes(cg, { word_dim });
}

ConstituentTreeLSTMState::ConstituentTreeLSTMState(ConstituentTreeLSTMModel & treelstm_model) 
  : treelstm_model(treelstm_model), beta(0) {
}

void ConstituentTreeLSTMState::initialize(const std::vector<dynet::expr::Expression>& input) {
  buffer.resize(input.size());
  for (unsigned i = 0; i < input.size(); ++i) { buffer[i] = input[i]; }
}

void ConstituentTreeLSTMState::new_graph(dynet::ComputationGraph & cg) {
  treelstm_model.new_graph(cg);
}

dynet::expr::Expression ConstituentTreeLSTMState::state_repr() {
  unsigned stack_size = stack.size();
  unsigned buffer_size = buffer.size();
  return dynet::expr::concatenate({
    stack_size > 1 ? stack[stack_size - 2].first : treelstm_model.sigma_guard_i,
    stack_size > 0 ? stack.back().first : treelstm_model.sigma_guard_j,
    beta < buffer_size ? buffer[beta] : treelstm_model.beta_guard
  });
}

dynet::expr::Expression ConstituentTreeLSTMState::final_repr() {
  return stack.back().first;
}

void ConstituentTreeLSTMState::perform_action(const unsigned & action) {
  if (ConstituentSystem::is_shift(action)) {
    stack.push_back(std::make_pair(buffer[beta], treelstm_model.zero_padding));
    beta++;
  } else {
    unsigned stack_size = stack.size();
    auto payload = compose2(stack[stack_size - 2],
                            stack[stack_size - 1],
                            treelstm_model.input_gate,
                            treelstm_model.output_gate,
                            treelstm_model.left_forget_gate,
                            treelstm_model.right_forget_gate,
                            treelstm_model.rnn_cell);
    stack.pop_back();
    stack.pop_back();
    stack.push_back(payload);
  }
}

ConstituentTreeLSTMStateBuilder::ConstituentTreeLSTMStateBuilder(dynet::Model & m,
                                                                 unsigned word_dim) :
  TreeLSTMStateBuilder(m),
  treelstm_model(new ConstituentTreeLSTMModel(m, word_dim)) {
}

unsigned ConstituentTreeLSTMStateBuilder::state_repr_dim() const {
  return treelstm_model->word_dim * 3;
}

unsigned ConstituentTreeLSTMStateBuilder::final_repr_dim() const {
  return treelstm_model->word_dim;
}


TreeLSTMState * ConstituentTreeLSTMStateBuilder::build() {
  return new ConstituentTreeLSTMState(*treelstm_model);
}

DependencyTreeLSTMModel::DependencyTreeLSTMModel(dynet::Model & m,
                                                 unsigned word_dim) :
  input_gate(m, word_dim, word_dim, word_dim),
  output_gate(m, word_dim, word_dim, word_dim),
  left_forget_gate(m, word_dim, word_dim, word_dim),
  right_forget_gate(m, word_dim, word_dim, word_dim),
  rnn_cell(m, word_dim, word_dim, word_dim),
  p_sigma_guard_j(m.add_parameters({ word_dim })),
  p_sigma_guard_i(m.add_parameters({ word_dim })),
  p_beta_guard(m.add_parameters({ word_dim })),
  word_dim(word_dim) {
}

void DependencyTreeLSTMModel::new_graph(dynet::ComputationGraph & cg) {
  input_gate.new_graph(cg);
  output_gate.new_graph(cg);
  left_forget_gate.new_graph(cg);
  right_forget_gate.new_graph(cg);
  rnn_cell.new_graph(cg);
  zero_padding = dynet::expr::zeroes(cg, { word_dim });
  sigma_guard_j = dynet::expr::parameter(cg, p_sigma_guard_j);
  sigma_guard_i = dynet::expr::parameter(cg, p_sigma_guard_i);
  beta_guard = dynet::expr::parameter(cg, p_beta_guard);
}

DependencyTreeLSTMState::DependencyTreeLSTMState(DependencyTreeLSTMModel & treelstm_model) :
  treelstm_model(treelstm_model), beta(0) {
}

void DependencyTreeLSTMState::initialize(const std::vector<dynet::expr::Expression>& input) {
  buffer.resize(input.size());
  for (unsigned i = 0; i < input.size(); ++i) { buffer[i] = input[i]; }
}

void DependencyTreeLSTMState::new_graph(dynet::ComputationGraph & cg) {
  treelstm_model.new_graph(cg);
}

dynet::expr::Expression DependencyTreeLSTMState::state_repr() {
  unsigned stack_size = stack.size();
  unsigned buffer_size = buffer.size();
  return dynet::expr::concatenate({
    stack_size > 1 ? stack[stack_size - 2].first : treelstm_model.sigma_guard_i,
    stack_size > 0 ? stack.back().first : treelstm_model.sigma_guard_j,
    beta < buffer_size ? buffer[beta] : treelstm_model.beta_guard
  });
}

dynet::expr::Expression DependencyTreeLSTMState::final_repr() {
  return stack.back().first;
}

void DependencyTreeLSTMState::perform_action(const unsigned & action) {
  if (DependencySystem::is_shift(action)) {
    stack.push_back(std::make_pair(buffer[beta], treelstm_model.zero_padding));
    beta++;
  } else if (DependencySystem::is_left(action)) {
    auto cell = compose2(stack[stack.size() - 2],
                         stack.back(),
                         treelstm_model.input_gate,
                         treelstm_model.output_gate,
                         treelstm_model.left_forget_gate,
                         treelstm_model.right_forget_gate,
                         treelstm_model.rnn_cell);
    stack.pop_back();
    stack.pop_back();
    stack.push_back(cell);
  } else {
    auto cell = compose2(stack.back(),
                         stack[stack.size() - 2],
                         treelstm_model.input_gate,
                         treelstm_model.output_gate,
                         treelstm_model.left_forget_gate,
                         treelstm_model.right_forget_gate,
                         treelstm_model.rnn_cell);
    stack.pop_back();
    stack.pop_back();
    stack.push_back(cell);
  }
}

DependencyTreeLSTMStateBuilder::DependencyTreeLSTMStateBuilder(dynet::Model & m,
                                                               unsigned word_dim) :
  TreeLSTMStateBuilder(m),
  treelstm_model(new DependencyTreeLSTMModel(m, word_dim)) {
}

unsigned DependencyTreeLSTMStateBuilder::state_repr_dim() const {
  return treelstm_model->word_dim * 3;
}

unsigned DependencyTreeLSTMStateBuilder::final_repr_dim() const {
  return treelstm_model->word_dim;
}

TreeLSTMState * DependencyTreeLSTMStateBuilder::build() {
  return new DependencyTreeLSTMState(*treelstm_model);
}

TreeLSTMStateBuilder * get_state_builder(const std::string & name,
                                         dynet::Model & m,
                                         unsigned hidde_dim) {
  TreeLSTMStateBuilder * state_builder = nullptr;
  if (name == "constituent" || name == "cons") {
    state_builder = new ConstituentTreeLSTMStateBuilder(m, hidde_dim);
  } else if (name == "dependency" || name == "dep") {
    state_builder = new DependencyTreeLSTMStateBuilder(m, hidde_dim);
  } else {
    _ERROR << "Unknown state builder name: " << name;
  }
  _INFO << "Using " << name << " state builder.";
  return state_builder;
}
