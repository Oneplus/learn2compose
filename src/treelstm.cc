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
  : beta(0), treelstm_model(treelstm_model) {
}

void ConstituentTreeLSTMState::initialize(const std::vector<dynet::expr::Expression>& input) {
  buffer.resize(input.size());
  for (unsigned i = 0; i < input.size(); ++i) { buffer[i] = input[i]; }
}

void ConstituentTreeLSTMState::new_graph(dynet::ComputationGraph & cg) {
  treelstm_model.new_graph(cg);
}

dynet::expr::Expression ConstituentTreeLSTMState::state_repr(const State & state) {
  unsigned stack_size = stack.size();
  unsigned buffer_size = buffer.size();
  return dynet::expr::concatenate({
    stack_size > 1 ? stack[stack_size - 2].first : treelstm_model.sigma_guard_i,
    stack_size > 0 ? stack.back().first : treelstm_model.sigma_guard_j,
    beta < buffer_size ? buffer[beta] : treelstm_model.beta_guard
  });
}

dynet::expr::Expression ConstituentTreeLSTMState::final_repr(const State & state) {
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
  forget_gate(m, word_dim, word_dim, word_dim),
  rnn_cell(m, word_dim, word_dim, word_dim),
  p_sigma_guard_j(m.add_parameters({ word_dim })),
  p_sigma_guard_i(m.add_parameters({ word_dim })),
  p_beta_guard(m.add_parameters({ word_dim })),
  word_dim(word_dim) {
}

void DependencyTreeLSTMModel::new_graph(dynet::ComputationGraph & cg) {
  input_gate.new_graph(cg);
  output_gate.new_graph(cg);
  forget_gate.new_graph(cg);
  rnn_cell.new_graph(cg);
  zero_padding = dynet::expr::zeroes(cg, { word_dim });
  sigma_guard_j = dynet::expr::parameter(cg, p_sigma_guard_j);
  sigma_guard_i = dynet::expr::parameter(cg, p_sigma_guard_i);
  beta_guard = dynet::expr::parameter(cg, p_beta_guard);
}

DependencyTreeLSTMState::DependencyTreeLSTMState(DependencyTreeLSTMModel & treelstm_model) 
  : treelstm_model(treelstm_model) {
}

void DependencyTreeLSTMState::initialize(const std::vector<dynet::expr::Expression>& input) {
  buffer.resize(input.size());
  for (unsigned i = 0; i < input.size(); ++i) { buffer[i] = input[i]; }
}

void DependencyTreeLSTMState::new_graph(dynet::ComputationGraph & cg) {
  treelstm_model.new_graph(cg);
}

dynet::expr::Expression DependencyTreeLSTMState::state_repr(const State & state) {
  unsigned stack_size = state.sigma.size();
  unsigned buffer_size = buffer.size();
  return dynet::expr::concatenate({
    stack_size > 1 ? buffer[state.sigma[stack_size - 2]] : treelstm_model.sigma_guard_i,
    stack_size > 0 ? buffer[state.sigma.back()] : treelstm_model.sigma_guard_j,
    state.beta < buffer_size ? buffer[state.beta] : treelstm_model.beta_guard
  });
}

dynet::expr::Expression DependencyTreeLSTMState::final_repr(const State & state) {
  unsigned n_words = buffer.size();
  unsigned root = UINT_MAX;
  std::vector<std::vector<unsigned>> tree(n_words);
  for (unsigned i = 0; i < n_words; ++i) {
    unsigned hed = state.heads[i];
    if (hed == UINT_MAX) { BOOST_ASSERT(root == UINT_MAX); root = i; } else { tree[hed].push_back(i); }
  }
  BOOST_ASSERT(root != UINT_MAX);
  return dynet::expr::Expression();
}

TreeLSTMState::TreeLSTMCell2 DependencyTreeLSTMState::final_repr_recursive(
  const std::vector<std::vector<unsigned>>& tree,
  unsigned now) {

  unsigned n_children = tree[now].size();
  if (n_children == 0) { /* leaf */
    return std::make_pair(buffer[now], treelstm_model.zero_padding);
  }
  
  std::vector<dynet::expr::Expression> h_(n_children);
  std::vector<dynet::expr::Expression> c_(n_children);
  for (unsigned k = 0; k < n_children; ++k) {
    unsigned c = tree[now][k];
    auto payload = final_repr_recursive(tree, c);
    h_[k] = payload.first;
    c_[k] = payload.second;
  }

  dynet::expr::Expression x_j = buffer[now];
  dynet::expr::Expression h_sum = dynet::expr::sum(h_);
  dynet::expr::Expression i_j = dynet::expr::logistic(treelstm_model.input_gate.get_output(x_j, h_sum));

  std::vector<dynet::expr::Expression> f_j_(n_children);
  for (unsigned k = 0; k < n_children; ++k) {
    f_j_[k] = dynet::expr::logistic(treelstm_model.forget_gate.get_output(x_j, h_[k]));
  }

  dynet::expr::Expression o_j = dynet::expr::logistic(treelstm_model.output_gate.get_output(x_j, h_sum));
  dynet::expr::Expression u_j = dynet::expr::tanh(treelstm_model.rnn_cell.get_output(x_j, h_sum));
  std::vector<dynet::expr::Expression> c_j_(n_children);
  for (unsigned k = 0; k < n_children; ++k) {
    c_j_[k] = dynet::expr::cmult(f_j_[k], c_[k]);
  }
  dynet::expr::Expression c = dynet::expr::cmult(i_j, u_j) + dynet::expr::sum(c_j_);
  dynet::expr::Expression h = dynet::expr::cmult(o_j, dynet::expr::tanh(c));
  return std::make_pair(h, c);
}

void DependencyTreeLSTMState::perform_action(const unsigned & action) {
  // not need to do anything.
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
