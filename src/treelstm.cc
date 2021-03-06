#include "treelstm.h"
#include "system.h"
#include "logging.h"

TreeLSTMState::TreeLSTMCell2 TreeLSTMState::compose1(const dynet::expr::Expression & _x,
                                                     DenseLayer & input_gate,
                                                     DenseLayer & output_gate) {
  dynet::expr::Expression c = input_gate.get_output(_x);
  dynet::expr::Expression o = dynet::expr::logistic(output_gate.get_output(_x));
  dynet::expr::Expression h = dynet::expr::cmult(o, dynet::expr::tanh(c));
  return std::make_pair(h, c);
}

TreeLSTMState::TreeLSTMCell2 TreeLSTMState::compose2(const TreeLSTMCell2 & _i,
                                                     const TreeLSTMCell2 & _j,
                                                     Merge2Layer & input_gate,
                                                     Merge2Layer & output_gate,
                                                     Merge2Layer & left_forget_gate,
                                                     Merge2Layer & right_forget_gate,
                                                     Merge2Layer & rnn_cell) {
  const dynet::expr::Expression & h_i = _i.first;
  const dynet::expr::Expression & h_j = _j.first;
  const dynet::expr::Expression & c_i = _i.second;
  const dynet::expr::Expression & c_j = _j.second;

  dynet::expr::Expression i = dynet::expr::logistic(input_gate.get_output(h_i, h_j));
  dynet::expr::Expression o = dynet::expr::logistic(output_gate.get_output(h_i, h_j));
  dynet::expr::Expression f_i = dynet::expr::logistic(left_forget_gate.get_output(h_i, h_j));
  dynet::expr::Expression f_j = dynet::expr::logistic(right_forget_gate.get_output(h_i, h_j));
  dynet::expr::Expression g = dynet::expr::tanh(rnn_cell.get_output(h_i, h_j));
  dynet::expr::Expression c = dynet::expr::cmult(f_i, c_i) + dynet::expr::cmult(f_j, c_j) + dynet::expr::cmult(i, g);
  dynet::expr::Expression h = dynet::expr::cmult(o, dynet::expr::tanh(c));

  return std::make_pair(h, c);
}

ConstituentTreeLSTMModel::ConstituentTreeLSTMModel(dynet::Model & m,
                                                   unsigned word_dim,
                                                   unsigned hidden_dim) :
  input_gate_leaves(m, word_dim, hidden_dim),
  output_gate_leaves(m, word_dim, hidden_dim),
  input_gate(m, hidden_dim, hidden_dim, hidden_dim),
  output_gate(m, hidden_dim, hidden_dim, hidden_dim),
  left_forget_gate(m, hidden_dim, hidden_dim, hidden_dim),
  right_forget_gate(m, hidden_dim, hidden_dim, hidden_dim),
  rnn_cell(m, hidden_dim, hidden_dim, hidden_dim),
  p_sigma_guard_j(m.add_parameters({ hidden_dim })),
  p_sigma_guard_i(m.add_parameters({ hidden_dim })),
  p_beta_guard(m.add_parameters({ hidden_dim })),
  word_dim(word_dim),
  hidden_dim(hidden_dim) {
}

void ConstituentTreeLSTMModel::new_graph(dynet::ComputationGraph & cg) {
  input_gate_leaves.new_graph(cg);
  output_gate_leaves.new_graph(cg);
  input_gate.new_graph(cg);
  output_gate.new_graph(cg);
  left_forget_gate.new_graph(cg);
  right_forget_gate.new_graph(cg);
  rnn_cell.new_graph(cg);
  sigma_guard_j = dynet::expr::parameter(cg, p_sigma_guard_j);
  sigma_guard_i = dynet::expr::parameter(cg, p_sigma_guard_i);
  beta_guard = dynet::expr::parameter(cg, p_beta_guard);
}

std::vector<dynet::expr::Expression> ConstituentTreeLSTMModel::get_params() {
  std::vector<dynet::expr::Expression> ret;
  for (auto & e : input_gate_leaves.get_params()) { ret.push_back(e); }
  for (auto & e : output_gate_leaves.get_params()) { ret.push_back(e); }
  for (auto & e : input_gate.get_params()) { ret.push_back(e); }
  for (auto & e : output_gate.get_params()) { ret.push_back(e); }
  for (auto & e : left_forget_gate.get_params()) { ret.push_back(e); }
  for (auto & e : right_forget_gate.get_params()) { ret.push_back(e); }
  for (auto & e : rnn_cell.get_params()) { ret.push_back(e); }
  ret.push_back(sigma_guard_i);
  ret.push_back(sigma_guard_j);
  ret.push_back(beta_guard);
  return ret;
}

ConstituentTreeLSTMState::ConstituentTreeLSTMState(ConstituentTreeLSTMModel & treelstm_model) 
  : beta(0), treelstm_model(treelstm_model) {
}

void ConstituentTreeLSTMState::initialize(const std::vector<dynet::expr::Expression>& input) {
  buffer.resize(input.size());
  for (unsigned i = 0; i < input.size(); ++i) {
    buffer[i] = compose1(input[i],
                         treelstm_model.input_gate_leaves,
                         treelstm_model.output_gate_leaves);
  }
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
    beta < buffer_size ? buffer[beta].first : treelstm_model.beta_guard
  });
}

dynet::expr::Expression ConstituentTreeLSTMState::final_repr(const State & state) {
  return stack.back().first;
}

void ConstituentTreeLSTMState::perform_action(const unsigned & action) {
  if (ConstituentSystem::is_shift(action)) {
    stack.push_back(buffer[beta]);
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
                                                                 unsigned word_dim,
                                                                 unsigned hidden_dim) :
  TreeLSTMStateBuilder(m),
  treelstm_model(new ConstituentTreeLSTMModel(m, word_dim, hidden_dim)) {
}

unsigned ConstituentTreeLSTMStateBuilder::state_repr_dim() const {
  return treelstm_model->hidden_dim * 3;
}

unsigned ConstituentTreeLSTMStateBuilder::final_repr_dim() const {
  return treelstm_model->hidden_dim;
}


TreeLSTMState * ConstituentTreeLSTMStateBuilder::build() {
  return new ConstituentTreeLSTMState(*treelstm_model);
}

std::vector<dynet::expr::Expression> ConstituentTreeLSTMStateBuilder::get_params() {
  return treelstm_model->get_params();
}

DependencyTreeLSTMModel::DependencyTreeLSTMModel(dynet::Model & m,
                                                 unsigned word_dim,
                                                 unsigned hidden_dim) :
  input_gate(m, word_dim, hidden_dim, hidden_dim),
  output_gate(m, word_dim, hidden_dim, hidden_dim),
  forget_gate(m, word_dim, hidden_dim, hidden_dim),
  rnn_cell(m, word_dim, hidden_dim, hidden_dim),
  p_sigma_guard_j(m.add_parameters({ word_dim })),
  p_sigma_guard_i(m.add_parameters({ word_dim })),
  p_beta_guard(m.add_parameters({ word_dim })),
  word_dim(word_dim),
  hidden_dim(hidden_dim) {
}

void DependencyTreeLSTMModel::new_graph(dynet::ComputationGraph & cg) {
  input_gate.new_graph(cg);
  output_gate.new_graph(cg);
  forget_gate.new_graph(cg);
  rnn_cell.new_graph(cg);
  zero_hidden = dynet::expr::zeroes(cg, { hidden_dim });
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

std::vector<dynet::expr::Expression> DependencyTreeLSTMModel::get_params() {
  //!
  return std::vector<dynet::expr::Expression>();
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
  TreeLSTMCell2 cell = final_repr_recursive(tree, root);
  return cell.first;
}

TreeLSTMState::TreeLSTMCell2 DependencyTreeLSTMState::final_repr_recursive(
  const std::vector<std::vector<unsigned>>& tree,
  unsigned now) {

  unsigned n_children = tree[now].size();
  dynet::expr::Expression h_sum;
  dynet::expr::Expression c_sum;
  dynet::expr::Expression x_j = buffer[now];

  if (n_children == 0) {
    h_sum = treelstm_model.zero_hidden;
    c_sum = treelstm_model.zero_hidden;
  } else {
    std::vector<dynet::expr::Expression> h_(n_children);
    std::vector<dynet::expr::Expression> c_(n_children);
    for (unsigned k = 0; k < n_children; ++k) {
      unsigned c = tree[now][k];
      auto payload = final_repr_recursive(tree, c);
      h_[k] = payload.first;
      c_[k] = payload.second;
    }
    std::vector<dynet::expr::Expression> f_j_(n_children);
    for (unsigned k = 0; k < n_children; ++k) {
      f_j_[k] = dynet::expr::logistic(treelstm_model.forget_gate.get_output(x_j, h_[k]));
    }
    std::vector<dynet::expr::Expression> c_j_(n_children);
    for (unsigned k = 0; k < n_children; ++k) {
      c_j_[k] = dynet::expr::cmult(f_j_[k], c_[k]);
    }
    h_sum = dynet::expr::sum(h_);
    c_sum = dynet::expr::sum(c_j_);
  }
  
  dynet::expr::Expression i_j = dynet::expr::logistic(treelstm_model.input_gate.get_output(x_j, h_sum));
  dynet::expr::Expression o_j = dynet::expr::logistic(treelstm_model.output_gate.get_output(x_j, h_sum));
  dynet::expr::Expression u_j = dynet::expr::tanh(treelstm_model.rnn_cell.get_output(x_j, h_sum));

  dynet::expr::Expression c = dynet::expr::cmult(i_j, u_j) + c_sum;
  dynet::expr::Expression h = dynet::expr::cmult(o_j, dynet::expr::tanh(c));
  return std::make_pair(h, c);
}

void DependencyTreeLSTMState::perform_action(const unsigned & action) {
  // not need to do anything.
}

DependencyTreeLSTMStateBuilder::DependencyTreeLSTMStateBuilder(dynet::Model & m,
                                                               unsigned word_dim,
                                                               unsigned hidden_dim) :
  TreeLSTMStateBuilder(m),
  treelstm_model(new DependencyTreeLSTMModel(m, word_dim, hidden_dim)) {
}

unsigned DependencyTreeLSTMStateBuilder::state_repr_dim() const {
  return treelstm_model->word_dim * 3;
}

unsigned DependencyTreeLSTMStateBuilder::final_repr_dim() const {
  return treelstm_model->hidden_dim;
}

TreeLSTMState * DependencyTreeLSTMStateBuilder::build() {
  return new DependencyTreeLSTMState(*treelstm_model);
}

std::vector<dynet::expr::Expression> DependencyTreeLSTMStateBuilder::get_params() {
  return treelstm_model->get_params();
}

DependencyTreeLSTMWithBiLSTMModel::DependencyTreeLSTMWithBiLSTMModel(dynet::Model & m,
                                                                     unsigned word_dim,
                                                                     unsigned hidden_dim) : 
  DependencyTreeLSTMModel(m, word_dim, hidden_dim),
  fwd_lstm(1, word_dim, hidden_dim, m),
  bwd_lstm(1, word_dim, hidden_dim, m) {
}

void DependencyTreeLSTMWithBiLSTMModel::new_graph(dynet::ComputationGraph & cg) {
  DependencyTreeLSTMModel::new_graph(cg);
  fwd_lstm.new_graph(cg);
  bwd_lstm.new_graph(cg);
}

std::vector<dynet::expr::Expression> DependencyTreeLSTMWithBiLSTMModel::get_params() {
  std::vector<dynet::expr::Expression> ret = DependencyTreeLSTMModel::get_params();
  for (auto & layer : fwd_lstm.param_vars) { for (auto & e : layer) { ret.push_back(e); } }
  for (auto & layer : bwd_lstm.param_vars) { for (auto & e : layer) { ret.push_back(e); } }
  return ret;
}

DependencyTreeLSTMWithBiLSTMState::DependencyTreeLSTMWithBiLSTMState(DependencyTreeLSTMWithBiLSTMModel & treelstm_model)
  : treelstm_model(treelstm_model) {
}

void DependencyTreeLSTMWithBiLSTMState::initialize(const std::vector<dynet::expr::Expression>& input) {
  treelstm_model.fwd_lstm.start_new_sequence();
  treelstm_model.bwd_lstm.start_new_sequence();
  unsigned len = input.size();
  this->input.resize(len);
  output.resize(len);
  std::vector<dynet::expr::Expression> fwd_output(len);
  std::vector<dynet::expr::Expression> bwd_output(len);
  for (unsigned i = 0; i < len; ++i) {
    this->input[i] = input[i];
    fwd_output[i] = treelstm_model.fwd_lstm.add_input(input[i]);
    bwd_output[len - i - 1] = treelstm_model.bwd_lstm.add_input(input[len - i - 1]);
  }
  for (unsigned i = 0; i < len; ++i) {
    output[i] = dynet::expr::concatenate({ fwd_output[i], bwd_output[i] });
  }
}

void DependencyTreeLSTMWithBiLSTMState::new_graph(dynet::ComputationGraph & cg) {
  treelstm_model.new_graph(cg);
}

dynet::expr::Expression DependencyTreeLSTMWithBiLSTMState::state_repr(const State & state) {
  unsigned stack_size = state.sigma.size();
  unsigned buffer_size = input.size();
  return dynet::expr::concatenate({
    stack_size > 1 ? output[state.sigma[stack_size - 2]] : treelstm_model.sigma_guard_i,
    stack_size > 0 ? output[state.sigma.back()] : treelstm_model.sigma_guard_j,
    state.beta < buffer_size ? output[state.beta] : treelstm_model.beta_guard
  });
}

dynet::expr::Expression DependencyTreeLSTMWithBiLSTMState::final_repr(const State & state) {
  unsigned n_words = input.size();
  unsigned root = UINT_MAX;
  std::vector<std::vector<unsigned>> tree(n_words);
  for (unsigned i = 0; i < n_words; ++i) {
    unsigned hed = state.heads[i];
    if (hed == UINT_MAX) { BOOST_ASSERT(root == UINT_MAX); root = i; } else { tree[hed].push_back(i); }
  }
  BOOST_ASSERT(root != UINT_MAX);
  TreeLSTMCell2 cell = final_repr_recursive(tree, root);
  return cell.first;
}

TreeLSTMState::TreeLSTMCell2 DependencyTreeLSTMWithBiLSTMState::final_repr_recursive(
  const std::vector<std::vector<unsigned>>& tree,
  unsigned now) {

  unsigned n_children = tree[now].size();
  dynet::expr::Expression h_sum;
  dynet::expr::Expression c_sum;
  dynet::expr::Expression x_j = input[now];

  if (n_children == 0) {
    h_sum = treelstm_model.zero_hidden;
    c_sum = treelstm_model.zero_hidden;
  } else {
    std::vector<dynet::expr::Expression> h_(n_children);
    std::vector<dynet::expr::Expression> c_(n_children);
    for (unsigned k = 0; k < n_children; ++k) {
      unsigned c = tree[now][k];
      auto payload = final_repr_recursive(tree, c);
      h_[k] = payload.first;
      c_[k] = payload.second;
    }
    std::vector<dynet::expr::Expression> f_j_(n_children);
    for (unsigned k = 0; k < n_children; ++k) {
      f_j_[k] = dynet::expr::logistic(treelstm_model.forget_gate.get_output(x_j, h_[k]));
    }
    std::vector<dynet::expr::Expression> c_j_(n_children);
    for (unsigned k = 0; k < n_children; ++k) {
      c_j_[k] = dynet::expr::cmult(f_j_[k], c_[k]);
    }
    h_sum = dynet::expr::sum(h_);
    c_sum = dynet::expr::sum(c_j_);
  }

  dynet::expr::Expression i_j = dynet::expr::logistic(treelstm_model.input_gate.get_output(x_j, h_sum));
  dynet::expr::Expression o_j = dynet::expr::logistic(treelstm_model.output_gate.get_output(x_j, h_sum));
  dynet::expr::Expression u_j = dynet::expr::tanh(treelstm_model.rnn_cell.get_output(x_j, h_sum));

  dynet::expr::Expression c = dynet::expr::cmult(i_j, u_j) + c_sum;
  dynet::expr::Expression h = dynet::expr::cmult(o_j, dynet::expr::tanh(c));
  return std::make_pair(h, c);
}

void DependencyTreeLSTMWithBiLSTMState::perform_action(const unsigned & action) {
  // do nothing.
}

DependencyTreeLSTMWithBiLSTMStateBuilder::DependencyTreeLSTMWithBiLSTMStateBuilder(dynet::Model & m, unsigned word_dim, unsigned hidden_dim) :
  TreeLSTMStateBuilder(m),
  treelstm_model(new DependencyTreeLSTMWithBiLSTMModel(m, word_dim, hidden_dim)) {
}

unsigned DependencyTreeLSTMWithBiLSTMStateBuilder::state_repr_dim() const {
  return treelstm_model->word_dim * 3;
}

unsigned DependencyTreeLSTMWithBiLSTMStateBuilder::final_repr_dim() const {
  return treelstm_model->hidden_dim;
}

TreeLSTMState * DependencyTreeLSTMWithBiLSTMStateBuilder::build() {
  return new DependencyTreeLSTMWithBiLSTMState(*treelstm_model);
}

std::vector<dynet::expr::Expression> DependencyTreeLSTMWithBiLSTMStateBuilder::get_params() {
  return treelstm_model->get_params();
}

TreeLSTMStateBuilder * get_state_builder(const std::string & name,
                                         dynet::Model & m,
                                         unsigned word_dim,
                                         unsigned hidden_dim) {
  TreeLSTMStateBuilder * state_builder = nullptr;
  if (name == "constituent" || name == "cons") {
    state_builder = new ConstituentTreeLSTMStateBuilder(m, word_dim, hidden_dim);
  } else if (name == "dependency" || name == "dep") {
    state_builder = new DependencyTreeLSTMStateBuilder(m, word_dim, hidden_dim);
  } else if (name == "dependency_bilstm" || name == "dep_bilstm") {
    state_builder = new DependencyTreeLSTMWithBiLSTMStateBuilder(m, word_dim, hidden_dim);
  } else {
    _ERROR << "Unknown state builder name: " << name;
  }
  _INFO << "Using " << name << " state builder.";
  return state_builder;
}

