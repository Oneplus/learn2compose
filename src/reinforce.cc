#include "reinforce.h"

Reinforce::Reinforce(dynet::Model & m, unsigned hidden_dim) :
  input_gate(m, hidden_dim, hidden_dim, hidden_dim),
  output_gate(m, hidden_dim, hidden_dim, hidden_dim),
  left_forget_gate(m, hidden_dim, hidden_dim, hidden_dim),
  right_forget_gate(m, hidden_dim, hidden_dim, hidden_dim),
  rnn_cell(m, hidden_dim, hidden_dim, hidden_dim) {
}

void Reinforce::new_graph(dynet::ComputationGraph & cg) {
  input_gate.new_graph(cg);
  output_gate.new_graph(cg);
  left_forget_gate.new_graph(cg);
  right_forget_gate.new_graph(cg);
  rnn_cell.new_graph(cg);
  new_graph_impl(cg);
}

Reinforce::TreeLSTMCell Reinforce::compose(Reinforce::TreeLSTMCell & _i,
                                           Reinforce::TreeLSTMCell & _j) {
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

void Reinforce::shift_function(std::vector<Reinforce::TreeLSTMCell> & stack,
                               dynet::expr::Expression & h,
                               dynet::expr::Expression & c) {
  stack.push_back(std::make_pair(h, c));
}

void Reinforce::reduce_function(std::vector<Reinforce::TreeLSTMCell> & stack) {
  Reinforce::TreeLSTMCell cell = compose(stack[stack.size() - 2], stack.back());
  stack.pop_back();
  stack.pop_back();
  stack.push_back(cell);
}
