#ifndef REINFORCE_H
#define REINFORCE_H

#include <vector>
#include "layer.h"
#include "system.h"
#include "dynet/expr.h"

struct Reinforce {
  typedef std::pair<
    dynet::expr::Expression,
    dynet::expr::Expression
  > TreeLSTMCell;

  // TreeLSTM function.
  Merge2Layer input_gate;
  Merge2Layer output_gate;
  Merge2Layer left_forget_gate;
  Merge2Layer right_forget_gate;
  Merge2Layer rnn_cell;

  Reinforce(dynet::Model & m, unsigned hidden_dim);

  void new_graph(dynet::ComputationGraph & cg);

  TreeLSTMCell compose(TreeLSTMCell & _i,
                       TreeLSTMCell & _j);

  virtual void new_graph_impl(dynet::ComputationGraph & cg) = 0;

  void shift_function(std::vector<TreeLSTMCell> & stack,
                      dynet::expr::Expression & h,
                      dynet::expr::Expression & c);

  void reduce_function(std::vector<TreeLSTMCell> & stack);
};

#endif  //  end for REINFORCE_H