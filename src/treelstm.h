#ifndef TREELSTM_H
#define TREELSTM_H

#include <vector>
#include "layer.h"
#include "system.h"
#include "dynet/expr.h"


struct TreeLSTMModel {
  virtual ~TreeLSTMModel() {}
  virtual void new_graph(dynet::ComputationGraph & cg) = 0;
};

struct TreeLSTMState {
  typedef std::pair<
    dynet::expr::Expression,
    dynet::expr::Expression
  > TreeLSTMCell2;

  typedef std::tuple<
    dynet::expr::Expression,
    dynet::expr::Expression,
    dynet::expr::Expression
  > TreeLSTMCell3;

  TreeLSTMCell2 compose2(TreeLSTMCell2 & _i,
                         TreeLSTMCell2 & _j,
                         Merge2Layer & input_gate,
                         Merge2Layer & output_gate,
                         Merge2Layer & left_forget_gate,
                         Merge2Layer & right_forget_gate,
                         Merge2Layer & rnn_cell);

  virtual ~TreeLSTMState() {}
  virtual void initialize(const std::vector<dynet::expr::Expression> & input) = 0;
  virtual void new_graph(dynet::ComputationGraph & cg) = 0;
  virtual dynet::expr::Expression state_repr() = 0;
  virtual dynet::expr::Expression final_repr() = 0;
  virtual void perform_action(const unsigned & action) = 0;
};

struct TreeLSTMStateBuilder {
  dynet::Model & m;

  TreeLSTMStateBuilder(dynet::Model & m) : m(m) {}
  ~TreeLSTMStateBuilder() {}

  virtual unsigned state_repr_dim() const = 0;
  virtual unsigned final_repr_dim() const = 0;
  virtual TreeLSTMState * build() = 0;
};

struct ConstituentTreeLSTMModel : public TreeLSTMModel {
  // TreeLSTM function.
  Merge2Layer input_gate;
  Merge2Layer output_gate;
  Merge2Layer left_forget_gate;
  Merge2Layer right_forget_gate;
  Merge2Layer rnn_cell;
  dynet::expr::Expression zero_padding;
  dynet::Parameter p_sigma_guard_j;
  dynet::Parameter p_sigma_guard_i;
  dynet::Parameter p_beta_guard;
  dynet::expr::Expression sigma_guard_j;
  dynet::expr::Expression sigma_guard_i;
  dynet::expr::Expression beta_guard;
  unsigned word_dim;

  ConstituentTreeLSTMModel(dynet::Model & m, unsigned word_dim);
  void new_graph(dynet::ComputationGraph & cg) override;
};

struct ConstituentTreeLSTMState : public TreeLSTMState {
  std::vector<TreeLSTMCell2> stack;
  std::vector<dynet::expr::Expression> buffer;
  unsigned beta;
  ConstituentTreeLSTMModel & treelstm_model;

  ConstituentTreeLSTMState(ConstituentTreeLSTMModel & treelstm_model);
  void initialize(const std::vector<dynet::expr::Expression> & input);
  void new_graph(dynet::ComputationGraph & cg) override;
  dynet::expr::Expression state_repr() override;
  dynet::expr::Expression final_repr() override;
  void perform_action(const unsigned & action) override;
};

struct ConstituentTreeLSTMStateBuilder : public TreeLSTMStateBuilder {
  ConstituentTreeLSTMModel * treelstm_model;

  ConstituentTreeLSTMStateBuilder(dynet::Model & m, unsigned word_dim);
  unsigned state_repr_dim() const override;
  unsigned final_repr_dim() const override;
  TreeLSTMState * build();
};

struct DependencyTreeLSTMModel : public TreeLSTMModel {
  Merge2Layer input_gate;
  Merge2Layer output_gate;
  Merge2Layer left_forget_gate;
  Merge2Layer right_forget_gate;
  Merge2Layer rnn_cell;
  dynet::expr::Expression zero_padding;
  dynet::Parameter p_sigma_guard_j;
  dynet::Parameter p_sigma_guard_i;
  dynet::Parameter p_beta_guard;
  dynet::expr::Expression sigma_guard_j;
  dynet::expr::Expression sigma_guard_i;
  dynet::expr::Expression beta_guard;
  unsigned word_dim;

  DependencyTreeLSTMModel(dynet::Model & m, unsigned word_dim);
  void new_graph(dynet::ComputationGraph & cg) override;
};

struct DependencyTreeLSTMState : public TreeLSTMState {
  typedef std::pair<dynet::expr::Expression, dynet::expr::Expression> TreeLSTMCell;
  std::vector<TreeLSTMCell2> stack;
  std::vector<dynet::expr::Expression> buffer;
  unsigned beta;
  DependencyTreeLSTMModel & treelstm_model;

  DependencyTreeLSTMState(DependencyTreeLSTMModel & treelstm_model);
  void initialize(const std::vector<dynet::expr::Expression> & input);
  void new_graph(dynet::ComputationGraph & cg) override;
  dynet::expr::Expression state_repr() override;
  dynet::expr::Expression final_repr() override;
  void perform_action(const unsigned & action) override;
};

struct DependencyTreeLSTMStateBuilder : public TreeLSTMStateBuilder {
  DependencyTreeLSTMModel * treelstm_model;

  DependencyTreeLSTMStateBuilder(dynet::Model & m, unsigned word_dim);
  unsigned state_repr_dim() const override;
  unsigned final_repr_dim() const override;
  TreeLSTMState * build();
};

TreeLSTMStateBuilder * get_state_builder(const std::string & name,
                                         dynet::Model & m,
                                         unsigned word_dim);

#endif  //  end for REINFORCE_H