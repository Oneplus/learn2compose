#ifndef TREELSTM_H
#define TREELSTM_H

#include <vector>
#include "layer.h"
#include "system.h"
#include "dynet/expr.h"


struct TreeLSTMModel {
  virtual ~TreeLSTMModel() {}
  virtual void new_graph(dynet::ComputationGraph & cg) = 0;
  virtual std::vector<dynet::expr::Expression> get_params() = 0;
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

  TreeLSTMCell2 compose1(const dynet::expr::Expression & _x,
                         DenseLayer & input_gate,
                         DenseLayer & output_gate);

  TreeLSTMCell2 compose2(const TreeLSTMCell2 & _i,
                         const TreeLSTMCell2 & _j,
                         Merge2Layer & input_gate,
                         Merge2Layer & output_gate,
                         Merge2Layer & left_forget_gate,
                         Merge2Layer & right_forget_gate,
                         Merge2Layer & rnn_cell);

  virtual ~TreeLSTMState() {}
  virtual void initialize(const std::vector<dynet::expr::Expression> & input) = 0;
  virtual void new_graph(dynet::ComputationGraph & cg) = 0;
  virtual dynet::expr::Expression state_repr(const State & state) = 0;
  virtual dynet::expr::Expression final_repr(const State & state) = 0;
  virtual void perform_action(const unsigned & action) = 0;
};

struct TreeLSTMStateBuilder {
  dynet::Model & m;

  TreeLSTMStateBuilder(dynet::Model & m) : m(m) {}
  ~TreeLSTMStateBuilder() {}

  virtual unsigned state_repr_dim() const = 0;
  virtual unsigned final_repr_dim() const = 0;
  virtual TreeLSTMState * build() = 0;
  virtual std::vector<dynet::expr::Expression> get_params() = 0;
};

//=========================================================
// Constituent Tree LSTM
//=========================================================

struct ConstituentTreeLSTMModel : public TreeLSTMModel {
  // TreeLSTM function.
  DenseLayer input_gate_leaves;
  DenseLayer output_gate_leaves;
  Merge2Layer input_gate;
  Merge2Layer output_gate;
  Merge2Layer left_forget_gate;
  Merge2Layer right_forget_gate;
  Merge2Layer rnn_cell;
  dynet::Parameter p_sigma_guard_j;
  dynet::Parameter p_sigma_guard_i;
  dynet::Parameter p_beta_guard;
  dynet::expr::Expression sigma_guard_j;
  dynet::expr::Expression sigma_guard_i;
  dynet::expr::Expression beta_guard;
  unsigned word_dim;
  unsigned hidden_dim;

  ConstituentTreeLSTMModel(dynet::Model & m, unsigned word_dim, unsigned hidden_dim);
  void new_graph(dynet::ComputationGraph & cg) override;
  std::vector<dynet::expr::Expression> get_params() override;
};

struct ConstituentTreeLSTMState : public TreeLSTMState {
  std::vector<TreeLSTMCell2> stack;
  std::vector<TreeLSTMCell2> buffer;
  unsigned beta;
  ConstituentTreeLSTMModel & treelstm_model;

  ConstituentTreeLSTMState(ConstituentTreeLSTMModel & treelstm_model);
  void initialize(const std::vector<dynet::expr::Expression> & input);
  void new_graph(dynet::ComputationGraph & cg) override;
  dynet::expr::Expression state_repr(const State & state) override;
  dynet::expr::Expression final_repr(const State & state) override;
  void perform_action(const unsigned & action) override;
};

struct ConstituentTreeLSTMStateBuilder : public TreeLSTMStateBuilder {
  ConstituentTreeLSTMModel * treelstm_model;

  ConstituentTreeLSTMStateBuilder(dynet::Model & m, unsigned word_dim, unsigned hidden_dim);
  unsigned state_repr_dim() const override;
  unsigned final_repr_dim() const override;
  TreeLSTMState * build();
  std::vector<dynet::expr::Expression> get_params();
};

//=========================================================
// Dependency Tree LSTM
//=========================================================

struct DependencyTreeLSTMModel : public TreeLSTMModel {
  Merge2Layer input_gate;
  Merge2Layer output_gate;
  Merge2Layer forget_gate;
  Merge2Layer rnn_cell;
  dynet::expr::Expression zero_hidden;
  dynet::Parameter p_sigma_guard_j;
  dynet::Parameter p_sigma_guard_i;
  dynet::Parameter p_beta_guard;
  dynet::expr::Expression sigma_guard_j;
  dynet::expr::Expression sigma_guard_i;
  dynet::expr::Expression beta_guard;
  unsigned word_dim;
  unsigned hidden_dim;

  DependencyTreeLSTMModel(dynet::Model & m, unsigned word_dim, unsigned hidden_dim);
  void new_graph(dynet::ComputationGraph & cg) override;
  virtual std::vector<dynet::expr::Expression> get_params() override;
};

struct DependencyTreeLSTMState : public TreeLSTMState {
  std::vector<dynet::expr::Expression> buffer;
  DependencyTreeLSTMModel & treelstm_model;

  DependencyTreeLSTMState(DependencyTreeLSTMModel & treelstm_model);
  void initialize(const std::vector<dynet::expr::Expression> & input);
  void new_graph(dynet::ComputationGraph & cg) override;
  dynet::expr::Expression state_repr(const State & state) override;
  dynet::expr::Expression final_repr(const State & state) override;
  TreeLSTMCell2 final_repr_recursive(const std::vector<std::vector<unsigned>>& tree,
                                     unsigned now);
  void perform_action(const unsigned & action) override;
};

struct DependencyTreeLSTMStateBuilder : public TreeLSTMStateBuilder {
  DependencyTreeLSTMModel * treelstm_model;

  DependencyTreeLSTMStateBuilder(dynet::Model & m, unsigned word_dim, unsigned hidden_dim);
  unsigned state_repr_dim() const override;
  unsigned final_repr_dim() const override;
  TreeLSTMState * build();
  std::vector<dynet::expr::Expression> get_params(); 
};

//=========================================================
// Dependency Tree with Stack-LSTM
//=========================================================

struct DependencyTreeLSTMWithBiLSTMModel : public DependencyTreeLSTMModel {
  dynet::LSTMBuilder fwd_lstm;
  dynet::LSTMBuilder bwd_lstm;

  DependencyTreeLSTMWithBiLSTMModel(dynet::Model & m,
                                    unsigned word_dim,
                                    unsigned hidden_dim);
  void new_graph(dynet::ComputationGraph & cg) override;
  std::vector<dynet::expr::Expression> get_params() override;
};

struct DependencyTreeLSTMWithBiLSTMState : public TreeLSTMState {
  std::vector<dynet::expr::Expression> input;
  std::vector<dynet::expr::Expression> output;
  DependencyTreeLSTMWithBiLSTMModel & treelstm_model;

  DependencyTreeLSTMWithBiLSTMState(DependencyTreeLSTMWithBiLSTMModel & treelstm_model);
  void initialize(const std::vector<dynet::expr::Expression> & input);
  void new_graph(dynet::ComputationGraph & cg) override;
  dynet::expr::Expression state_repr(const State & state) override;
  dynet::expr::Expression final_repr(const State & state) override;
  TreeLSTMCell2 final_repr_recursive(const std::vector<std::vector<unsigned>>& tree,
                                     unsigned now);
  void perform_action(const unsigned & action) override;
};

struct DependencyTreeLSTMWithBiLSTMStateBuilder : public TreeLSTMStateBuilder {
  DependencyTreeLSTMWithBiLSTMModel * treelstm_model;

  DependencyTreeLSTMWithBiLSTMStateBuilder(dynet::Model & m, unsigned word_dim, unsigned hidden_dim);
  unsigned state_repr_dim() const override;
  unsigned final_repr_dim() const override;
  TreeLSTMState * build();
  std::vector<dynet::expr::Expression> get_params();
};

TreeLSTMStateBuilder * get_state_builder(const std::string & name,
                                         dynet::Model & m,
                                         unsigned word_dim,
                                         unsigned hidden_dim);

#endif  //  end for REINFORCE_H