#ifndef MODEL_H
#define MODEL_H

#include "dynet/expr.h"
#include "treelstm.h"
#include "system.h"
#include "layer.h"
#include "corpus.h"

struct Model {
  enum POLICY_TYPE { kSample, kLeft, kRight };
  enum OBJECTIVE_TYPE { kPolicyOnly, kRewardOnly, kBothPolicyAndReward };
  typedef std::pair<POLICY_TYPE, OBJECTIVE_TYPE> Param;

  POLICY_TYPE policy_type;
  TreeLSTMStateBuilder & state_builder;
  TransitionSystem & system;

  Model(TransitionSystem & system,
        TreeLSTMStateBuilder & state_builder,
        const std::string & policy_name);

  dynet::expr::Expression reinforce(dynet::ComputationGraph & cg,
                                    const std::vector<dynet::expr::Expression> & input,
                                    std::vector<dynet::expr::Expression> & trans_logits,
                                    std::vector<unsigned>& actions,
                                    bool train);

  dynet::expr::Expression decode(dynet::ComputationGraph & cg,
                                 const std::vector<dynet::expr::Expression> & input,
                                 std::vector<dynet::expr::Expression> & trans_logits,
                                 std::vector<unsigned> & actions,
                                 State & state,
                                 bool train);

  dynet::expr::Expression decode(dynet::ComputationGraph & cg,
                                 const std::vector<dynet::expr::Expression> & input,
                                 State & state,
                                 bool train);

  dynet::expr::Expression decode(dynet::ComputationGraph & cg,
                                 const std::vector<dynet::expr::Expression> & input,
                                 std::vector<dynet::expr::Expression> & trans_logits,
                                 std::vector<unsigned> & actions,
                                 bool train);

  dynet::expr::Expression decode(dynet::ComputationGraph & cg,
                                 const std::vector<dynet::expr::Expression> & input,
                                 bool train);

  dynet::expr::Expression left(dynet::ComputationGraph & cg,
                               const std::vector<dynet::expr::Expression> & input);

  dynet::expr::Expression left(dynet::ComputationGraph & cg,
                               const std::vector<dynet::expr::Expression> & input,
                               State & state);

  dynet::expr::Expression left(dynet::ComputationGraph & cg,
                               const std::vector<dynet::expr::Expression> & input,
                               std::vector<dynet::expr::Expression> & trans_logits,
                               std::vector<unsigned> & actions,
                               bool train);

  dynet::expr::Expression left(dynet::ComputationGraph & cg,
                               const std::vector<dynet::expr::Expression> & input,
                               std::vector<dynet::expr::Expression> & trans_logits,
                               std::vector<unsigned> & actions,
                               State & state,
                               bool train);

  dynet::expr::Expression right(dynet::ComputationGraph & cg,
                                const std::vector<dynet::expr::Expression> & input);

  dynet::expr::Expression right(dynet::ComputationGraph & cg,
                                const std::vector<dynet::expr::Expression> & input,
                                State & state);

  dynet::expr::Expression right(dynet::ComputationGraph & cg,
                                const std::vector<dynet::expr::Expression> & input,
                                std::vector<dynet::expr::Expression> & trans_logits,
                                std::vector<unsigned> & actions,
                                State & state,
                                bool train);

  dynet::expr::Expression right(dynet::ComputationGraph & cg,
                                const std::vector<dynet::expr::Expression> & input,
                                std::vector<dynet::expr::Expression> & trans_logits,
                                std::vector<unsigned> & actions,
                                bool train);

  dynet::expr::Expression execute(dynet::ComputationGraph & cg,
                                  const std::vector<dynet::expr::Expression> & input,
                                  const std::vector<unsigned> & actions,
                                  std::vector<dynet::expr::Expression> & trans_logits,
                                  State & state,
                                  bool train);

  dynet::expr::Expression execute(dynet::ComputationGraph & cg,
                                  const std::vector<dynet::expr::Expression> & input,
                                  const std::vector<unsigned> & actions,
                                  State & state,
                                  bool train);

  dynet::expr::Expression execute(dynet::ComputationGraph & cg,
                                  const std::vector<dynet::expr::Expression> & input,
                                  const std::vector<unsigned> & actions,
                                  std::vector<dynet::expr::Expression> & trans_logits,
                                  bool train);

  dynet::expr::Expression execute(dynet::ComputationGraph & cg,
                                  const std::vector<dynet::expr::Expression> & input,
                                  const std::vector<unsigned> & actions,
                                  bool train);

  void set_policy(const std::string & policy_name);

  void set_policy(const POLICY_TYPE & policy_type_);

  virtual dynet::expr::Expression get_policy_logits(TreeLSTMState * machine,
                                                    const State & state,
                                                    bool train) = 0;
};

#endif  //  end for MODEL_H