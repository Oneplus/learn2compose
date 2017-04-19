#include "sst_model_i.h"

SSTModelI::SSTModelI(TransitionSystem & system,
                     TreeLSTMStateBuilder & state_builder,
                     const std::string & policy_name) :
  Model(system, state_builder, policy_name) {
}
