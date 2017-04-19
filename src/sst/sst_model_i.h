#ifndef SST_MODEL_I_H
#define SST_MODEL_I_H

#include "model.h"
#include "sst_corpus.h"
#include "dynet/expr.h"
#include "dynet/dynet.h"

struct SSTModelI : public Model {
  SSTModelI(TransitionSystem & system,
            TreeLSTMStateBuilder & state_builder,
            const std::string & policy_name);

  virtual unsigned predict(const SSTInstance & inst, State & state) = 0;
};

#endif  //  end for SST_MODEL_I_H