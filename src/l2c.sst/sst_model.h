#ifndef SST_MODEL_H
#define SST_MODEL_H

#include "layer.h"
#include "system.h"
#include "treelstm.h"
#include "model.h"
#include "sst_corpus.h"
#include "dynet/expr.h"

struct SSTModel : public Model {
  DenseLayer policy_projector;
  DenseLayer policy_scorer;
  DenseLayer classifier_projector;
  DenseLayer classifier_scorer;
  SymbolEmbedding word_emb;

  unsigned n_actions;
  unsigned n_classes;
  unsigned word_dim;

  SSTModel(unsigned word_size,
           unsigned word_dim,
           unsigned hidden_dim,
           unsigned n_classes,
           TransitionSystem & system,
           TreeLSTMStateBuilder & state_builder,
           const Embeddings & embeddings,
           const std::string & policy_name);

  void new_graph(dynet::ComputationGraph & cg);

  dynet::expr::Expression objective(dynet::ComputationGraph & cg,
                                    const SSTInstance & inst,
                                    OBJECTIVE_TYPE objective_type = Model::kBothPolicyAndReward);

  unsigned predict(const SSTInstance & inst);

  dynet::expr::Expression get_policy_logits(TreeLSTMState * machine,
                                            const State & state) override;

  dynet::expr::Expression get_classifier_logits(dynet::expr::Expression repr);
};

#endif  //  end for SST_MODEL_H