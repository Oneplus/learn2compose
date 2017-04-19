#ifndef TREELSTM_SST_MODEL_H
#define TREELSTM_SST_MODEL_H

#include "layer.h"
#include "system.h"
#include "treelstm.h"
#include "sst_corpus_with_tree.h"
#include "sst_model_i.h"
#include "dynet/expr.h"

struct TreeLSTMSSTModel : public SSTModelI {
  DenseLayer classifier_scorer;
  SymbolEmbedding word_emb;
  float dropout;
  float l2;

  unsigned n_actions;
  unsigned n_classes;
  unsigned word_dim;

  TreeLSTMSSTModel(unsigned word_size,
                   unsigned word_dim,
                   unsigned hidden_dim,
                   unsigned n_classes,
                   TransitionSystem & system,
                   TreeLSTMStateBuilder & state_builder,
                   float dropout,
                   float l2,
                   bool tune_embedding,
                   const Embeddings & embeddings,
                   const std::string & policy_name);

  void new_graph(dynet::ComputationGraph & cg);

  dynet::expr::Expression objective(dynet::ComputationGraph & cg,
                                    const SSTInstanceWithTree & inst);
  
  unsigned predict(const SSTInstance& inst, State & state);
  
  unsigned predict(const SSTInstanceWithTree & inst);

  dynet::expr::Expression get_policy_logits(TreeLSTMState * machine,
                                            const State & state,
                                            bool train) override;

  dynet::expr::Expression get_classifier_logits(dynet::expr::Expression repr,
                                                bool train);
};

#endif  //  end for TREELSTM_SST_MODEL_H