#ifndef SST_MODEL_H
#define SST_MODEL_H

#include "layer.h"
#include "system.h"
#include "reinforce.h"
#include "sst_corpus.h"
#include "dynet/expr.h"

struct SSTModel : public Reinforce {
  Merge3Layer policy_merger;
  DenseLayer policy_scorer;
  DenseLayer classifier_projector;
  DenseLayer classifier_scorer;
  SymbolEmbedding word_emb;
  dynet::Parameter p_sigma_guard_j;
  dynet::Parameter p_sigma_guard_i;
  dynet::Parameter p_beta_guard;
  dynet::expr::Expression zero_padding;
  dynet::expr::Expression sigma_guard_j;
  dynet::expr::Expression sigma_guard_i;
  dynet::expr::Expression beta_guard; 
  unsigned n_classes;
  unsigned word_dim;

  std::vector<TreeLSTMCell> stack;

  SSTModel(dynet::Model & m,
           unsigned word_size,
           unsigned word_dim,
           unsigned hidden_dim,
           unsigned n_classes,
           const std::unordered_map<unsigned, std::vector<float>>& embeddings);

  void new_graph_impl(dynet::ComputationGraph & cg) override;

  dynet::expr::Expression reinforce(dynet::ComputationGraph & cg,
                                    const SSTInstance & inst);

  unsigned predict(const SSTInstance & inst);

  dynet::expr::Expression get_policy_logits(const State & state,
                                            const SSTInstance & inst);
};

#endif  //  end for SST_MODEL_H