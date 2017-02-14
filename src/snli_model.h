#ifndef SNLI_MODEL_H
#define SNLI_MODEL_H 

#include "layer.h"
#include "system.h"
#include "reinforce.h"
#include "snli_corpus.h"
#include "dynet/expr.h"

struct SNLIModel : public Reinforce {
  const static unsigned n_classes = 3;
  Merge3Layer policy_merger;
  DenseLayer policy_scorer;
  Merge4Layer classifier_merger;
  DenseLayer classifier_scorer;
  SymbolEmbedding word_emb;
  dynet::Parameter p_sigma_guard_j;
  dynet::Parameter p_sigma_guard_i;
  dynet::Parameter p_beta_guard;
  dynet::expr::Expression zero_padding;
  dynet::expr::Expression sigma_guard_j;
  dynet::expr::Expression sigma_guard_i;
  dynet::expr::Expression beta_guard;
  unsigned word_dim;
  std::vector<TreeLSTMCell> stack1;
  std::vector<TreeLSTMCell> stack2;

  SNLIModel(dynet::Model & m,
            unsigned word_size,
            unsigned word_dim,
            unsigned hidden_dim,
            const std::unordered_map<unsigned, std::vector<float>>& embeddings);

  void new_graph_impl(dynet::ComputationGraph & cg) override;

  dynet::expr::Expression reinforce(dynet::ComputationGraph & cg,
                                    const SNLIInstance & inst);

  unsigned predict(const SNLIInstance & inst);

  dynet::expr::Expression get_policy_logits(const State & state,
                                            const SNLIInstance & inst);
};

#endif  //  end for CLASSIFIER_H