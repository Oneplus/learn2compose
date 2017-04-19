#ifndef PG_SST_MODEL_H
#define PG_SST_MODEL_H

#include "sst_model_i.h"

struct PolicyGradientSSTModel : public SSTModelI {
  DenseLayer policy_projector;
  DenseLayer policy_scorer;
  DenseLayer classifier_scorer;
  SymbolEmbedding word_emb;
  float dropout;

  unsigned n_actions;
  unsigned n_classes;
  unsigned word_dim;

  PolicyGradientSSTModel(unsigned word_size,
                         unsigned word_dim,
                         unsigned hidden_dim,
                         unsigned n_classes,
                         TransitionSystem & system,
                         TreeLSTMStateBuilder & state_builder,
                         float dropout,
                         bool tune_embedding,
                         const Embeddings & embeddings,
                         const std::string & policy_name);

  void new_graph(dynet::ComputationGraph & cg);

  dynet::expr::Expression objective(dynet::ComputationGraph & cg,
                                    const SSTInstance & inst);

  void discount_rewards(std::vector<float> & rewards);

  unsigned predict(const SSTInstance & inst, State & state) override;

  dynet::expr::Expression get_policy_logits(TreeLSTMState * machine,
                                            const State & state,
                                            bool train) override;

  dynet::expr::Expression get_classifier_logits(dynet::expr::Expression repr,
                                                bool train);
};

#endif  //  end for PG_SST_MODEL_H