#ifndef YELP_MODEL_H
#define YELP_MODEL_H

#include "layer.h"
#include "system.h"
#include "treelstm.h"
#include "model.h"
#include "yelp_corpus.h"
#include "dynet/expr.h"
#include "dynet/gru.h"

struct YelpAvgPipeL2CModel : public Model {
  DenseLayer policy_projector;
  DenseLayer policy_scorer;
  DenseLayer classifier_projector;
  DenseLayer classifier_scorer;
  SymbolEmbedding word_emb;

  unsigned n_actions;
  unsigned n_classes;
  unsigned word_dim;

  YelpAvgPipeL2CModel(unsigned word_size,
                      unsigned word_dim,
                      unsigned hidden_dim,
                      unsigned n_classes,
                      TransitionSystem & system,
                      TreeLSTMStateBuilder & state_builder,
                      const Embeddings & embeddings,
                      const std::string & policy_name,
                      bool tune_embedding);

  virtual void new_graph(dynet::ComputationGraph & cg);

  dynet::expr::Expression objective(dynet::ComputationGraph & cg,
                                    const YelpInstance & inst,
                                    OBJECTIVE_TYPE objective_type = Model::kBothPolicyAndReward);

  unsigned predict(const YelpInstance & inst);

  unsigned predict(const YelpInstance & inst, State & state);

  dynet::expr::Expression get_policy_logits(TreeLSTMState * machine,
                                            const State & state) override;

  dynet::expr::Expression get_classifier_logits(dynet::expr::Expression repr);

  virtual dynet::expr::Expression sentence_expr(const YelpInstance & inst, unsigned sid);
};

struct YelpBiGRUPipeL2CModel : public YelpAvgPipeL2CModel {
  dynet::GRUBuilder fwd_gru;
  dynet::GRUBuilder bwd_gru;

  YelpBiGRUPipeL2CModel(unsigned word_size,
                        unsigned word_dim,
                        unsigned hidden_dim,
                        unsigned n_classes,
                        TransitionSystem & system,
                        TreeLSTMStateBuilder & state_builder,
                        const Embeddings & embeddings,
                        const std::string & policy_name,
                        bool tune_embedding);

  void new_graph(dynet::ComputationGraph & cg);

  dynet::expr::Expression sentence_expr(const YelpInstance & inst, unsigned sid);
};

struct YelpBiGRUPipeL2CModelBatch : public YelpAvgPipeL2CModel {
  dynet::GRUBuilder fwd_gru;
  dynet::GRUBuilder bwd_gru;
  std::vector<dynet::expr::Expression> sentence_expr_cache;

  YelpBiGRUPipeL2CModelBatch(unsigned word_size,
                             unsigned word_dim,
                             unsigned hidden_dim,
                             unsigned n_classes,
                             TransitionSystem & system,
                             TreeLSTMStateBuilder & state_builder,
                             const Embeddings & embeddings,
                             const std::string & policy_name,
                             bool tune_embedding);

  void new_graph(dynet::ComputationGraph & cg);

  dynet::expr::Expression sentence_expr(const YelpInstance & inst, unsigned sid);
};

#endif  //  end for YELP_MODEL_H