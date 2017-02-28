#ifndef YELP_MODEL_H
#define YELP_MODEL_H

#include "layer.h"
#include "system.h"
#include "treelstm.h"
#include "model.h"
#include "yelp_corpus.h"
#include "dynet/expr.h"

struct YelpModel : public Model {
  DenseLayer policy_projector;
  DenseLayer policy_scorer;
  DenseLayer classifier_projector;
  DenseLayer classifier_scorer;
  SymbolEmbedding word_emb;

  unsigned n_actions;
  unsigned n_classes;
  unsigned word_dim;

  YelpModel(unsigned word_size,
            unsigned word_dim,
            unsigned hidden_dim,
            unsigned n_classes,
            TransitionSystem & system,
            TreeLSTMStateBuilder & state_builder,
            const Embeddings & embeddings,
            const std::string & policy_name);

  void new_graph(dynet::ComputationGraph & cg);

  dynet::expr::Expression objective(dynet::ComputationGraph & cg,
                                    const YelpInstance & inst,
                                    OBJECTIVE_TYPE objective_type = Model::kBothPolicyAndReward);

  unsigned predict(const YelpInstance & inst);

  dynet::expr::Expression get_policy_logits(TreeLSTMState * machine);

  dynet::expr::Expression get_classifier_logits(dynet::expr::Expression repr);

  dynet::expr::Expression sentence_expr(const YelpInstance & inst, unsigned sid);
};

#endif  //  end for YELP_MODEL_H