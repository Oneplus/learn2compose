#ifndef SNLI_MODEL_H
#define SNLI_MODEL_H 

#include "layer.h"
#include "system.h"
#include "treelstm.h"
#include "snli_corpus.h"
#include "dynet/expr.h"

struct SNLIModel {
  enum POLICY_TYPE { kSample, kRight };
  POLICY_TYPE policy_type;
  TreeLSTMStateBuilder & state_builder;
  TransitionSystem & system;

  DenseLayer policy_projector;
  DenseLayer policy_scorer;
  Merge4Layer classifier_merger;
  DenseLayer classifier_scorer;
  SymbolEmbedding word_emb;
  unsigned word_dim;
  unsigned n_actions;
  unsigned n_classes;

  SNLIModel(unsigned word_size,
            unsigned word_dim,
            unsigned hidden_dim,
            unsigned n_classes,
            TransitionSystem & system,
            TreeLSTMStateBuilder & state_builder,
            const Embeddings & embeddings,
            const std::string & policy_name);

  void new_graph(dynet::ComputationGraph & cg);

  dynet::expr::Expression reinforce(dynet::ComputationGraph & cg,
                                    const SNLIInstance & inst);

  dynet::expr::Expression rollin(dynet::ComputationGraph & cg,
                                 const std::vector<unsigned> & sentence,
                                 std::vector<dynet::expr::Expression> & probs);

  dynet::expr::Expression decode(dynet::ComputationGraph & cg,
                                 const std::vector<unsigned> & sentence);

  unsigned predict(const SNLIInstance & inst);

  dynet::expr::Expression get_policy_logits(TreeLSTMState * machine);

  dynet::expr::Expression get_classifier_logits(dynet::expr::Expression & s1,
                                                dynet::expr::Expression & s2);
};

#endif  //  end for CLASSIFIER_H