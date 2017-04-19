#ifndef CORPUS_H
#define CORPUS_H

#include <unordered_map>
#include <vector>
#include <set>
#include "alphabet.h"

struct Corpus {
  const static char* UNK;
  const static char* BAD0;

  unsigned n_train;
  unsigned n_devel;
  unsigned n_test;

  Alphabet word_map;
  Corpus();

  unsigned get_or_add_word(const std::string& word);
};

typedef std::unordered_map<unsigned, std::vector<float>> Embeddings;

void load_word_embeddings(const std::string& embedding_file,
                          unsigned dim,
                          Embeddings & embeddings,
                          Corpus& corpus,
                          bool set_zero_to_bad0 = false,
                          bool set_zero_to_unk = false);

#endif  //  end for CORPUS_H
