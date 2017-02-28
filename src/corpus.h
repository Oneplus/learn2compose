#ifndef CORPUS_H
#define CORPUS_H

#include <unordered_map>
#include <vector>
#include <set>
#include "alphabet.h"

struct Instance {
  unsigned label;
  std::vector<unsigned> sentence1;
  std::vector<unsigned> sentence2;
};

struct Corpus {
  const static char* UNK;
  const static char* BAD0;

  unsigned n_train;
  unsigned n_devel;
  unsigned n_test;

  Alphabet word_map;
  Corpus();

  virtual void load_training_data(const std::string& filename) = 0;
  virtual void load_devel_data(const std::string& filename) = 0;
  virtual void load_test_data(const std::string& filename) = 0;
  unsigned get_or_add_word(const std::string& word);
};

typedef std::unordered_map<unsigned, std::vector<float>> Embeddings;

void load_word_embeddings(const std::string& embedding_file,
                          unsigned dim,
                          Embeddings & embeddings,
                          Corpus& corpus);

#endif  //  end for CORPUS_H
