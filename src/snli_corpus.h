#ifndef SNLI_CORPUS_H
#define SNLI_CORPUS_H

#include <unordered_map>
#include <vector>
#include <set>
#include "corpus.h"
#include "alphabet.h"

struct SNLIInstance {
  unsigned label;
  std::vector<unsigned> sentence1;
  std::vector<unsigned> sentence2;
};

struct SNLICorpus : public Corpus {
  std::unordered_map<unsigned, SNLIInstance> training_instances;
  std::unordered_map<unsigned, SNLIInstance> devel_instances;
  std::unordered_map<unsigned, SNLIInstance> test_instances;

  SNLICorpus();

  void load_training_data(const std::string& filename);
  void load_devel_data(const std::string& filename);
  void load_test_data(const std::string& filename);
  bool parse_data(const std::string& data, SNLIInstance& input, bool train);
};

#endif  //  end for CORPUS_H
