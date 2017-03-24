#ifndef SST_CORPUS_H
#define SST_CORPUS_H

#include <unordered_map>
#include <vector>
#include <set>
#include "corpus.h"
#include "alphabet.h"

struct SSTInstance {
  unsigned label;
  std::vector<unsigned> sentence;
};

struct SSTCorpus : public Corpus {
  unsigned n_classes;
  std::unordered_map<unsigned, SSTInstance> training_instances;
  std::unordered_map<unsigned, SSTInstance> devel_instances;
  std::unordered_map<unsigned, SSTInstance> test_instances;

  SSTCorpus();

  void load_training_data(const std::string& filename, bool allow_new_token);
  void load_devel_data(const std::string& filename);
  void load_test_data(const std::string& filename);
  void parse_data(const std::string& data, SSTInstance & inst,
                  bool allow_new_class, bool allow_new_token);
};

#endif  //  end for SST_CORPUS_H