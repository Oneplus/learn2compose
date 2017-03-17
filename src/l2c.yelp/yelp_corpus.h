#ifndef YELP_CORPUS_H
#define YELP_CORPUS_H

#include <unordered_map>
#include <vector>
#include <set>
#include "corpus.h"
#include "alphabet.h"

struct YelpInstance {
  unsigned label;
  std::vector<std::vector<unsigned>> document;
};

struct YelpCorpus : public Corpus {
  unsigned n_classes;
  std::unordered_map<unsigned, YelpInstance> training_instances;
  std::unordered_map<unsigned, YelpInstance> devel_instances;
  std::unordered_map<unsigned, YelpInstance> test_instances;

  YelpCorpus();

  void load_training_data(const std::string& filename);
  void load_training_data(const std::string& filename, bool allow_new_token);
  void load_devel_data(const std::string& filename);
  void load_test_data(const std::string& filename);
  void parse_data(const std::string& data, YelpInstance & inst,
                  bool allow_new_token, bool allow_new_class);
};

#endif  //  end for SST_CORPUS_H