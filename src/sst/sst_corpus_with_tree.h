#ifndef SST_CORPUS_WITH_TREE_H
#define SST_CORPUS_WITH_TREE_H

#include <unordered_map>
#include <vector>
#include <set>
#include "corpus.h"
#include "alphabet.h"

struct SSTInstanceWithTree {
  unsigned label;
  std::vector<unsigned> sentence;
  std::vector<unsigned> parents;
  std::vector<unsigned> labels;
};

struct SSTCorpusWithTree : public Corpus {
  unsigned n_classes;
  std::unordered_map<unsigned, SSTInstanceWithTree> training_instances;
  std::unordered_map<unsigned, SSTInstanceWithTree> devel_instances;
  std::unordered_map<unsigned, SSTInstanceWithTree> test_instances;

  SSTCorpusWithTree();

  void load_training_data(const std::string& filename, bool dependency,
                          bool allow_new_token);
  void load_devel_data(const std::string& filename, bool dependency);
  void load_test_data(const std::string& filename, bool dependency);
  void parse_data(const std::string& data, SSTInstanceWithTree & inst,
                  bool dependency, bool allow_new_class, bool allow_new_token);
  void get_span(const std::vector<unsigned> & parents, unsigned now,
                std::vector<std::pair<unsigned, unsigned>> & span);
  void get_order(const std::vector<unsigned> & parents, unsigned now,
                 const std::vector<std::pair<unsigned, unsigned>> & span,
                 std::vector<unsigned> & order, unsigned & counter);

};

#endif  //  end for SST_CORPUS_WITH_TREE_H