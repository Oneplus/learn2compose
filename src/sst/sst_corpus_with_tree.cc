#include "sst_corpus_with_tree.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "logging.h"
#include <boost/assert.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

SSTCorpusWithTree::SSTCorpusWithTree() : Corpus(), n_classes(0) {
}

void SSTCorpusWithTree::load_training_data(const std::string& filename,
                                           bool dependency, 
                                           bool allow_new_token) {
  _INFO << "Corpus:: reading training data from: " << filename;
  word_map.insert(Corpus::BAD0);
  word_map.insert(Corpus::UNK);

  std::ifstream in(filename);
  BOOST_ASSERT_MSG(in, "Corpus:: failed to open the training file.");

  n_train = 0;
  std::string line;
  while (std::getline(in, line)) {
    boost::trim(line);
    parse_data(line, training_instances[n_train], dependency, true, allow_new_token);
    ++n_train;     
  }
  _INFO << "Corpus:: loaded " << n_train << " training sentences.";
  _INFO << "Corpus:: # word in alphabet is " << word_map.size() << " after loading training";
}

void SSTCorpusWithTree::load_devel_data(const std::string& filename, bool dependency) {
  _INFO << "Corpus:: reading development data from: " << filename;
  BOOST_ASSERT_MSG(word_map.size() > 1,
                   "Corpus:: BAD0 and UNK should be inserted before loading devel data.");

  std::ifstream in(filename);
  BOOST_ASSERT_MSG(in, "Corpus:: failed to open the devel file.");

  n_devel = 0;
  std::string line;
  while (std::getline(in, line)) {
    boost::algorithm::trim(line);
    parse_data(line, devel_instances[n_devel], dependency, false, false);
    ++n_devel;
  }
  _INFO << "Corpus:: loaded " << n_devel << " development sentences.";
  _INFO << "Corpus:: # word in alphabet is " << word_map.size() << " after loading development.";
}

void SSTCorpusWithTree::load_test_data(const std::string& filename, bool dependency) {
  _INFO << "Corpus:: reading test data from: " << filename;
  BOOST_ASSERT_MSG(word_map.size() > 1,
                   "Corpus:: BAD0 and UNK should be inserted before loading devel data.");

  std::ifstream in(filename);
  BOOST_ASSERT_MSG(in, "Corpus:: failed to open the devel file.");

  n_test = 0;
  std::string line;
  while (std::getline(in, line)) {
    boost::algorithm::trim(line);
    parse_data(line, test_instances[n_test], dependency, false, false);
    ++n_test;
  }
  _INFO << "Corpus:: loaded " << n_test << " test sentences.";
  _INFO << "Corpus:: # word in alphabet is " << word_map.size() << " after loading test.";
}

void SSTCorpusWithTree::parse_data(const std::string& data,
                                   SSTInstanceWithTree& input,
                                   bool dependency,
                                   bool allow_new_class,
                                   bool allow_new_token) {
  // data in the format of {class}\t{sentence}\t{tree}
  unsigned p = data.find('\t');
  std::string label_name = data.substr(0, p);
  input.label = boost::lexical_cast<unsigned>(label_name); // 0 - 4
  if (allow_new_class) {
    if (input.label >= n_classes) { n_classes = input.label + 1; }
  } else {
    assert(input.label < n_classes);
  }
  std::vector<std::string> tokens;
  std::string rest = data.substr(p + 1);
  p = rest.find('\t');
  assert(p != std::string::npos);
  std::string sentence_str = rest.substr(0, p);
  boost::split(tokens, sentence_str, boost::is_any_of(" "), boost::token_compress_on);

  input.sentence.clear();
  for (auto& token : tokens) {
    boost::replace_all(token, "\\", "");
    input.sentence.push_back(
      allow_new_token ?
      word_map.insert(token) :
      (word_map.contains(token) ? word_map.get(token) : word_map.get(Corpus::UNK))
    );
  }
  
  input.parents.clear();
  std::string tree_str = rest.substr(p + 1);
  tokens.clear();
  boost::split(tokens, tree_str, boost::is_any_of(" "), boost::token_compress_on);
  for (auto & token : tokens) {
    unsigned p = token.find_first_of(':');
    unsigned parent, label;
    if (p != std::string::npos) {
      parent = boost::lexical_cast<unsigned>(token.substr(0, p));
      label = boost::lexical_cast<unsigned>(token.substr(p + 1));
      if (allow_new_class) {
        if (label >= n_classes) { n_classes = input.label + 1; }
      } else {
        assert(input.label < n_classes);
      }
    } else {
      parent = boost::lexical_cast<unsigned>(token);
      label = UINT_MAX;
    }
    if (parent == 0) { parent = UINT_MAX; } else { parent--; }
    input.parents.push_back(parent);
    input.labels.push_back(label);
  }
  if (dependency) {
    assert(input.parents.size() == input.sentence.size());
  } else {
    assert(input.parents.size() > input.sentence.size());
    std::vector<std::pair<unsigned, unsigned>> span(input.parents.size());
    unsigned root = input.parents.size() - 1;
    get_span(input.parents, root, span);
    unsigned counter = input.sentence.size();
    std::vector<unsigned> order(input.parents.size());
    get_order(input.parents, root, span, order, counter);
    std::vector<unsigned> new_parents(input.parents.size());
    std::vector<unsigned> new_labels(input.parents.size());
    for (unsigned i = 0; i < input.parents.size(); ++i) {
      new_parents[order[i]] = (input.parents[i] == UINT_MAX ? UINT_MAX : order[input.parents[i]]);
      new_labels[order[i]] = input.labels[i];
    }
    input.parents = new_parents;
    input.labels = new_labels;
  }
}

void SSTCorpusWithTree::get_span(const std::vector<unsigned>& parents,
                                 unsigned now,
                                 std::vector<std::pair<unsigned, unsigned>> & span) {
  unsigned left = UINT_MAX, right = UINT_MAX;
  for (unsigned i = 0; i < parents.size(); ++i) {
    unsigned parent = parents[i];
    if (parent == now) {
      if (left == UINT_MAX) { left = i; }
      else if (right == UINT_MAX) { right = i; }
      else { assert(false); }
    }
  }
  if (left == UINT_MAX && right == UINT_MAX) {
    span[now].first = now;
    span[now].second = now;
  } else if (left != UINT_MAX && right != UINT_MAX) {
    get_span(parents, left, span);
    get_span(parents, right, span);
    span[now].first = std::min(span[left].first, span[right].first);
    span[now].second = std::max(span[left].second, span[right].second);
  } else {
    assert(false);
  }
}

void SSTCorpusWithTree::get_order(const std::vector<unsigned>& parents,
                                  unsigned now,
                                  const std::vector<std::pair<unsigned, unsigned>> & span,
                                  std::vector<unsigned>& order,
                                  unsigned & counter) {
  unsigned left = UINT_MAX, right = UINT_MAX;
  for (unsigned i = 0; i < parents.size(); ++i) {
    unsigned parent = parents[i];
    if (parent == now) {
      if (left == UINT_MAX) { left = i; } 
      else if (right == UINT_MAX) { right = i; }
      else { assert(false); }
    }
  }
  if (left == UINT_MAX && right == UINT_MAX) {
    order[now] = now;
  } else {
    get_order(parents, (span[left].first < span[right].first ? left : right), span, order, counter);
    get_order(parents, (span[left].first < span[right].first ? right : left), span, order, counter);
    order[now] = counter++;
  }
}
