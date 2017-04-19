#include "sst_corpus.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "logging.h"
#include <boost/assert.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

SSTCorpus::SSTCorpus() : Corpus(), n_classes(0) {
}

void SSTCorpus::load_training_data(const std::string& filename, bool allow_new_token) {
  _INFO << "Corpus:: reading training data from: " << filename;
  word_map.insert(Corpus::BAD0);
  word_map.insert(Corpus::UNK);

  std::ifstream in(filename);
  BOOST_ASSERT_MSG(in, "Corpus:: failed to open the training file.");

  n_train = 0;
  std::string line;
  while (std::getline(in, line)) {
    boost::algorithm::trim(line);
    parse_data(line, training_instances[n_train], true, allow_new_token);
    ++n_train;
  }
  _INFO << "Corpus:: loaded " << n_train << " training sentences.";
  _INFO << "Corpus:: # word in alphabet is " << word_map.size() << " after loading training";
}

void SSTCorpus::load_devel_data(const std::string& filename) {
  _INFO << "Corpus:: reading development data from: " << filename;
  BOOST_ASSERT_MSG(word_map.size() > 1,
                   "Corpus:: BAD0 and UNK should be inserted before loading devel data.");

  std::ifstream in(filename);
  BOOST_ASSERT_MSG(in, "Corpus:: failed to open the devel file.");

  n_devel = 0;
  std::string line;
  while (std::getline(in, line)) {
    boost::algorithm::trim(line);
    parse_data(line, devel_instances[n_devel], false, false);
    ++n_devel;
  }
  _INFO << "Corpus:: loaded " << n_devel << " development sentences.";
  _INFO << "Corpus:: # word in alphabet is " << word_map.size() << " after loading development.";
}

void SSTCorpus::load_test_data(const std::string& filename) {
  _INFO << "Corpus:: reading test data from: " << filename;
  BOOST_ASSERT_MSG(word_map.size() > 1,
                   "Corpus:: BAD0 and UNK should be inserted before loading devel data.");

  std::ifstream in(filename);
  BOOST_ASSERT_MSG(in, "Corpus:: failed to open the devel file.");

  n_test = 0;
  std::string line;
  while (std::getline(in, line)) {
    boost::algorithm::trim(line);
    parse_data(line, test_instances[n_test], false, false);
    ++n_test;
  }
  _INFO << "Corpus:: loaded " << n_test << " test sentences.";
  _INFO << "Corpus:: # word in alphabet is " << word_map.size() << " after loading test.";
}

void SSTCorpus::parse_data(const std::string& data, SSTInstance& input,
                           bool allow_new_class, bool allow_new_token) {
  // json format.
  unsigned p = data.find('\t');
  std::string label_name = data.substr(0, p);
  input.label = boost::lexical_cast<unsigned>(label_name); // 0 - 4
  if (allow_new_class) {
    if (input.label >= n_classes) { n_classes = input.label + 1; }
  } else {
    assert(input.label < n_classes);
  }
  std::vector<std::string> tokens;
  std::string sentence_str = data.substr(p);
  boost::algorithm::split(tokens, sentence_str, boost::is_any_of(" "), boost::token_compress_on);

  input.sentence.clear();
  for (auto& token : tokens) {
    boost::replace_all(token, "\\", "");
    input.sentence.push_back(
      allow_new_token ?
      word_map.insert(token) :
      (word_map.contains(token) ? word_map.get(token) : word_map.get(Corpus::UNK))
    );
  }
}
