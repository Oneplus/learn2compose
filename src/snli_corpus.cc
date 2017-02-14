#include "snli_corpus.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "logging.h"
#include <boost/assert.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/property_tree/json_parser.hpp>

SNLICorpus::SNLICorpus() : Corpus() {
}

void SNLICorpus::load_training_data(const std::string& filename) {
  _INFO << "Corpus:: reading training data from: " << filename;
  word_map.insert(Corpus::BAD0);
  word_map.insert(Corpus::UNK);

  std::ifstream in(filename);
  BOOST_ASSERT_MSG(in, "Corpus:: failed to open the training file.");

  n_train = 0;
  std::string line;
  while (std::getline(in, line)) {
    boost::algorithm::trim(line);
    if (parse_data(line, training_instances[n_train], true)) { ++n_train; }
  }
  _INFO << "Corpus:: loaded " << n_train << " training sentences.";
  _INFO << "Corpus:: # word in alphabet is " << word_map.size() << " after loading training";
}

void SNLICorpus::load_devel_data(const std::string& filename) {
  _INFO << "Corpus:: reading development data from: " << filename;
  BOOST_ASSERT_MSG(word_map.size() > 1,
    "Corpus:: BAD0 and UNK should be inserted before loading devel data.");

  std::ifstream in(filename);
  BOOST_ASSERT_MSG(in, "Corpus:: failed to open the devel file.");

  n_devel = 0;
  std::string line;
  while (std::getline(in, line)) {
    boost::algorithm::trim(line);
    if (parse_data(line, devel_instances[n_devel], false)) { ++n_devel; }
  }
  _INFO << "Corpus:: loaded " << n_devel << " development sentences.";
  _INFO << "Corpus:: # word in alphabet is " << word_map.size() << " after loading development.";
}

void SNLICorpus::load_test_data(const std::string& filename) {
  _INFO << "Corpus:: reading test data from: " << filename;
  BOOST_ASSERT_MSG(word_map.size() > 1,
    "Corpus:: BAD0 and UNK should be inserted before loading devel data.");

  std::ifstream in(filename);
  BOOST_ASSERT_MSG(in, "Corpus:: failed to open the devel file.");

  n_test = 0;
  std::string line;
  while (std::getline(in, line)) {
    boost::algorithm::trim(line);
    if (parse_data(line, test_instances[n_test], false)) { ++n_test; }
  }
  _INFO << "Corpus:: loaded " << n_test << " test sentences.";
  _INFO << "Corpus:: # word in alphabet is " << word_map.size() << " after loading test.";
}

bool SNLICorpus::parse_data(const std::string& data, SNLIInstance& input, bool train) {
  // json format.
  std::stringstream S(data);
  boost::property_tree::ptree pt;
  boost::property_tree::read_json(S, pt);

  std::string gold_label = pt.get_value("gold_label");
  if (gold_label != "neutral" && gold_label != "entailment" && gold_label != "contradiction") {
    return false;
  }
  input.label == (gold_label[0] == 'n' ? 0 : (gold_label[0] == 'e' ? 1 : 2));
  
  std::string sentence1 = pt.get_value("sentence1_binary_parse");
  std::replace(sentence1.begin(), sentence1.end(), '(', ' ');
  std::replace(sentence1.begin(), sentence1.end(), ')', ' ');
  std::vector <std::string> tokens;
  boost::algorithm::split(tokens, sentence1, boost::is_any_of(" "), boost::token_compress_on);
  input.sentence1.clear();
  for (const auto& token : tokens) {
    input.sentence1.push_back(
      train ? 
      word_map.insert(token) : 
      (word_map.contains(Corpus::UNK) ? word_map.get(token) : word_map.get(Corpus::UNK))
    );
  }

  std::string sentence2 = pt.get_value("sentence2_binary_parse");
  std::replace(sentence2.begin(), sentence2.end(), '(', ' ');
  std::replace(sentence2.begin(), sentence2.end(), ')', ' ');  
  boost::algorithm::split(tokens, sentence2, boost::is_any_of(" "), boost::token_compress_on);
  input.sentence2.clear();
  for (const auto& token : tokens) {
    input.sentence2.push_back(
      train ? 
      word_map.insert(token) : 
      (word_map.contains(Corpus::UNK) ? word_map.get(token) : word_map.get(Corpus::UNK))
    );
  }
  return true;
}
