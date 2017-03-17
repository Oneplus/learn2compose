#include "yelp_corpus.h"
#include "logging.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <boost/assert.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

YelpCorpus::YelpCorpus() : Corpus(), n_classes(0) {
}

void YelpCorpus::load_training_data(const std::string& filename) {
  _INFO << "Corpus:: reading training data from: " << filename;
  word_map.insert(Corpus::BAD0);
  word_map.insert(Corpus::UNK);

  std::ifstream in(filename);
  BOOST_ASSERT_MSG(in, "Corpus:: failed to open the training file.");

  n_train = 0;
  std::string data = "";
  std::string line;
  while (std::getline(in, line)) {
    boost::algorithm::trim(line);
    if (line.size() == 0) {
      // end for an instance.
      parse_data(data, training_instances[n_train], true, true);
      data = "";
      ++n_train;
    } else {
      data += (line + "\n");
    }
  }
  if (data.size() > 0) {
    parse_data(data, training_instances[n_train], true, true);
    ++n_train;
  }

  _INFO << "Corpus:: loaded " << n_train << " training sentences.";
  _INFO << "Corpus:: # word in alphabet is " << word_map.size() << " after loading training";
}

void YelpCorpus::load_training_data(const std::string& filename, bool allow_new_token) {
  _INFO << "Corpus:: reading training data from: " << filename;
  word_map.insert(Corpus::BAD0);
  word_map.insert(Corpus::UNK);

  std::ifstream in(filename);
  BOOST_ASSERT_MSG(in, "Corpus:: failed to open the training file.");

  n_train = 0;
  std::string data = "";
  std::string line;
  while (std::getline(in, line)) {
    boost::algorithm::trim(line);
    if (line.size() == 0) {
      // end for an instance.
      parse_data(data, training_instances[n_train], allow_new_token, true);
      data = "";
      ++n_train;
    } else {
      data += (line + "\n");
    }
  }
  if (data.size() > 0) {
    parse_data(data, training_instances[n_train], allow_new_token, true);
    ++n_train;
  }

  _INFO << "Corpus:: loaded " << n_train << " training sentences.";
  _INFO << "Corpus:: # word in alphabet is " << word_map.size() << " after loading training";
}

void YelpCorpus::load_devel_data(const std::string& filename) {
  _INFO << "Corpus:: reading development data from: " << filename;
  BOOST_ASSERT_MSG(word_map.size() > 1,
                   "Corpus:: BAD0 and UNK should be inserted before loading devel data.");

  std::ifstream in(filename);
  BOOST_ASSERT_MSG(in, "Corpus:: failed to open the devel file.");

  n_devel = 0;
  std::string data = "";
  std::string line;
  while (std::getline(in, line)) {
    boost::algorithm::trim(line);
    if (line.size() == 0) {
      // end for an instance.
      parse_data(data, devel_instances[n_devel], false, false);
      data = "";
      ++n_devel;
    } else {
      data += (line + "\n");
    }
  }
  if (data.size() > 0) {
    parse_data(data, devel_instances[n_devel], false, false);
    ++n_devel;
  }

  _INFO << "Corpus:: loaded " << n_devel << " development sentences.";
  _INFO << "Corpus:: # word in alphabet is " << word_map.size() << " after loading development.";
}

void YelpCorpus::load_test_data(const std::string& filename) {
  _INFO << "Corpus:: reading test data from: " << filename;
  BOOST_ASSERT_MSG(word_map.size() > 1,
                   "Corpus:: BAD0 and UNK should be inserted before loading devel data.");

  std::ifstream in(filename);
  BOOST_ASSERT_MSG(in, "Corpus:: failed to open the devel file.");

  n_test = 0;
  std::string data = "";
  std::string line;
  while (std::getline(in, line)) {
    boost::algorithm::trim(line);
    if (line.size() == 0) {
      // end for an instance.
      parse_data(data, test_instances[n_test], false, false);
      data = "";
      ++n_test;
    } else {
      data += (line + "\n");
    }
  }
  if (data.size() > 0) {
    parse_data(data, test_instances[n_test], false, false);
    ++n_test;
  }

  _INFO << "Corpus:: loaded " << n_test << " test sentences.";
  _INFO << "Corpus:: # word in alphabet is " << word_map.size() << " after loading test.";
}

void YelpCorpus::parse_data(const std::string& data, YelpInstance& input,
                            bool allow_new_token, bool allow_new_class) {
  // json format.
  std::stringstream S(data);
  std::string label_name;
  std::getline(S, label_name);
  boost::algorithm::trim(label_name);
  input.label = boost::lexical_cast<unsigned>(label_name); // 0 - 4
  if (allow_new_class) {
    if (input.label >= n_classes) { n_classes = input.label + 1; }
  } else {
    assert(input.label < n_classes);
  } 

  std::string line;
  while (std::getline(S, line)) {
    boost::algorithm::trim(line);
    std::vector<std::string> tokens;
    boost::algorithm::split(tokens, line, boost::is_any_of(" "), boost::token_compress_on);

    std::vector<unsigned> sentence;
    for (const auto & token : tokens) {
      sentence.push_back(
        allow_new_token ?
        word_map.insert(token) :
        (word_map.contains(token) ? word_map.get(token) : word_map.get(Corpus::UNK))
      );
    }
    input.document.push_back(sentence);
  }
}
