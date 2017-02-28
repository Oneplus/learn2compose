#include "corpus.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "logging.h"
#include <boost/assert.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/property_tree/json_parser.hpp>

const char* Corpus::UNK = "_UNK_";
const char* Corpus::BAD0 = "_BAD0_";

Corpus::Corpus() : n_train(0), n_devel(0), n_test(0) {
}

unsigned Corpus::get_or_add_word(const std::string& word) {
  return word_map.insert(word);
}

void load_word_embeddings(const std::string& embedding_file,
                          unsigned dim,
                          Embeddings & embeddings,
                          Corpus& corpus) {
  embeddings[corpus.get_or_add_word(Corpus::BAD0)] = std::vector<float>(dim, 0.);
  embeddings[corpus.get_or_add_word(Corpus::UNK)] = std::vector<float>(dim, 0.);
  _INFO << "Main:: Loading from " << embedding_file << " with " << dim << " dimensions.";
  std::ifstream ifs(embedding_file);
  BOOST_ASSERT_MSG(ifs, "Failed to load embedding file.");
  std::string line;
  // get the header in word2vec styled embedding.
  std::getline(ifs, line);
  std::vector<float> v(dim, 0.);
  std::string word;
  while (std::getline(ifs, line)) {
    std::istringstream iss(line);
    iss >> word;
    // actually, there should be a checking about the embedding dimension.
    for (unsigned i = 0; i < dim; ++i) { iss >> v[i]; }
    unsigned id = corpus.get_or_add_word(word);
    embeddings[id] = v;
  }
}
