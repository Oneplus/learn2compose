#include "evaluate.h"
#include <fstream>

float evaluate(SSTModelI & engine,
               SSTCorpus & corpus,
               bool devel,
               const std::string & tempfile) {
  unsigned n = devel ? corpus.n_devel : corpus.n_test;
  unsigned n_corr = 0;
  std::ofstream ofs(tempfile);
  for (unsigned i = 0; i < n; ++i) {
    SSTInstance & inst = (devel ? corpus.devel_instances[i] : corpus.test_instances[i]);
    State state(inst.sentence.size());
    unsigned result = engine.predict(inst, state);
    ofs << "gold: " << inst.label << " " << "pred: " << result << std::endl;
    engine.system.print_tree(state, ofs);
    if (result == inst.label) { n_corr++; }
  }

  return float(n_corr) / n;
}