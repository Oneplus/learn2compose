#ifndef SST_EVALUATE_H
#define SST_EVALUATE_H

#include "sst_model_i.h"
#include "sst_corpus.h"

float evaluate(SSTModelI & engine,
               SSTCorpus & corpus,
               bool devel,
               const std::string & tempfile);

#endif  //  end for SST_EVALUATE_H