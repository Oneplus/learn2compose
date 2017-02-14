#ifndef TRAIN_UTILS_H
#define TRAIN_UTILS_H

#include <iostream>
#include <set>
#include <boost/program_options.hpp>
#include "corpus.h"
#include "dynet/model.h"
#include "dynet/training.h"

namespace po = boost::program_options;

po::options_description get_optimizer_options();

dynet::Trainer* get_trainer(const po::variables_map& conf,
                            dynet::Model& model);

void update_trainer(const po::variables_map& conf,
                    dynet::Trainer* trainer);

#endif  //  end for TRAIN_H