#ifndef TRAIN_UTILS_H
#define TRAIN_UTILS_H

#include <iostream>
#include <set>
#include <boost/program_options.hpp>
#include "corpus.h"
#include "model.h"
#include "dynet/model.h"
#include "dynet/training.h"

namespace po = boost::program_options;

po::options_description get_optimizer_options();

dynet::Trainer* get_trainer(const po::variables_map& conf,
                            dynet::Model& model);

void get_objective_sequence(std::string expr,
                            unsigned max_iter,
                            std::vector<Model::Param> & seq);

void get_objective_sequence_parse_loop(const std::string & expr,
                                       unsigned max_iter,
                                       std::vector<Model::Param> & seq);

void get_objective_sequence_parse_one_param(const std::string & expr,
                                            unsigned & time,
                                            Model::POLICY_TYPE & policy_type,
                                            Model::OBJECTIVE_TYPE & objective_type);

void update_trainer(const po::variables_map& conf,
                    dynet::Trainer* trainer);

#endif  //  end for TRAIN_H