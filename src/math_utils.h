#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <vector>

void mean_and_stddev(const std::vector<float>& data,
                     float & mean,
                     float & stddev);

void softmax_inplace(std::vector<float>& x);

#endif  //  end for MATH_UTILS_H