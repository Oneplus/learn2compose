#include "math_utils.h"
#include "math_utils.h"
#include <cmath>
#include <boost/assert.hpp>

void mean_and_stddev(const std::vector<float>& data,
                     float & mean,
                     float & stddev) {
  float n = 0.;
  float sum1 = 0., sum2 = 0.;
  for (auto x : data) { sum1 += x; n += 1.; }
  mean = sum1 / n;
  for (auto x : data) { sum2 += (x - mean) * (x - mean); }
  stddev = sqrt(sum2 / (n - 1));
}

void softmax_inplace(std::vector<float>& x) {
  BOOST_ASSERT_MSG(x.size() > 0, "input should have one or more element.");
  float m = x[0];
  for (const float& _x : x) { m = (_x > m ? _x : m); }
  float s = 0.;
  for (unsigned i = 0; i < x.size(); ++i) {
    x[i] = exp(x[i] - m);
    s += x[i];
  }
  for (unsigned i = 0; i < x.size(); ++i) { x[i] /= s; }
}
