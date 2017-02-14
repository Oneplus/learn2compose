#ifndef SYSTEM_H
#define SYSTEM_H

#include <vector>

struct State {
  State(unsigned n_) : n(n_), nid(n_), beta(0) {}

  std::vector<unsigned> sigma;
  unsigned n;
  unsigned nid;
  unsigned beta;
};

struct TransitionSystem {
  const static unsigned n_actions = 2;
  static bool is_shift(const unsigned& action) { return action == 0; }
  static bool is_reduce(const unsigned& action) { return action == 1; }
  static unsigned get_shift_id() { return 0; }
  static unsigned get_reduce_id() { return 1; }
  static bool is_terminated(const State & state) {
    return (state.sigma.size() == 1 && state.beta >= state.n);
  }
  static bool is_valid(State & state, const unsigned & action);
  static void shift(State & state);
  static void reduce(State & state);
};

#endif  //  end for SYSTEM