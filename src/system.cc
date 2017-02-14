#include "system.h"

bool TransitionSystem::is_valid(State & state, const unsigned & action) {
  if (is_shift(action)) {
    return (state.beta < state.n);
  } else { 
    return (state.sigma.size() > 1);
  }
  return false;
}

void TransitionSystem::shift(State & state) {
  state.sigma.push_back(state.beta);
  state.beta++;
}

void TransitionSystem::reduce(State & state) {
  state.sigma.pop_back();
  state.sigma.pop_back();
  state.sigma.push_back(state.nid);
  state.nid++;
}
