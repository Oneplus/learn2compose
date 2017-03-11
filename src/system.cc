#include "system.h"
#include "logging.h"


State::State(unsigned n_) : heads(n_, UINT_MAX), n(n_), nid(n_), beta(0) {
}

bool State::is_terminated() const {
  return (sigma.size() == 1 && beta >= n);
}

void ConstituentSystem::get_valid_actions(const State & state,
                                          std::vector<unsigned>& actions) {
  if (state.beta < state.n) {
    actions.push_back(get_shift_id());
  }
  if (state.sigma.size() > 1) {
    actions.push_back(get_reduce_id());
  }
}

void ConstituentSystem::perform_action(State & state,
                                       const unsigned & action) {
  if (is_shift(action)) {
    shift(state);
  } else {
    reduce(state);
  }
}

bool ConstituentSystem::is_valid(const State & state, const unsigned & action) {
  if (is_shift(action)) {
    return (state.beta < state.n);
  } else {
    return (state.sigma.size() > 1);
  }
  return false;
}

void ConstituentSystem::shift(State & state) {
  state.sigma.push_back(state.beta);
  state.beta++;
}

void ConstituentSystem::reduce(State & state) {
  state.sigma.pop_back();
  state.sigma.pop_back();
  state.sigma.push_back(state.nid);
  state.nid++;
}

void DependencySystem::get_valid_actions(const State & state,
                                         std::vector<unsigned>& actions) {
  if (state.beta < state.n) {
    actions.push_back(get_shift_id());
  }
  if (state.sigma.size() > 1) {
    actions.push_back(get_left_id());
    actions.push_back(get_right_id());
  }
}

void DependencySystem::perform_action(State & state,
                                      const unsigned & action) {
  if (is_shift(action)) {
    shift(state);
  } else if (is_left(action)) {
    left(state);
  } else {
    right(state);
  }
}

bool DependencySystem::is_valid(const State & state,
                                const unsigned & action) {
  if (is_shift(action)) {
    return (state.beta < state.n);
  } else {
    return (state.sigma.size() > 1);
  }
  return false;
}

void DependencySystem::shift(State & state) {
  state.sigma.push_back(state.beta);
  state.beta++;
}

void DependencySystem::left(State & state) {
  unsigned hed = state.sigma.back();
  unsigned mod = state.sigma[state.sigma.size() - 2];
  state.sigma.pop_back();
  state.sigma.pop_back();
  state.sigma.push_back(hed);
  state.heads[mod] = hed;
}

void DependencySystem::right(State & state) {
  unsigned mod = state.sigma.back();
  unsigned hed = state.sigma[state.sigma.size() - 2];
  state.sigma.pop_back();
  state.heads[mod] = hed;
}

TransitionSystem * get_system(const std::string & name) {
  TransitionSystem * system = nullptr;
  if (name == "constituent" || name == "cons") {
    system = new ConstituentSystem();
  } else if (name == "dependency" || name == "dep") {
    system = new DependencySystem();
  } else {
    _ERROR << "Unknown state builder name: " << name;
  }
  return system;
}
