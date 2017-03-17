#include "system.h"
#include "logging.h"
#include <boost/lexical_cast.hpp>


State::State(unsigned n_) : 
  heads(n_, UINT_MAX),
  pst(n_ * 2, std::pair<unsigned, unsigned>(UINT_MAX, UINT_MAX)),
  n(n_), 
  nid(n_), 
  beta(0) {
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

void ConstituentSystem::print_tree(const State & state, std::ostream & os) {
  BOOST_ASSERT_MSG(state.nid + 1 == state.n * 2, "Wouldn't print tree before end of parsing.");
  unsigned root = state.nid;
  _print_tree(state.pst, root, os);
  os << std::endl;
}

void ConstituentSystem::_print_tree(const std::vector<std::pair<unsigned, unsigned>> & tree,
                                    unsigned now,
                                    std::ostream & os) {
  if (tree[now].first == UINT_MAX && tree[now].second == UINT_MAX) {
    os << now;
    return;
  } else {
    os << "(";
    _print_tree(tree, tree[now].first, os);
    os << " ";
    _print_tree(tree, tree[now].second, os);
    os << ")";
  }
}

void ConstituentSystem::shift(State & state) {
  state.sigma.push_back(state.beta);
  state.beta++;
}

void ConstituentSystem::reduce(State & state) {
  unsigned right = state.sigma.back();
  unsigned left = state.sigma[state.sigma.size() - 2];
  state.sigma.pop_back();
  state.sigma.pop_back();
  state.sigma.push_back(state.nid);
  state.pst[state.nid].first = left;
  state.pst[state.nid].second = right;
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

void DependencySystem::print_tree(const State & state, std::ostream & os) {
  std::vector<std::vector<unsigned>> tree(state.n);
  unsigned root = UINT_MAX;
  for (unsigned i = 0; i < state.n; ++i) {
    unsigned hed = state.heads[i];
    if (hed == UINT_MAX) {
      root = i;
    } else {
      tree[hed].push_back(i);
    }
  }
  unsigned depth = _print_tree_get_depth(tree, root);
  std::vector<std::string> canvas(state.n, std::string(depth * 4, ' '));
  _print_tree(tree, root, 0, canvas);
  for (auto & line : canvas) { os << line << std::endl; }
}

unsigned DependencySystem::_print_tree_get_depth(const std::vector<std::vector<unsigned>>& tree,
                                                 unsigned now) {
  const std::vector<unsigned> & node = tree[now];
  unsigned ret = 0;
  for (auto & c : node) {
    unsigned depth = _print_tree_get_depth(tree, c);
    if (depth > ret) { ret = depth; }
  }
  return ret + 1;
}

void DependencySystem::_print_tree(const std::vector<std::vector<unsigned>>& tree,
                                   unsigned now,
                                   unsigned offset,
                                   std::vector<std::string> & canvas) {
  const std::vector<unsigned> & node = tree[now];
  if (node.size() > 0) {
    unsigned start = (node[0] < now ? node[0] : now);
    unsigned end = (node.back() > now ? node.back() : now);
    canvas[start][offset] = '.';
    canvas[end][offset] = '`';
    for (unsigned k = start + 1; k < end; ++k) { canvas[k][offset] = '|'; }
    for (auto & c : node) {
      canvas[c][offset + 1] = '-'; canvas[c][offset + 2] = '-'; canvas[c][offset + 3] = '-';
      _print_tree(tree, c, offset + 4, canvas);
    }
  }
  std::string name = boost::lexical_cast<std::string>(now);
  for (unsigned i = 0; i < name.size(); ++i)
    canvas[now][offset + i] = name[i];
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
