#ifndef SYSTEM_H
#define SYSTEM_H

#include <iostream>
#include <vector>

struct State {
  State(unsigned n_);

  std::vector<unsigned> sigma;
  std::vector<unsigned> heads;
  std::vector<std::pair<unsigned, unsigned>> pst;
  unsigned n;
  unsigned nid;
  unsigned beta;

  bool is_terminated() const;
};

struct TransitionSystem {
  virtual unsigned num_actions() const = 0;
  virtual void get_oracle_actions(const std::vector<unsigned> & parents,
                                  std::vector<unsigned> & actions) = 0;
  virtual bool is_valid(const State & state, const unsigned & action) = 0;
  virtual void get_valid_actions(const State & state, std::vector<unsigned> & actions) = 0;
  virtual void perform_action(State & state, const unsigned & action) = 0;
  virtual unsigned get_shift() = 0;
  virtual unsigned get_reduce() = 0;
  virtual void print_tree(const State & stat, std::ostream & os) = 0;
};

struct ConstituentSystem : public TransitionSystem {
  unsigned num_actions() const { return 2; }
  void get_oracle_actions(const std::vector<unsigned> & parents,
                          std::vector<unsigned> & actions);
  void get_oracle_actions_travel(const std::vector<std::pair<unsigned, unsigned>> & tree,
                                 unsigned now,
                                 std::vector<unsigned> & actions);                                                  
  void get_valid_actions(const State & state, std::vector<unsigned> & actions);
  void perform_action(State & state, const unsigned & action) override;
  bool is_valid(const State & state, const unsigned & action) override;
  unsigned get_shift() override { return 0; }
  unsigned get_reduce() override { return 1; }
  void print_tree(const State & stat, std::ostream & os) override;
  static void _print_tree(const std::vector<std::pair<unsigned, unsigned>> & pst, unsigned now, std::ostream & os);

  static bool is_shift(const unsigned& action) { return action == 0; }
  static bool is_reduce(const unsigned& action) { return action == 1; }

  static unsigned get_shift_id() { return 0; }
  static unsigned get_reduce_id() { return 1; }

  static void shift(State & state);
  static void reduce(State & state);
};

struct DependencySystem : public TransitionSystem {
  unsigned num_actions() const { return 3; }
  void get_oracle_actions(const std::vector<unsigned> & parents,
                          std::vector<unsigned> & actions);
  void get_oracle_actions_travel(const std::vector<std::vector<unsigned>> & tree,
                                 unsigned now,
                                 std::vector<unsigned> & actions);
  void get_valid_actions(const State & state, std::vector<unsigned> & actions);
  void perform_action(State & state, const unsigned & action) override;
  bool is_valid(const State & state, const unsigned & action) override;
  unsigned get_shift() override { return 0; }
  unsigned get_reduce() override { return 1; }
  void print_tree(const State & state, std::ostream & os) override;
  static unsigned _print_tree_get_depth(const std::vector<std::vector<unsigned>> & tree, unsigned now);
  static void _print_tree(const std::vector<std::vector<unsigned>> & tree,
                          unsigned now,
                          unsigned horizon_offset,
                          std::vector<std::string> & canvas);

  static bool is_shift(const unsigned & action) { return action == 0; }
  static bool is_left(const unsigned & action) { return action == 1; }
  static bool is_right(const unsigned & action) { return action == 2; }

  static unsigned get_shift_id() { return 0; }
  static unsigned get_left_id() { return 1; }
  static unsigned get_right_id() { return 2; }

  static void shift(State & state);
  static void left(State & state);
  static void right(State & state);
};

TransitionSystem * get_system(const std::string & name);

#endif  //  end for SYSTEM