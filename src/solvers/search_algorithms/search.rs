use super::search_nodes::{SearchNode, StateRegistry};
use crate::Bound;
use crate::dp::{Dominance, Dp};
use crate::timer::Timer;
use std::fmt::Display;
use std::hash::Hash;
use std::rc::Rc;

/// Search parameters.
#[derive(Default)]
pub struct SearchParameters<C> {
    /// Primal bound, upper/lower bound on the cost for minimization/maximization.
    pub primal_bound: Option<C>,
    /// Dual bound, lower/upper bound on the cost for minimization/maximization.
    pub dual_bound: Option<C>,
    /// Whether to get all solutions found.
    pub get_all_solutions: bool,
    /// Whether to suppress output.
    pub quiet: bool,
    /// Time limit in seconds.
    pub time_limit: Option<f64>,
    /// Maximum number of nodes to expand.
    pub expansion_limit: Option<usize>,
    /// Initial capacity of the state registry.
    pub initial_registry_capacity: Option<usize>,
}

impl<C> SearchParameters<C> {
    /// Updates the primal and dual bounds with the given DP model.
    pub fn update_bounds<D, S>(&mut self, dp: &D)
    where
        C: Copy,
        D: Dp<State = S, CostType = C> + Bound<State = S, CostType = C>,
    {
        match (self.primal_bound, dp.get_global_primal_bound()) {
            (Some(primal_bound), Some(model_bound))
                if dp.is_better_cost(model_bound, primal_bound) =>
            {
                self.primal_bound = Some(model_bound)
            }
            (None, Some(model_bound)) => self.primal_bound = Some(model_bound),
            _ => {}
        }

        match (self.dual_bound, dp.get_global_dual_bound()) {
            (Some(dual_bound), Some(model_bound)) if dp.is_better_cost(dual_bound, model_bound) => {
                self.dual_bound = Some(model_bound)
            }
            (None, Some(model_bound)) => self.dual_bound = Some(model_bound),
            _ => {}
        }
    }
}

/// Solution information.
#[derive(Clone, PartialEq, Debug)]
pub struct Solution<C> {
    /// Cost of the solution.
    pub cost: Option<C>,
    /// Best dual bound found (lower/upper bound for minimization/maximization).
    pub best_bound: Option<C>,
    /// Whether the solution is optimal.
    pub is_optimal: bool,
    /// Whether the model is infeasible.
    pub is_infeasible: bool,
    /// Transitions of the solution.
    pub transitions: Vec<usize>,
    /// Number of nodes expanded.
    pub expanded: usize,
    /// Number of nodes generated.
    pub generated: usize,
    /// Elapsed time in seconds.
    pub time: f64,
    /// Whether the time limit is reached.
    pub is_time_limit_reached: bool,
    /// Whether the expansion limit is reached.
    pub is_expansion_limit_reached: bool,
}

impl<C> Default for Solution<C> {
    fn default() -> Self {
        Self {
            cost: None,
            best_bound: None,
            is_optimal: false,
            is_infeasible: false,
            transitions: Vec::new(),
            expanded: 0,
            generated: 0,
            time: 0.0,
            is_time_limit_reached: false,
            is_expansion_limit_reached: false,
        }
    }
}

/// Search trait.
pub trait Search {
    type CostType;

    /// Searches for the next solution.
    ///
    /// The second return value indicates whether the search is terminated.
    fn search_next(&mut self) -> (Solution<Self::CostType>, bool);

    /// Performs search until termination.
    fn search(&mut self) -> Solution<Self::CostType> {
        loop {
            let (solution, terminated) = self.search_next();

            if terminated {
                return solution;
            }
        }
    }
}

/// Base search structure.
///
/// - `D` is the DP model.
/// - `C` is the cost type associated with the DP model.
/// - `K` is the key type associated with dominance detection.
/// - `N` is the search node type.
/// - `F` is the node constructor, which generates a new node.
/// - `G` is the solution checker, which checks whether a node is a solution.
pub struct SearchBase<D, C, K, N, F, G> {
    dp: D,
    node_constructor: F,
    solution_checker: G,
    parameters: SearchParameters<C>,
    registry: StateRegistry<K, N>,
    primal_bound: Option<C>,
    solution: Solution<C>,
}

/// Expansion result.
#[derive(Clone, PartialEq, Debug)]
pub enum ExpansionResult<C> {
    /// Node is closed.
    Closed,
    /// Node is pruned by bound.
    PrunedByBound,
    /// Solution found.
    Solution(C, Vec<usize>),
    /// Solution found but pruned since it is not better.
    SolutionPruned,
    /// Node is expanded.
    Expanded,
}

impl<D, C, K, N, F, G, S> SearchBase<D, C, K, N, F, G>
where
    D: Dp<State = S, CostType = C> + Dominance<State = S, Key = K>,
    C: Ord + Copy + Display,
    K: Hash + Eq,
    N: Ord + SearchNode<DpData = D, State = S, CostType = C>,
    F: FnMut(&D, S, C, usize, &N, Option<C>, Option<&N>) -> Option<N>,
    G: FnMut(&D, &N) -> Option<(C, Vec<usize>)>,
{
    /// Creates a new search base.
    ///
    /// `root_node_constructor` is a function that constructs a root search node from the given state and primal bound.
    ///
    /// `node_constructor` is a function that constructs a new search node from the given state,
    /// cost, transition, parent node, and primal bound.
    ///
    /// `solution_checker` is a function that checks whether the given node is a solution
    /// and returns the cost and transitions if it is.
    ///
    /// `callback` is called with the root node is successfully generated by `root_node_constructor`.
    pub fn new(
        dp: D,
        root_node_constructor: impl FnOnce(&D, Option<C>) -> Option<N>,
        node_constructor: F,
        solution_checker: G,
        callback: impl FnOnce(Rc<N>),
        parameters: SearchParameters<C>,
    ) -> Self {
        let mut registry = parameters
            .initial_registry_capacity
            .map(StateRegistry::with_capacity)
            .unwrap_or_default();
        let primal_bound = parameters.primal_bound;
        let mut solution = Solution {
            best_bound: parameters.dual_bound,
            ..Solution::default()
        };

        if let Some(result) = root_node_constructor(&dp, primal_bound).map(|node| {
            registry
                .insert_if_not_dominated(&dp, node)
                .inserted
                .unwrap()
        }) {
            callback(result);
            solution.generated += 1;
        };

        Self {
            dp,
            parameters,
            registry,
            node_constructor,
            solution_checker,
            primal_bound,
            solution,
        }
    }

    /// Expands the node.
    ///
    /// `callback` is called with the generated node.
    pub fn expand(
        &mut self,
        node: &N,
        callback: &mut impl FnMut(Rc<N>),
        timer: &Timer,
    ) -> ExpansionResult<C> {
        if node.is_closed() {
            return ExpansionResult::Closed;
        }

        node.close();

        if let (Some(dual_bound), Some(primal_bound)) =
            (node.get_bound(&self.dp), self.primal_bound)
        {
            if !self.dp.is_better_cost(dual_bound, primal_bound) {
                return ExpansionResult::PrunedByBound;
            }
        }

        if let Some((solution_cost, transitions)) = (self.solution_checker)(&self.dp, node) {
            if self
                .primal_bound
                .is_none_or(|bound| self.dp.is_better_cost(solution_cost, bound))
            {
                self.primal_bound = Some(solution_cost);
                self.solution.cost = Some(solution_cost);
                self.solution.transitions = transitions.clone();

                if Some(solution_cost) == self.solution.best_bound {
                    self.solution.is_optimal = true;
                }

                if !self.parameters.quiet {
                    println!(
                        "New primal bound: {solution_cost}, expanded: {expanded}, generated: {generated}, elapsed time: {time}s.",
                        expanded = self.solution.expanded,
                        generated = self.solution.generated,
                        time = timer.get_elapsed_time()
                    );
                }

                self.solution.time = timer.get_elapsed_time();

                return ExpansionResult::Solution(solution_cost, transitions);
            } else if self.parameters.get_all_solutions {
                return ExpansionResult::Solution(solution_cost, transitions);
            } else {
                return ExpansionResult::SolutionPruned;
            }
        }

        let state = node.get_state(&self.dp);
        let cost = node.get_cost(&self.dp);

        for (successor_state, weight, transition) in self.dp.get_successors(state) {
            let successor_cost = self.dp.combine_cost_weights(cost, weight);
            let constructor = |state, cost, other: Option<&_>| {
                (self.node_constructor)(
                    &self.dp,
                    state,
                    cost,
                    transition,
                    node,
                    self.primal_bound,
                    other,
                )
            };
            let result = self.registry.insert_with_if_not_dominated(
                &self.dp,
                successor_state,
                successor_cost,
                constructor,
            );

            for d in result.dominated.iter() {
                if !d.is_closed() {
                    d.close();
                }
            }

            if let Some(inserted) = result.inserted {
                if result.dominated.is_empty() {
                    self.solution.generated += 1;
                }

                callback(inserted);
            }
        }

        self.solution.expanded += 1;

        if self
            .parameters
            .expansion_limit
            .is_some_and(|limit| self.solution.expanded >= limit)
        {
            if !self.parameters.quiet {
                println!("Expansion limit reached.");
            }

            self.solution.is_expansion_limit_reached = true;
            self.solution.time = timer.get_elapsed_time();
        }

        ExpansionResult::Expanded
    }

    /// Updates the dual bound if it is better.
    pub fn update_dual_bound_if_better(&mut self, dual_bound: C, timer: &Timer) -> bool {
        let dual_bound = if let Some(primal_bound) = self.primal_bound {
            if self.dp.is_better_cost(primal_bound, dual_bound) {
                primal_bound
            } else {
                dual_bound
            }
        } else {
            dual_bound
        };

        if self.solution.best_bound.is_none()
            || self
                .dp
                .is_better_cost(self.solution.best_bound.unwrap(), dual_bound)
        {
            if !self.parameters.quiet {
                println!(
                    "New dual bound: {dual_bound}, expanded: {expanded}, generated: {generated}, elapsed time: {time}s.",
                    expanded = self.solution.expanded,
                    generated = self.solution.generated,
                    time = timer.get_elapsed_time()
                );
            }

            self.solution.best_bound = Some(dual_bound);

            if self.solution.best_bound == self.primal_bound {
                self.solution.is_optimal = self.solution.cost.is_some();
                self.solution.is_infeasible = self.solution.cost.is_none();
                self.solution.time = timer.get_elapsed_time();
            }

            true
        } else {
            false
        }
    }

    /// Notifies that the time limit is reached.
    pub fn notify_time_limit_reached(&mut self, timer: &Timer) {
        self.solution.is_time_limit_reached = true;

        if !self.parameters.quiet {
            println!("Time limit reached.");
        }

        self.solution.time = timer.get_elapsed_time();
    }

    /// Notifies that the search is finished.
    pub fn notify_finished(&mut self, timer: &Timer) {
        self.solution.is_optimal = self.solution.cost.is_some();
        self.solution.best_bound = self.solution.cost;
        self.solution.is_infeasible = self.solution.cost.is_none();

        if !self.parameters.quiet {
            if self.solution.is_optimal {
                println!("Optimal solution found.");
            } else if self.solution.is_infeasible {
                println!("Proved infeasible.");
            }
        }

        self.solution.time = timer.get_elapsed_time();
    }

    /// Returns whether the search is terminated.
    pub fn is_terminated(&self) -> bool {
        self.solution.is_optimal
            || self.solution.is_infeasible
            || self.solution.is_time_limit_reached
            || self.solution.is_expansion_limit_reached
    }

    /// Returns the DP model.
    pub fn get_dp(&self) -> &D {
        &self.dp
    }

    /// Returns the solution.
    pub fn get_solution(&self) -> &Solution<C> {
        &self.solution
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;
    use std::{cell::Cell, collections::VecDeque};

    #[derive(PartialEq, Eq)]
    struct MockDp(i32);

    impl Dp for MockDp {
        type State = i32;
        type CostType = i32;

        fn get_target(&self) -> Self::State {
            self.0
        }

        fn get_successors(
            &self,
            state: &Self::State,
        ) -> impl IntoIterator<Item = (Self::State, Self::CostType, usize)> {
            vec![(*state + 1, 1, 0), (*state - 1, 1, 1)]
        }

        fn get_base_cost(&self, state: &Self::State) -> Option<Self::CostType> {
            if *state <= 0 { Some(0) } else { None }
        }
    }

    impl Dominance for MockDp {
        type State = i32;
        type Key = i32;

        fn get_key(&self, state: &Self::State) -> Self::Key {
            *state
        }
    }

    impl Bound for MockDp {
        type State = i32;
        type CostType = i32;

        fn get_dual_bound(&self, state: &Self::State) -> Option<Self::CostType> {
            Some(*state)
        }

        fn get_global_primal_bound(&self) -> Option<Self::CostType> {
            Some(4)
        }

        fn get_global_dual_bound(&self) -> Option<Self::CostType> {
            Some(0)
        }
    }

    struct MockNode(i32, i32, Cell<bool>, Vec<usize>);

    impl SearchNode for MockNode {
        type DpData = MockDp;
        type State = i32;
        type CostType = i32;

        fn get_state(&self, _: &Self::DpData) -> &Self::State {
            &self.0
        }

        fn get_state_mut(&mut self, _: &Self::DpData) -> &mut Self::State {
            &mut self.0
        }

        fn get_cost(&self, _: &Self::DpData) -> Self::CostType {
            self.1
        }

        fn get_bound(&self, dp: &Self::DpData) -> Option<Self::CostType> {
            dp.get_dual_bound(&self.0)
        }

        fn close(&self) {
            self.2.set(true)
        }

        fn is_closed(&self) -> bool {
            self.2.get()
        }

        fn get_transitions(&self, _: &Self::DpData) -> Vec<usize> {
            self.3.clone()
        }
    }

    impl PartialEq for MockNode {
        fn eq(&self, other: &Self) -> bool {
            self.1 == other.1
        }
    }

    impl Eq for MockNode {}

    impl Ord for MockNode {
        fn cmp(&self, other: &Self) -> Ordering {
            other.1.cmp(&self.1)
        }
    }

    impl PartialOrd for MockNode {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    struct MockSearch;

    impl Search for MockSearch {
        type CostType = i32;

        fn search_next(&mut self) -> (Solution<Self::CostType>, bool) {
            (
                Solution {
                    cost: Some(42),
                    best_bound: Some(42),
                    is_optimal: true,
                    ..Default::default()
                },
                true,
            )
        }
    }

    #[test]
    fn test_search_parameters_update_bounds() {
        let mut parameters = SearchParameters::default();
        let dp = MockDp(2);

        parameters.update_bounds(&dp);

        assert_eq!(parameters.primal_bound, Some(4));
        assert_eq!(parameters.dual_bound, Some(0));
    }

    #[test]
    fn test_search_parameters_update_bounds_not_updated() {
        let mut parameters = SearchParameters {
            primal_bound: Some(3),
            dual_bound: Some(1),
            ..SearchParameters::default()
        };
        let dp = MockDp(2);

        parameters.update_bounds(&dp);

        assert_eq!(parameters.primal_bound, Some(3));
        assert_eq!(parameters.dual_bound, Some(1));
    }

    #[test]
    fn test_search() {
        let mut search = MockSearch;

        let solution = search.search();

        assert_eq!(
            solution,
            Solution {
                cost: Some(42),
                best_bound: Some(42),
                is_optimal: true,
                ..Default::default()
            }
        );
    }

    #[test]
    fn test_search_base() {
        let timer = Timer::default();
        let dp = MockDp(2);
        let root_node_constructor = |dp: &MockDp, _| {
            Some(MockNode(
                dp.get_target(),
                dp.get_identity_weight(),
                Cell::new(false),
                Vec::new(),
            ))
        };
        let node_constructor =
            |_: &_, state, cost, transition, parent: &MockNode, _, _: Option<&_>| {
                let mut transitions = parent.3.clone();
                transitions.push(transition);
                Some(MockNode(state, cost, Cell::new(false), transitions))
            };
        let solution_checker = |dp: &MockDp, node: &MockNode| {
            dp.get_base_cost(node.get_state(dp)).map(|cost| {
                (
                    dp.combine_cost_weights(node.get_cost(dp), cost),
                    node.3.clone(),
                )
            })
        };
        let mut open = VecDeque::new();
        let mut callback = |node| open.push_back(node);
        let parameters = SearchParameters {
            quiet: true,
            ..Default::default()
        };

        let mut search = SearchBase::new(
            dp,
            root_node_constructor,
            node_constructor,
            solution_checker,
            &mut callback,
            parameters,
        );

        assert!(!search.is_terminated());
        assert_eq!(search.get_dp().0, 2);
        let solution = Solution {
            generated: 1,
            ..Default::default()
        };
        assert_eq!(search.get_solution(), &solution);

        assert_eq!(open.len(), 1);
        let node = open.pop_front().unwrap();
        assert_eq!(node.0, 2);
        assert_eq!(node.1, 0);
        assert!(!node.2.get());
        assert!(node.3.is_empty());

        let mut callback = |node| open.push_back(node);
        let result = search.expand(&node, &mut callback, &timer);
        assert_eq!(result, ExpansionResult::Expanded);
        assert!(node.is_closed());
        assert!(!search.is_terminated());
        let solution = search.get_solution();
        assert_eq!(solution.expanded, 1);
        assert_eq!(solution.generated, 3);
        assert_eq!(solution.cost, None);
        assert_eq!(solution.transitions, vec![]);
        assert_eq!(solution.best_bound, None);
        assert!(!solution.is_optimal);
        assert!(!solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);

        assert_eq!(open.len(), 2);
        let node = open.pop_front().unwrap();
        assert_eq!(node.0, 3);
        assert_eq!(node.1, 1);
        assert!(!node.2.get());
        assert_eq!(node.3, vec![0]);

        let mut callback = |node| open.push_back(node);
        let result = search.expand(&node, &mut callback, &timer);
        assert_eq!(result, ExpansionResult::Expanded);
        assert!(node.is_closed());
        assert!(!search.is_terminated());
        let solution = search.get_solution();
        assert_eq!(solution.expanded, 2);
        assert_eq!(solution.generated, 4);
        assert_eq!(solution.cost, None);
        assert_eq!(solution.transitions, vec![]);
        assert_eq!(solution.best_bound, None);
        assert!(!solution.is_optimal);
        assert!(!solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);

        assert_eq!(open.len(), 2);
        let node = open.pop_front().unwrap();
        assert_eq!(node.0, 1);
        assert_eq!(node.1, 1);
        assert!(!node.2.get());
        assert_eq!(node.3, vec![1]);

        let mut callback = |node| open.push_back(node);
        let result = search.expand(&node, &mut callback, &timer);
        assert_eq!(result, ExpansionResult::Expanded);
        assert!(node.is_closed());
        assert!(!search.is_terminated());
        let solution = search.get_solution();
        assert_eq!(solution.expanded, 3);
        assert_eq!(solution.generated, 5);
        assert_eq!(solution.cost, None);
        assert_eq!(solution.transitions, vec![]);
        assert_eq!(solution.best_bound, None);
        assert!(!solution.is_optimal);
        assert!(!solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);

        assert_eq!(open.len(), 2);
        let node = open.pop_front().unwrap();
        assert_eq!(node.0, 4);
        assert_eq!(node.1, 2);
        assert!(!node.2.get());
        assert_eq!(node.3, vec![0, 0]);

        let mut callback = |node| open.push_back(node);
        let result = search.expand(&node, &mut callback, &timer);
        assert_eq!(result, ExpansionResult::Expanded);
        assert!(node.is_closed());
        assert!(!search.is_terminated());
        let solution = search.get_solution();
        assert_eq!(solution.expanded, 4);
        assert_eq!(solution.generated, 6);
        assert_eq!(solution.cost, None);
        assert_eq!(solution.transitions, vec![]);
        assert_eq!(solution.best_bound, None);
        assert!(!solution.is_optimal);
        assert!(!solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);

        assert_eq!(open.len(), 2);
        let node = open.pop_front().unwrap();
        assert_eq!(node.0, 0);
        assert_eq!(node.1, 2);
        assert!(!node.2.get());
        assert_eq!(node.3, vec![1, 1]);

        let mut callback = |node| open.push_back(node);
        let result = search.expand(&node, &mut callback, &timer);
        assert_eq!(result, ExpansionResult::Solution(2, vec![1, 1]));
        assert!(node.is_closed());
        assert!(!search.is_terminated());
        let solution = search.get_solution();
        assert_eq!(solution.expanded, 4);
        assert_eq!(solution.generated, 6);
        assert_eq!(solution.cost, Some(2));
        assert_eq!(solution.transitions, vec![1, 1]);
        assert_eq!(solution.best_bound, None);
        assert!(!solution.is_optimal);
        assert!(!solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);

        assert_eq!(open.len(), 1);
        let node = open.pop_front().unwrap();
        assert_eq!(node.0, 5);
        assert_eq!(node.1, 3);
        assert!(!node.2.get());
        assert_eq!(node.3, vec![0, 0, 0]);

        let mut callback = |node| open.push_back(node);
        let result = search.expand(&node, &mut callback, &timer);
        assert_eq!(result, ExpansionResult::PrunedByBound);
        assert!(node.is_closed());
        assert!(!search.is_terminated());
        let solution = search.get_solution();
        assert_eq!(solution.expanded, 4);
        assert_eq!(solution.generated, 6);
        assert_eq!(solution.cost, Some(2));
        assert_eq!(solution.transitions, vec![1, 1]);
        assert_eq!(solution.best_bound, None);
        assert!(!solution.is_optimal);
        assert!(!solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);

        let node = MockNode(-2, 2, Cell::new(false), Vec::new());
        let mut callback = |node| open.push_back(node);
        let result = search.expand(&node, &mut callback, &timer);
        assert_eq!(result, ExpansionResult::SolutionPruned);

        search.notify_finished(&timer);
        assert!(search.is_terminated());
        let solution = search.get_solution();
        assert_eq!(solution.expanded, 4);
        assert_eq!(solution.generated, 6);
        assert_eq!(solution.cost, Some(2));
        assert_eq!(solution.transitions, vec![1, 1]);
        assert_eq!(solution.best_bound, Some(2));
        assert!(solution.is_optimal);
        assert!(!solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);
    }

    #[test]
    fn test_search_base_get_all_solutions() {
        let timer = Timer::default();
        let dp = MockDp(2);
        let root_node_constructor =
            |_: &MockDp, _| Some(MockNode(0, 2, Cell::new(false), Vec::new()));
        let node_constructor =
            |_: &_, state, cost, transition, parent: &MockNode, _, _: Option<&_>| {
                let mut transitions = parent.3.clone();
                transitions.push(transition);
                Some(MockNode(state, cost, Cell::new(false), transitions))
            };
        let solution_checker = |dp: &MockDp, node: &MockNode| {
            dp.get_base_cost(node.get_state(dp)).map(|cost| {
                (
                    dp.combine_cost_weights(node.get_cost(dp), cost),
                    node.3.clone(),
                )
            })
        };
        let mut open = VecDeque::new();
        let mut callback = |node| open.push_back(node);
        let parameters = SearchParameters {
            quiet: true,
            get_all_solutions: true,
            ..Default::default()
        };

        let mut search = SearchBase::new(
            dp,
            root_node_constructor,
            node_constructor,
            solution_checker,
            &mut callback,
            parameters,
        );

        assert_eq!(open.len(), 1);
        let node = open.pop_back().unwrap();
        let mut callback = |node| open.push_back(node);
        search.expand(&node, &mut callback, &timer);
        assert!(!search.is_terminated());
        let solution = search.get_solution();
        assert_eq!(solution.expanded, 0);
        assert_eq!(solution.generated, 1);
        assert_eq!(solution.cost, Some(2));
        assert_eq!(solution.transitions, vec![]);
        assert_eq!(solution.best_bound, None);
        assert!(!solution.is_optimal);
        assert!(!solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);

        let node = MockNode(-2, 2, Cell::new(false), vec![1, 1, 1, 1]);
        let mut callback = |node| open.push_back(node);
        let result = search.expand(&node, &mut callback, &timer);
        assert_eq!(result, ExpansionResult::Solution(2, vec![1, 1, 1, 1]));
        let solution = search.get_solution();
        assert_eq!(solution.expanded, 0);
        assert_eq!(solution.generated, 1);
        assert_eq!(solution.cost, Some(2));
        assert_eq!(solution.transitions, vec![]);
        assert_eq!(solution.best_bound, None);
        assert!(!solution.is_optimal);
        assert!(!solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);
    }

    #[test]
    fn test_search_base_expansion_limit() {
        let timer = Timer::default();
        let dp = MockDp(2);
        let root_node_constructor = |dp: &MockDp, _| {
            Some(MockNode(
                dp.get_target(),
                dp.get_identity_weight(),
                Cell::new(false),
                Vec::new(),
            ))
        };
        let node_constructor =
            |_: &_, state, cost, transition, parent: &MockNode, _, _: Option<&_>| {
                let mut transitions = parent.3.clone();
                transitions.push(transition);
                Some(MockNode(state, cost, Cell::new(false), transitions))
            };
        let solution_checker = |dp: &MockDp, node: &MockNode| {
            dp.get_base_cost(node.get_state(dp)).map(|cost| {
                (
                    dp.combine_cost_weights(node.get_cost(dp), cost),
                    node.3.clone(),
                )
            })
        };
        let mut open = VecDeque::new();
        let mut callback = |node| open.push_back(node);
        let parameters = SearchParameters {
            quiet: true,
            expansion_limit: Some(1),
            ..Default::default()
        };

        let mut search = SearchBase::new(
            dp,
            root_node_constructor,
            node_constructor,
            solution_checker,
            &mut callback,
            parameters,
        );

        assert!(!search.is_terminated());
        assert_eq!(search.get_dp().0, 2);
        let solution = Solution {
            generated: 1,
            ..Default::default()
        };
        assert_eq!(search.get_solution(), &solution);

        assert_eq!(open.len(), 1);
        let node = open.pop_front().unwrap();
        assert_eq!(node.0, 2);
        assert_eq!(node.1, 0);
        assert!(!node.2.get());
        assert!(node.3.is_empty());

        let mut callback = |node| open.push_back(node);
        let result = search.expand(&node, &mut callback, &timer);
        assert_eq!(result, ExpansionResult::Expanded);
        assert!(node.is_closed());
        assert!(search.is_terminated());
        let solution = search.get_solution();
        assert_eq!(solution.expanded, 1);
        assert_eq!(solution.generated, 3);
        assert_eq!(solution.cost, None);
        assert_eq!(solution.transitions, vec![]);
        assert_eq!(solution.best_bound, None);
        assert!(!solution.is_optimal);
        assert!(!solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(solution.is_expansion_limit_reached);
    }

    #[test]
    fn test_search_base_update_dual_bound_if_better() {
        let timer = Timer::default();
        let dp = MockDp(2);
        let root_node_constructor = |dp: &MockDp, _| {
            Some(MockNode(
                dp.get_target(),
                dp.get_identity_weight(),
                Cell::new(false),
                Vec::new(),
            ))
        };
        let node_constructor =
            |_: &_, state, cost, transition, parent: &MockNode, _, _: Option<&_>| {
                let mut transitions = parent.3.clone();
                transitions.push(transition);
                Some(MockNode(state, cost, Cell::new(false), transitions))
            };
        let solution_checker = |dp: &MockDp, node: &MockNode| {
            dp.get_base_cost(node.get_state(dp)).map(|cost| {
                (
                    dp.combine_cost_weights(node.get_cost(dp), cost),
                    node.3.clone(),
                )
            })
        };
        let mut open = VecDeque::new();
        let mut callback = |node| open.push_back(node);
        let parameters = SearchParameters {
            quiet: true,
            primal_bound: Some(2),
            ..Default::default()
        };

        let mut search = SearchBase::new(
            dp,
            root_node_constructor,
            node_constructor,
            solution_checker,
            &mut callback,
            parameters,
        );

        search.update_dual_bound_if_better(0, &timer);
        assert!(!search.is_terminated());
        let solution = search.get_solution();
        assert_eq!(solution.expanded, 0);
        assert_eq!(solution.generated, 1);
        assert_eq!(solution.cost, None);
        assert_eq!(solution.transitions, vec![]);
        assert_eq!(solution.best_bound, Some(0));
        assert!(!solution.is_optimal);
        assert!(!solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);

        search.update_dual_bound_if_better(4, &timer);
        assert!(search.is_terminated());
        let solution = search.get_solution();
        assert_eq!(solution.expanded, 0);
        assert_eq!(solution.generated, 1);
        assert_eq!(solution.cost, None);
        assert_eq!(solution.transitions, vec![]);
        assert_eq!(solution.best_bound, Some(2));
        assert!(!solution.is_optimal);
        assert!(solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);
    }

    #[test]
    fn test_search_base_update_dual_bound_if_better_with_solution() {
        let timer = Timer::default();
        let dp = MockDp(2);
        let root_node_constructor =
            |_: &MockDp, _| Some(MockNode(0, 2, Cell::new(false), Vec::new()));
        let node_constructor =
            |_: &_, state, cost, transition, parent: &MockNode, _, _: Option<&_>| {
                let mut transitions = parent.3.clone();
                transitions.push(transition);
                Some(MockNode(state, cost, Cell::new(false), transitions))
            };
        let solution_checker = |dp: &MockDp, node: &MockNode| {
            dp.get_base_cost(node.get_state(dp)).map(|cost| {
                (
                    dp.combine_cost_weights(node.get_cost(dp), cost),
                    node.3.clone(),
                )
            })
        };
        let mut open = VecDeque::new();
        let mut callback = |node| open.push_back(node);
        let parameters = SearchParameters {
            quiet: true,
            ..Default::default()
        };

        let mut search = SearchBase::new(
            dp,
            root_node_constructor,
            node_constructor,
            solution_checker,
            &mut callback,
            parameters,
        );

        assert_eq!(open.len(), 1);
        let node = open.pop_back().unwrap();
        let mut callback = |node| open.push_back(node);
        search.expand(&node, &mut callback, &timer);

        search.update_dual_bound_if_better(0, &timer);
        assert!(!search.is_terminated());
        let solution = search.get_solution();
        assert_eq!(solution.expanded, 0);
        assert_eq!(solution.generated, 1);
        assert_eq!(solution.cost, Some(2));
        assert_eq!(solution.transitions, vec![]);
        assert_eq!(solution.best_bound, Some(0));
        assert!(!solution.is_optimal);
        assert!(!solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);

        search.update_dual_bound_if_better(4, &timer);
        assert!(search.is_terminated());
        let solution = search.get_solution();
        assert_eq!(solution.expanded, 0);
        assert_eq!(solution.generated, 1);
        assert_eq!(solution.cost, Some(2));
        assert_eq!(solution.transitions, vec![]);
        assert_eq!(solution.best_bound, Some(2));
        assert!(solution.is_optimal);
        assert!(!solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);
    }

    #[test]
    fn test_notify_time_limit_reached() {
        let timer = Timer::default();
        let dp = MockDp(2);
        let root_node_constructor =
            |_: &MockDp, _| Some(MockNode(0, 2, Cell::new(false), Vec::new()));
        let node_constructor =
            |_: &_, state, cost, transition, parent: &MockNode, _, _: Option<&_>| {
                let mut transitions = parent.3.clone();
                transitions.push(transition);
                Some(MockNode(state, cost, Cell::new(false), transitions))
            };
        let solution_checker = |dp: &MockDp, node: &MockNode| {
            dp.get_base_cost(node.get_state(dp)).map(|cost| {
                (
                    dp.combine_cost_weights(node.get_cost(dp), cost),
                    node.3.clone(),
                )
            })
        };
        let mut open = VecDeque::new();
        let mut callback = |node| open.push_back(node);
        let parameters = SearchParameters {
            quiet: true,
            ..Default::default()
        };

        let mut search = SearchBase::new(
            dp,
            root_node_constructor,
            node_constructor,
            solution_checker,
            &mut callback,
            parameters,
        );

        assert!(!search.is_terminated());
        search.notify_time_limit_reached(&timer);
        assert!(search.is_terminated());
        let solution = search.get_solution();
        assert_eq!(solution.expanded, 0);
        assert_eq!(solution.generated, 1);
        assert_eq!(solution.cost, None);
        assert_eq!(solution.transitions, vec![]);
        assert_eq!(solution.best_bound, None);
        assert!(!solution.is_optimal);
        assert!(!solution.is_infeasible);
        assert!(solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);
    }

    #[test]
    fn test_search_base_notify_finished() {
        let timer = Timer::default();
        let dp = MockDp(2);
        let root_node_constructor = |dp: &MockDp, _| {
            Some(MockNode(
                dp.get_target(),
                dp.get_identity_weight(),
                Cell::new(false),
                Vec::new(),
            ))
        };
        let node_constructor =
            |_: &_, state, cost, transition, parent: &MockNode, _, _: Option<&_>| {
                let mut transitions = parent.3.clone();
                transitions.push(transition);
                Some(MockNode(state, cost, Cell::new(false), transitions))
            };
        let solution_checker = |dp: &MockDp, node: &MockNode| {
            dp.get_base_cost(node.get_state(dp)).map(|cost| {
                (
                    dp.combine_cost_weights(node.get_cost(dp), cost),
                    node.3.clone(),
                )
            })
        };
        let mut open = VecDeque::new();
        let mut callback = |node| open.push_back(node);
        let parameters = SearchParameters {
            quiet: true,
            primal_bound: Some(2),
            ..Default::default()
        };

        let mut search = SearchBase::new(
            dp,
            root_node_constructor,
            node_constructor,
            solution_checker,
            &mut callback,
            parameters,
        );

        search.notify_finished(&timer);
        assert!(search.is_terminated());
        let solution = search.get_solution();
        assert_eq!(solution.expanded, 0);
        assert_eq!(solution.generated, 1);
        assert_eq!(solution.cost, None);
        assert_eq!(solution.transitions, vec![]);
        assert_eq!(solution.best_bound, None);
        assert!(!solution.is_optimal);
        assert!(solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);
    }
}
