use super::search::{ExpansionResult, Search, SearchBase, SearchParameters, Solution};
use super::search_nodes::SearchNode;
use crate::dp::{Dominance, DpMut};
use crate::timer::Timer;
use std::collections::BinaryHeap;
use std::fmt::Display;
use std::hash::Hash;
use std::rc::Rc;

/// Best-first search.
pub struct BestFirstSearch<D, S, C, L, K, N, F, G> {
    base: SearchBase<D, S, C, L, K, N, F, G>,
    open: BinaryHeap<Rc<N>>,
    timer: Timer,
}

impl<D, S, C, L, K, N, F, G> BestFirstSearch<D, S, C, L, K, N, F, G>
where
    D: DpMut<State = S, CostType = C, Label = L> + Dominance<State = S, Key = K>,
    C: Ord + Copy + Display,
    L: Copy,
    K: Hash + Eq,
    N: Ord + SearchNode<DpData = D, State = S, CostType = C, Label = L>,
    F: FnMut(&mut D, S, C, L, &N, Option<C>, Option<&N>) -> Option<N>,
    G: FnMut(&mut D, &N) -> Option<(C, Vec<L>)>,
{
    /// Creates a new instance of the best-first search algorithm.
    ///
    /// `root_node_constructor` is a function that constructs a root search node from the given state and primal bound.
    ///
    /// `node_constructor` is a function that constructs a new search node from the given state,
    /// cost, transition, parent node, and primal bound.
    ///
    /// `solution_checker` is a function that checks whether the given node is a solution
    /// and returns the cost and transitions if it is.
    pub fn new(
        dp: D,
        root_node_constructor: impl FnOnce(&mut D, Option<C>) -> Option<N>,
        node_constructor: F,
        solution_checker: G,
        parameters: SearchParameters<C>,
    ) -> Self {
        let mut timer = parameters
            .time_limit
            .map(Timer::with_time_limit)
            .unwrap_or_default();

        let mut open = BinaryHeap::with_capacity(1);
        let callback = |node| {
            open.push(node);
        };
        let base = SearchBase::new(
            dp,
            root_node_constructor,
            node_constructor,
            solution_checker,
            callback,
            parameters,
        );

        timer.stop();

        Self { base, open, timer }
    }
}

impl<D, S, C, L, K, N, F, G> Search for BestFirstSearch<D, S, C, L, K, N, F, G>
where
    D: DpMut<State = S, CostType = C, Label = L> + Dominance<State = S, Key = K>,
    C: Ord + Copy + Display,
    L: Copy,
    K: Hash + Eq,
    N: Ord + SearchNode<DpData = D, State = S, CostType = C, Label = L>,
    F: FnMut(&mut D, S, C, L, &N, Option<C>, Option<&N>) -> Option<N>,
    G: FnMut(&mut D, &N) -> Option<(C, Vec<L>)>,
{
    type CostType = C;
    type Label = L;

    fn search_next(&mut self) -> (Solution<C, L>, bool) {
        self.timer.start();

        if self.base.is_terminated() {
            self.timer.stop();

            return (self.base.get_solution().clone(), true);
        }

        while let Some(node) = self.open.pop() {
            if self.timer.check_time_limit() {
                self.base.notify_time_limit_reached(&self.timer);
            }

            if N::ordered_by_bound() || self.open.is_empty() {
                if let Some(bound) = node.get_bound(self.base.get_dp()) {
                    self.base.update_dual_bound_if_better(bound, &self.timer);
                }
            }

            let mut callback = |node| {
                self.open.push(node);
            };

            let result = self.base.expand(&node, &mut callback, &self.timer);

            match result {
                ExpansionResult::PrunedByBound if N::ordered_by_bound() => {
                    self.open.clear();
                }
                ExpansionResult::Solution(cost, transitions) => {
                    let mut solution = self.base.get_solution().clone();
                    solution.cost = Some(cost);
                    solution.transitions = transitions;
                    solution.time = self.timer.get_elapsed_time();
                    self.timer.stop();

                    return (solution, self.base.is_terminated());
                }
                _ => {}
            }

            if self.base.is_terminated() {
                self.timer.stop();

                return (self.base.get_solution().clone(), true);
            }
        }

        self.base.notify_finished(&self.timer);
        self.timer.stop();

        (self.base.get_solution().clone(), true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dp::Dp;
    use std::cell::Cell;
    use std::cmp::Ordering;

    #[derive(PartialEq, Eq)]
    struct MockDp(i32);

    impl Dp for MockDp {
        type State = i32;
        type CostType = i32;
        type Label = usize;

        fn get_target(&self) -> Self::State {
            self.0
        }

        fn get_successors(
            &self,
            state: &Self::State,
        ) -> impl IntoIterator<Item = (Self::State, Self::CostType, Self::Label)> {
            vec![(*state - 1, 1, 1)]
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

    struct MockNode(i32, i32, Cell<bool>, Vec<usize>);

    impl SearchNode for MockNode {
        type DpData = MockDp;
        type State = i32;
        type CostType = i32;
        type Label = usize;

        fn get_state(&self, _: &Self::DpData) -> &Self::State {
            &self.0
        }

        fn get_state_mut(&mut self, _: &Self::DpData) -> &mut Self::State {
            &mut self.0
        }

        fn get_cost(&self, _: &Self::DpData) -> Self::CostType {
            self.1
        }

        fn get_bound(&self, _: &Self::DpData) -> Option<Self::CostType> {
            None
        }

        fn close(&self) {
            self.2.set(true)
        }

        fn is_closed(&self) -> bool {
            self.2.get()
        }

        fn get_transitions(&self, _: &Self::DpData) -> Vec<Self::Label> {
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

    #[test]
    fn test_search() {
        let dp = MockDp(2);
        let root_node_constructor = |dp: &mut _, _| {
            Some(MockNode(
                Dp::get_target(dp),
                Dp::get_identity_weight(dp),
                Cell::new(false),
                Vec::new(),
            ))
        };
        let node_constructor =
            |_: &mut _, state, cost, transition, parent: &MockNode, _, _: Option<&_>| {
                let mut transitions = parent.3.clone();
                transitions.push(transition);
                Some(MockNode(state, cost, Cell::new(false), transitions))
            };
        let solution_checker = |dp: &mut MockDp, node: &MockNode| {
            dp.get_base_cost(node.get_state(dp)).map(|cost| {
                (
                    Dp::combine_cost_weights(dp, node.get_cost(dp), cost),
                    node.3.clone(),
                )
            })
        };
        let parameters = SearchParameters {
            quiet: true,
            ..Default::default()
        };

        let mut search = BestFirstSearch::new(
            dp,
            root_node_constructor,
            node_constructor,
            solution_checker,
            parameters,
        );

        let solution = search.search();
        assert_eq!(solution.cost, Some(2));
        assert_eq!(solution.transitions, vec![1, 1]);
        assert_eq!(solution.best_bound, Some(2));
        assert!(solution.is_optimal);
        assert!(!solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);
    }

    #[test]
    fn test_search_infeasible() {
        let dp = MockDp(2);
        let root_node_constructor = |dp: &mut _, _| {
            Some(MockNode(
                Dp::get_target(dp),
                Dp::get_identity_weight(dp),
                Cell::new(false),
                Vec::new(),
            ))
        };
        let node_constructor =
            |_: &mut _, state, cost, transition, parent: &MockNode, _, _: Option<&_>| {
                let mut transitions = parent.3.clone();
                transitions.push(transition);
                Some(MockNode(state, cost, Cell::new(false), transitions))
            };
        let solution_checker = |dp: &mut MockDp, node: &MockNode| {
            dp.get_base_cost(node.get_state(dp)).map(|cost| {
                (
                    Dp::combine_cost_weights(dp, node.get_cost(dp), cost),
                    node.3.clone(),
                )
            })
        };
        let parameters = SearchParameters {
            primal_bound: Some(2),
            quiet: true,
            ..Default::default()
        };

        let mut search = BestFirstSearch::new(
            dp,
            root_node_constructor,
            node_constructor,
            solution_checker,
            parameters,
        );

        let solution = search.search();
        assert_eq!(solution.cost, None);
        assert_eq!(solution.transitions, vec![]);
        assert_eq!(solution.best_bound, None);
        assert!(!solution.is_optimal);
        assert!(solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);
    }
}
