use crate::solvers::search_algorithms::{BestFirstSearch, DualBoundNode, SearchNode};
use crate::solvers::{Search, SearchParameters};
use crate::{Bound, Dominance, Dp};
use num_traits::Signed;
use std::fmt::Display;
use std::hash::Hash;

/// Creates cost-algebraic A* solver.
///
/// Search nodes are ordered by the f-value, which is the combination of the cost and the dual bound.
///
/// The DP model must implement the `Dominance` and `DualBound` traits.
///
/// # Examples
///
/// ```
/// use rpid::prelude::*;
/// use rpid::solvers;
/// use fixedbitset::FixedBitSet;
///
/// struct Tsp {
///     c: Vec<Vec<i32>>,
/// }
///
/// struct TspState {
///     unvisited: FixedBitSet,
///     current: usize,
/// }
///
/// impl Dp for Tsp {
///     type State = TspState;
///     type CostType = i32;
///
///     fn get_target(&self) -> TspState {
///         let mut unvisited = FixedBitSet::with_capacity(self.c.len());
///         unvisited.insert_range(1..);
///
///         TspState {
///             unvisited,
///             current: 0,
///        }
///     }
///
///     fn get_successors(&self, state: &TspState) -> impl IntoIterator<Item = (TspState, i32, usize)> {
///         state.unvisited.ones().map(|next| {
///             let mut unvisited = state.unvisited.clone();
///             unvisited.remove(next);
///
///             let successor = TspState {
///                 unvisited,
///                 current: next,
///             };
///             let weight = self.c[state.current][next];
///             
///             (successor, weight, next)
///         })
///     }
///
///     fn get_base_cost(&self, state: &TspState) -> Option<i32> {
///         if state.unvisited.is_clear() {
///             Some(self.c[state.current][0])
///         } else {
///             None
///         }
///     }
/// }
///
/// impl Dominance for Tsp {
///     type State = TspState;
///     type Key = (FixedBitSet, usize);
///
///     fn get_key(&self, state: &TspState) -> Self::Key {
///         (state.unvisited.clone(), state.current)
///     }
/// }
///
/// impl Bound for Tsp {
///     type State = TspState;
///     type CostType = i32;
///
///     fn get_dual_bound(&self, state: &TspState) -> Option<i32> {
///         Some(0)
///     }
/// }
///
/// let tsp = Tsp { c: vec![vec![0, 1, 2], vec![1, 0, 3], vec![2, 3, 0]] };
/// let parameters = SearchParameters {
///     quiet: true,
///     ..Default::default()
/// };
/// let mut solver = solvers::create_astar(tsp, parameters);
/// let solution = solver.search();
/// assert_eq!(solution.cost, Some(6));
/// assert_eq!(solution.transitions, vec![1, 2]);
/// assert!(solution.is_optimal);
/// assert!(!solution.is_infeasible);
/// assert_eq!(solution.best_bound, Some(6));
/// ```
pub fn create_astar<D, S, C, K>(
    dp: D,
    mut parameters: SearchParameters<C>,
) -> impl Search<CostType = C>
where
    D: Dp<State = S, CostType = C> + Dominance<State = S, Key = K> + Bound<State = S, CostType = C>,
    C: Ord + Copy + Signed + Display,
    K: Hash + Eq,
{
    let root_node_constructor = |dp: &D, bound| {
        DualBoundNode::create_root(dp, dp.get_target(), dp.get_identity_weight(), bound)
    };
    let node_constructor = |dp: &_,
                            state,
                            cost,
                            transition,
                            parent: &DualBoundNode<_, _, _>,
                            primal_bound,
                            other: Option<&_>| {
        parent.create_child(dp, state, cost, transition, primal_bound, other)
    };
    let solution_checker = |dp: &_, node: &DualBoundNode<_, _, _>| node.check_solution(dp);
    parameters.update_bounds(&dp);

    BestFirstSearch::new(
        dp,
        root_node_constructor,
        node_constructor,
        solution_checker,
        parameters,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::cmp::Ordering;

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

    impl Bound for MockDp {
        type State = i32;
        type CostType = i32;

        fn get_dual_bound(&self, _: &Self::State) -> Option<Self::CostType> {
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

        fn get_bound(&self, _: &Self::DpData) -> Option<Self::CostType> {
            None
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

    #[test]
    fn test_astar() {
        let dp = MockDp(2);
        let parameters = SearchParameters {
            quiet: true,
            ..Default::default()
        };
        let mut search = create_astar(dp, parameters);

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
    fn test_astar_infeasible() {
        let dp = MockDp(2);
        let parameters = SearchParameters {
            primal_bound: Some(2),
            quiet: true,
            ..Default::default()
        };
        let mut search = create_astar(dp, parameters);

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
