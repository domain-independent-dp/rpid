use crate::solvers::search_algorithms::{BestFirstSearch, CostNode, SearchNode};
use crate::solvers::{Search, SearchParameters};
use crate::{Dominance, DpMut};
use num_traits::Signed;
use std::fmt::Display;
use std::hash::Hash;

/// Creates Dijkstra solver.
///
/// Search nodes are ordered by the cost.
///
/// The DP model must implement the `Dominance` trait.
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
///     type Label = usize;
///
///     fn get_target(&self) -> Self::State {
///         let mut unvisited = FixedBitSet::with_capacity(self.c.len());
///         unvisited.insert_range(1..);
///
///         TspState {
///             unvisited,
///             current: 0,
///        }
///     }
///
///     fn get_successors(
///         &self,
///         state: &Self::State,
///     ) -> impl IntoIterator<Item = (Self::State, Self::CostType, Self::Label)> {
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
///     fn get_base_cost(&self, state: &Self::State) -> Option<Self::CostType> {
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
///     fn get_key(&self, state: &Self::State) -> Self::Key {
///         (state.unvisited.clone(), state.current)
///     }
/// }
///
/// let tsp = Tsp { c: vec![vec![0, 1, 2], vec![1, 0, 3], vec![2, 3, 0]] };
/// let parameters = SearchParameters {
///     quiet: true,
///     ..Default::default()
/// };
/// let mut solver = solvers::create_dijkstra(tsp, parameters);
/// let solution = solver.search();
/// assert_eq!(solution.cost, Some(6));
/// assert_eq!(solution.transitions, vec![1, 2]);
/// assert!(solution.is_optimal);
/// assert!(!solution.is_infeasible);
/// assert_eq!(solution.best_bound, Some(6));
/// ```
pub fn create_dijkstra<'a, D, S, C, L, K>(
    dp: D,
    parameters: SearchParameters<C>,
) -> Box<dyn Search<CostType = C, Label = L> + 'a>
where
    D: DpMut<State = S, CostType = C, Label = L> + Dominance<State = S, Key = K> + 'a,
    S: 'a,
    C: Ord + Copy + Signed + Display + 'a,
    L: Default + Copy + 'a,
    K: Hash + Eq + 'a,
{
    let root_node_constructor = |dp: &mut D, _| {
        Some(CostNode::create_root(
            dp,
            dp.get_target(),
            dp.get_identity_weight(),
        ))
    };
    let node_constructor =
        |dp: &mut _, state, cost, transition, parent: &CostNode<_, _, _, _>, _, _: Option<&_>| {
            Some(parent.create_child(dp, state, cost, transition))
        };
    let solution_checker = |dp: &mut _, node: &CostNode<_, _, _, _>| node.check_solution(dp);

    Box::new(BestFirstSearch::new(
        dp,
        root_node_constructor,
        node_constructor,
        solution_checker,
        parameters,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Solution;
    use crate::dp::Dp;

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

    #[test]
    fn test_dijkstra() {
        let dp = MockDp(2);
        let parameters = SearchParameters {
            quiet: true,
            ..Default::default()
        };
        let mut search = create_dijkstra(dp, parameters);

        let solution: Solution<_, usize> = search.search();
        assert_eq!(solution.cost, Some(2));
        assert_eq!(solution.transitions, vec![1, 1]);
        assert_eq!(solution.best_bound, Some(2));
        assert!(solution.is_optimal);
        assert!(!solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);
    }

    #[test]
    fn test_dijkstra_infeasible() {
        let dp = MockDp(2);
        let parameters = SearchParameters {
            primal_bound: Some(2),
            quiet: true,
            ..Default::default()
        };
        let mut search = create_dijkstra(dp, parameters);

        let solution: Solution<_, usize> = search.search();
        assert_eq!(solution.cost, None);
        assert_eq!(solution.transitions, vec![]);
        assert_eq!(solution.best_bound, None);
        assert!(!solution.is_optimal);
        assert!(solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);
    }
}
