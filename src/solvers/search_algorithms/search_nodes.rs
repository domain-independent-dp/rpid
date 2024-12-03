//! Search nodes for state space search.

mod cost_node;
mod dual_bound_node;
mod id_tree;
mod state_registry;

pub use crate::dp::Dp;
pub use cost_node::CostNode;
pub use dual_bound_node::DualBoundNode;
pub use id_tree::IdTree;
pub use state_registry::{InsertionResult, StateRegistry};

/// Trait for search nodes.
pub trait SearchNode: Sized {
    /// Type of the DP data.
    type DpData: Dp<State = Self::State, CostType = Self::CostType>;
    /// Type of the state.
    type State;
    /// Type of the cost.
    type CostType;

    /// Returns the state of the node.
    fn get_state(&self, dp: &Self::DpData) -> &Self::State;

    /// Returns the state of the node.
    fn get_state_mut(&mut self, dp: &Self::DpData) -> &mut Self::State;

    /// Returns the cost of the node.
    fn get_cost(&self, dp: &Self::DpData) -> Self::CostType;

    /// Returns the dual bound of the path cost extending the node.
    fn get_bound(&self, dp: &Self::DpData) -> Option<Self::CostType>;

    /// Returns whether the node is closed.
    fn is_closed(&self) -> bool;

    /// Closes the node.
    fn close(&self);

    /// Returns the transitions to reach the node.
    fn get_transitions(&self, dp: &Self::DpData) -> Vec<usize>;

    /// Checks if the node is a solution and returns the cost and transitions if it is.
    fn check_solution(&self, dp: &Self::DpData) -> Option<(Self::CostType, Vec<usize>)> {
        let state = self.get_state(dp);
        let cost = self.get_cost(dp);

        if let Some(solution_cost) = dp
            .get_base_cost(state)
            .map(|base_cost| dp.combine_cost_weights(cost, base_cost))
        {
            let transitions = self.get_transitions(dp);

            Some((solution_cost, transitions))
        } else {
            None
        }
    }

    /// Returns whether the nodes are ordered by dual bound values.
    fn ordered_by_bound() -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockDp;

    impl Dp for MockDp {
        type State = i32;
        type CostType = i32;

        fn get_target(&self) -> Self::State {
            2
        }

        fn get_successors(
            &self,
            _: &Self::State,
        ) -> impl IntoIterator<Item = (Self::State, Self::CostType, usize)> {
            vec![]
        }

        fn get_base_cost(&self, state: &Self::State) -> Option<Self::CostType> {
            if *state == 0 {
                Some(0)
            } else {
                None
            }
        }

        fn combine_cost_weights(&self, a: Self::CostType, b: Self::CostType) -> Self::CostType {
            a + b
        }
    }

    struct MockNode(i32);

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
            2
        }

        fn get_bound(&self, _: &Self::DpData) -> Option<Self::CostType> {
            Some(0)
        }

        fn is_closed(&self) -> bool {
            false
        }

        fn close(&self) {}

        fn get_transitions(&self, _: &Self::DpData) -> Vec<usize> {
            vec![0, 1]
        }
    }

    #[test]
    fn test_check_solution() {
        let dp = MockDp;
        let node = MockNode(0);

        assert_eq!(node.check_solution(&dp), Some((2, vec![0, 1])));
    }

    #[test]
    fn test_check_solution_none() {
        let dp = MockDp;
        let node = MockNode(1);

        assert_eq!(node.check_solution(&dp), None);
    }
}
