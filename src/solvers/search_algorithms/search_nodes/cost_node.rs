use super::id_tree::IdTree;
use super::SearchNode;
use crate::dp::{Dp, OptimizationMode};
use std::cell::Cell;
use std::cmp::Ordering;
use std::marker::PhantomData;
use std::ops::Neg;
use std::rc::Rc;

/// Node ordered by the cost.
pub struct CostNode<D, S, C> {
    state: S,
    cost: C,
    closed: Cell<bool>,
    transition_tree: Rc<IdTree>,
    _phantom: PhantomData<D>,
}

impl<D, S, C> CostNode<D, S, C>
where
    D: Dp<State = S, CostType = C>,
    C: Neg<Output = C>,
{
    /// Creates a new root node given the state and the cost.
    pub fn create_root(dp: &D, state: S, cost: C) -> Self {
        Self {
            state,
            cost: match dp.get_optimization_mode() {
                OptimizationMode::Minimization => -cost,
                OptimizationMode::Maximization => cost,
            },
            closed: Cell::new(false),
            transition_tree: Rc::new(IdTree::default()),
            _phantom: PhantomData,
        }
    }

    /// Creates a new child node given the state, the cost, and the transition.
    pub fn create_child(&self, dp: &D, state: S, cost: C, transition: usize) -> Self {
        Self {
            state,
            cost: match dp.get_optimization_mode() {
                OptimizationMode::Minimization => -cost,
                OptimizationMode::Maximization => cost,
            },
            closed: Cell::new(false),
            transition_tree: Rc::new(IdTree::create_child(
                self.transition_tree.clone(),
                transition,
            )),
            _phantom: PhantomData,
        }
    }
}

impl<D, S, C> Clone for CostNode<D, S, C>
where
    S: Clone,
    C: Clone,
{
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            cost: self.cost.clone(),
            closed: self.closed.clone(),
            transition_tree: self.transition_tree.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<D, S, C> SearchNode for CostNode<D, S, C>
where
    D: Dp<State = S, CostType = C>,
    C: Copy + Neg<Output = C>,
{
    type DpData = D;
    type State = S;
    type CostType = C;

    fn get_state(&self, _: &Self::DpData) -> &Self::State {
        &self.state
    }

    fn get_state_mut(&mut self, _: &Self::DpData) -> &mut Self::State {
        &mut self.state
    }

    fn get_cost(&self, dp: &Self::DpData) -> Self::CostType {
        match dp.get_optimization_mode() {
            OptimizationMode::Minimization => -self.cost,
            OptimizationMode::Maximization => self.cost,
        }
    }

    fn get_bound(&self, _: &Self::DpData) -> Option<Self::CostType> {
        None
    }

    fn is_closed(&self) -> bool {
        self.closed.get()
    }

    fn close(&self) {
        self.closed.set(true);
    }

    fn get_transitions(&self, _: &D) -> Vec<usize> {
        self.transition_tree.get_path()
    }
}

impl<D, S, C> PartialEq for CostNode<D, S, C>
where
    C: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl<D, S, C> Eq for CostNode<D, S, C> where C: Eq {}

impl<D, S, C> Ord for CostNode<D, S, C>
where
    C: Eq + Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost.cmp(&other.cost)
    }
}

impl<D, S, C> PartialOrd for CostNode<D, S, C>
where
    C: Eq + Ord,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dp::Dp;

    struct MockDp(OptimizationMode);

    impl Dp for MockDp {
        type State = i32;
        type CostType = i32;

        fn get_target(&self) -> i32 {
            0
        }

        #[allow(refining_impl_trait_internal)]
        fn get_successors(&self, _: &Self::State) -> Vec<(i32, i32, usize)> {
            vec![]
        }

        fn get_base_cost(&self, _: &Self::State) -> Option<Self::CostType> {
            None
        }

        fn get_optimization_mode(&self) -> OptimizationMode {
            self.0
        }
    }

    #[test]
    fn test_create_root_minimization() {
        let dp = MockDp(OptimizationMode::Minimization);
        let node = CostNode::<_, _, i32>::create_root(&dp, 0, 1);

        assert_eq!(node.get_state(&dp), &0);
        assert_eq!(node.get_cost(&dp), 1);
        assert_eq!(node.get_bound(&dp), None);
        assert!(!node.is_closed());
        assert_eq!(node.get_transitions(&dp), vec![]);
    }

    #[test]
    fn test_create_root_maximization() {
        let dp = MockDp(OptimizationMode::Maximization);
        let node = CostNode::<_, _, i32>::create_root(&dp, 0, 1);

        assert_eq!(node.get_state(&dp), &0);
        assert_eq!(node.get_cost(&dp), 1);
        assert_eq!(node.get_bound(&dp), None);
        assert!(!node.is_closed());
        assert_eq!(node.get_transitions(&dp), vec![]);
    }

    #[test]
    fn test_create_child_minimization() {
        let dp = MockDp(OptimizationMode::Minimization);
        let parent = CostNode::<_, _, i32>::create_root(&dp, 0, 1);
        let child = parent.create_child(&dp, 1, 2, 0);

        assert_eq!(child.get_state(&dp), &1);
        assert_eq!(child.get_cost(&dp), 2);
        assert_eq!(child.get_bound(&dp), None);
        assert!(!child.is_closed());
        assert_eq!(child.get_transitions(&dp), vec![0]);
    }

    #[test]
    fn test_create_child_maximization() {
        let dp = MockDp(OptimizationMode::Minimization);
        let parent = CostNode::<_, _, i32>::create_root(&dp, 0, 1);
        let child = parent.create_child(&dp, 1, 2, 0);

        assert_eq!(child.get_state(&dp), &1);
        assert_eq!(child.get_cost(&dp), 2);
        assert_eq!(child.get_bound(&dp), None);
        assert!(!child.is_closed());
        assert_eq!(child.get_transitions(&dp), vec![0]);
    }

    #[test]
    fn test_clone() {
        let dp = MockDp(OptimizationMode::Minimization);
        let node = CostNode::<_, _, i32>::create_root(&dp, 0, 1);
        let cloned = node.clone();
        assert_eq!(node.get_state(&dp), cloned.get_state(&dp));
        assert_eq!(node.get_cost(&dp), cloned.get_cost(&dp));
        assert_eq!(node.get_bound(&dp), cloned.get_bound(&dp));
        assert_eq!(node.is_closed(), cloned.is_closed());
        assert_eq!(node.get_transitions(&dp), cloned.get_transitions(&dp));
    }

    #[test]
    fn test_state_mut() {
        let dp = MockDp(OptimizationMode::Minimization);
        let mut node = CostNode::<_, _, i32>::create_root(&dp, 0, 1);

        *node.get_state_mut(&dp) = 1;
        assert_eq!(node.get_state(&dp), &1);
    }

    #[test]
    fn test_close() {
        let dp = MockDp(OptimizationMode::Minimization);
        let node = CostNode::<_, _, i32>::create_root(&dp, 0, 1);

        assert!(!node.is_closed());
        node.close();
        assert!(node.is_closed());
    }

    #[test]
    fn test_ord_minimization() {
        let dp = MockDp(OptimizationMode::Minimization);
        let node1 = CostNode::<_, _, i32>::create_root(&dp, 0, 1);
        let node2 = CostNode::<_, _, i32>::create_root(&dp, 1, 1);
        let node3 = CostNode::<_, _, i32>::create_root(&dp, 0, 2);

        assert!(node1 == node1);
        assert!(node1 == node2);
        assert!(node1 >= node3);
        assert!(node1 > node3);
    }

    #[test]
    fn test_ord_maximization() {
        let dp = MockDp(OptimizationMode::Maximization);
        let node1 = CostNode::<_, _, i32>::create_root(&dp, 0, 1);
        let node2 = CostNode::<_, _, i32>::create_root(&dp, 1, 1);
        let node3 = CostNode::<_, _, i32>::create_root(&dp, 0, 2);

        assert!(node1 == node1);
        assert!(node1 == node2);
        assert!(node1 <= node3);
        assert!(node1 < node3);
    }
}
