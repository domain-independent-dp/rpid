use crate::dp::{DpMut, OptimizationMode};
use crate::solvers::parallel_search_algorithms::ArcIdTree;
use crate::solvers::search_algorithms::{SearchNode, Sequence};
use std::cmp::Ordering;
use std::marker::PhantomData;
use std::ops::{Deref, Neg};
use std::sync::{Arc, atomic};

/// Node ordered by the cost.
pub struct SendableCostNode<D, S, C, L, T = ArcIdTree<L>, P = Arc<T>> {
    state: S,
    cost: C,
    closed: atomic::AtomicBool,
    transition_tree: P,
    _phantom: PhantomData<(D, L, T)>,
}

impl<D, S, C, L, T, P> SendableCostNode<D, S, C, L, T, P>
where
    D: DpMut<State = S, CostType = C>,
    C: Neg<Output = C>,
    T: Default + Sequence<L, P>,
    P: From<T> + Clone,
{
    /// Creates a new root node given the state and the cost.
    pub fn create_root(dp: &D, state: S, cost: C) -> Self {
        Self {
            state,
            cost: match dp.get_optimization_mode() {
                OptimizationMode::Minimization => -cost,
                OptimizationMode::Maximization => cost,
            },
            closed: atomic::AtomicBool::new(false),
            transition_tree: P::from(T::default()),
            _phantom: PhantomData,
        }
    }

    /// Creates a new child node given the state, the cost, and the transition.
    pub fn create_child(&self, dp: &D, state: S, cost: C, transition: L) -> Self {
        Self {
            state,
            cost: match dp.get_optimization_mode() {
                OptimizationMode::Minimization => -cost,
                OptimizationMode::Maximization => cost,
            },
            closed: atomic::AtomicBool::new(false),
            transition_tree: P::from(T::create_child(
                self.transition_tree.clone(),
                transition,
            )),
            _phantom: PhantomData,
        }
    }
}

impl<D, S, C, L, T, P> Clone for SendableCostNode<D, S, C, L, T, P>
where
    S: Clone,
    C: Clone,
    P: Clone,
{
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            cost: self.cost.clone(),
            closed: atomic::AtomicBool::new(self.closed.load(atomic::Ordering::Relaxed)),
            transition_tree: self.transition_tree.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<D, S, C, L, T, P> SearchNode for SendableCostNode<D, S, C, L, T, P>
where
    D: DpMut<State = S, CostType = C, Label = L>,
    C: Copy + Neg<Output = C>,
    T: Sequence<L, P>,
    P: Deref<Target=T>,
{
    type DpData = D;
    type State = S;
    type CostType = C;
    type Label = L;

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
        self.closed.load(atomic::Ordering::Relaxed)
    }

    fn close(&self) {
        self.closed.store(true, atomic::Ordering::Relaxed);
    }

    fn get_transitions(&self, _: &D) -> Vec<L> {
        self.transition_tree.get_path()
    }
}

impl<D, S, C, L, T, P> PartialEq for SendableCostNode<D, S, C, L, T, P>
where
    C: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl<D, S, C, L, T, P> Eq for SendableCostNode<D, S, C, L, T, P> where C: Eq {}

impl<D, S, C, L, T, P> Ord for SendableCostNode<D, S, C, L, T, P>
where
    C: Eq + Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost.cmp(&other.cost)
    }
}

impl<D, S, C, L, T, P> PartialOrd for SendableCostNode<D, S, C, L, T, P>
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
        type Label = usize;

        fn get_target(&self) -> i32 {
            0
        }

        #[allow(refining_impl_trait_internal)]
        fn get_successors(
            &self,
            _: &Self::State,
        ) -> Vec<(Self::State, Self::CostType, Self::Label)> {
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
        let node = SendableCostNode::<_, _, i32, usize>::create_root(&dp, 0, 1);

        assert_eq!(node.get_state(&dp), &0);
        assert_eq!(node.get_cost(&dp), 1);
        assert_eq!(node.get_bound(&dp), None);
        assert!(!node.is_closed());
        assert_eq!(node.get_transitions(&dp), vec![]);
    }

    #[test]
    fn test_create_root_maximization() {
        let dp = MockDp(OptimizationMode::Maximization);
        let node = SendableCostNode::<_, _, i32, usize>::create_root(&dp, 0, 1);

        assert_eq!(node.get_state(&dp), &0);
        assert_eq!(node.get_cost(&dp), 1);
        assert_eq!(node.get_bound(&dp), None);
        assert!(!node.is_closed());
        assert_eq!(node.get_transitions(&dp), vec![]);
    }

    #[test]
    fn test_create_child_minimization() {
        let dp = MockDp(OptimizationMode::Minimization);
        let parent = SendableCostNode::<_, _, i32, usize>::create_root(&dp, 0, 1);
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
        let parent = SendableCostNode::<_, _, i32, usize>::create_root(&dp, 0, 1);
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
        let node = SendableCostNode::<_, _, i32, usize>::create_root(&dp, 0, 1);
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
        let mut node = SendableCostNode::<_, _, i32, usize>::create_root(&dp, 0, 1);

        *node.get_state_mut(&dp) = 1;
        assert_eq!(node.get_state(&dp), &1);
    }

    #[test]
    fn test_close() {
        let dp = MockDp(OptimizationMode::Minimization);
        let node = SendableCostNode::<_, _, i32, usize>::create_root(&dp, 0, 1);

        assert!(!node.is_closed());
        node.close();
        assert!(node.is_closed());
    }

    #[test]
    fn test_ord_minimization() {
        let dp = MockDp(OptimizationMode::Minimization);
        let node1 = SendableCostNode::<_, _, i32, usize>::create_root(&dp, 0, 1);
        let node2 = SendableCostNode::<_, _, i32, usize>::create_root(&dp, 1, 1);
        let node3 = SendableCostNode::<_, _, i32, usize>::create_root(&dp, 0, 2);

        assert!(node1 == node1);
        assert!(node1 == node2);
        assert!(node1 >= node3);
        assert!(node1 > node3);
    }

    #[test]
    fn test_ord_maximization() {
        let dp = MockDp(OptimizationMode::Maximization);
        let node1 = SendableCostNode::<_, _, i32, usize>::create_root(&dp, 0, 1);
        let node2 = SendableCostNode::<_, _, i32, usize>::create_root(&dp, 1, 1);
        let node3 = SendableCostNode::<_, _, i32, usize>::create_root(&dp, 0, 2);

        assert!(node1 == node1);
        assert!(node1 == node2);
        assert!(node1 <= node3);
        assert!(node1 < node3);
    }
}
