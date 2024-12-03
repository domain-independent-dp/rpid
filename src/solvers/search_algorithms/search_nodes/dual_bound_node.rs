use super::id_tree::IdTree;
use super::SearchNode;
use crate::dp::{Dp, OptimizationMode};
use crate::Bound;
use std::cell::Cell;
use std::cmp::Ordering;
use std::marker::PhantomData;
use std::ops::Neg;
use std::rc::Rc;

/// Node ordered by the path dual bound (f-value) computed from the state dual bound (h-value).
///
/// Ties are broken by the h-value.
#[derive(Debug)]
pub struct DualBoundNode<D, S, C> {
    state: S,
    g: C,
    h: C,
    f: C,
    closed: Cell<bool>,
    transition_tree: Rc<IdTree>,
    _phantom: PhantomData<D>,
}

impl<D, S, C> DualBoundNode<D, S, C>
where
    D: Dp<State = S, CostType = C> + Bound<State = S, CostType = C>,
    C: Copy + Neg<Output = C>,
{
    fn compute_h_and_f(dp: &D, g: C, h: C, primal_bound: Option<C>) -> Option<(C, C)> {
        let f = dp.combine_cost_weights(g, h);

        if let Some(primal_bound) = primal_bound {
            if !dp.is_better_cost(f, primal_bound) {
                return None;
            }
        }

        match dp.get_optimization_mode() {
            OptimizationMode::Minimization => Some((-h, -f)),
            OptimizationMode::Maximization => Some((h, f)),
        }
    }

    /// Creates a new root node given the state, the cost, and a primal bound.
    ///
    /// Returns `None` if the dual bound is not better than the primal bound.
    pub fn create_root(dp: &D, state: S, cost: C, primal_bound: Option<C>) -> Option<Self> {
        let h = dp.get_dual_bound(&state)?;
        let (h, f) = Self::compute_h_and_f(dp, cost, h, primal_bound)?;

        Some(Self {
            state,
            g: cost,
            h,
            f,
            closed: Cell::new(false),
            transition_tree: Rc::new(IdTree::default()),
            _phantom: PhantomData,
        })
    }

    /// Creates a new child node given the state, the cost, the transition, the primal bound,
    /// and an optional node sharing the same state.
    ///
    /// Returns `None` if the dual bound is not better than the primal bound.
    pub fn create_child(
        &self,
        dp: &D,
        state: S,
        cost: C,
        transition: usize,
        primal_bound: Option<C>,
        other: Option<&Self>,
    ) -> Option<Self> {
        let h = match (other, dp.get_optimization_mode()) {
            (Some(other), OptimizationMode::Minimization) => -other.h,
            (Some(other), OptimizationMode::Maximization) => other.h,
            (None, _) => dp.get_dual_bound(&state)?,
        };
        let (h, f) = Self::compute_h_and_f(dp, cost, h, primal_bound)?;

        Some(Self {
            state,
            g: cost,
            h,
            f,
            closed: Cell::new(false),
            transition_tree: Rc::new(IdTree::create_child(
                self.transition_tree.clone(),
                transition,
            )),
            _phantom: PhantomData,
        })
    }
}

impl<D, S, C> Clone for DualBoundNode<D, S, C>
where
    S: Clone,
    C: Clone,
{
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            g: self.g.clone(),
            h: self.h.clone(),
            f: self.f.clone(),
            closed: self.closed.clone(),
            transition_tree: self.transition_tree.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<D, S, C> PartialEq for DualBoundNode<D, S, C>
where
    C: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f && self.h == other.h
    }
}

impl<D, S, C> Eq for DualBoundNode<D, S, C> where C: Eq {}

impl<D, S, C> Ord for DualBoundNode<D, S, C>
where
    C: Eq + Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        match self.f.cmp(&other.f) {
            Ordering::Equal => self.h.cmp(&other.h),
            ordering => ordering,
        }
    }
}

impl<D, S, C> PartialOrd for DualBoundNode<D, S, C>
where
    C: Eq + Ord,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<D, S, C> SearchNode for DualBoundNode<D, S, C>
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

    fn get_cost(&self, _: &Self::DpData) -> Self::CostType {
        self.g
    }

    fn get_bound(&self, dp: &Self::DpData) -> Option<Self::CostType> {
        match dp.get_optimization_mode() {
            OptimizationMode::Minimization => Some(-self.f),
            OptimizationMode::Maximization => Some(self.f),
        }
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

    fn ordered_by_bound() -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dp::{Bound, Dp, OptimizationMode};

    struct MockDp(OptimizationMode);

    impl Dp for MockDp {
        type State = i32;
        type CostType = i32;

        fn get_target(&self) -> i32 {
            0
        }

        fn get_successors(&self, _: &Self::State) -> impl IntoIterator<Item = (i32, i32, usize)> {
            vec![]
        }

        fn get_base_cost(&self, _: &Self::State) -> Option<Self::CostType> {
            None
        }

        fn get_optimization_mode(&self) -> OptimizationMode {
            self.0
        }
    }

    impl Bound for MockDp {
        type State = i32;
        type CostType = i32;

        fn get_dual_bound(&self, state: &Self::State) -> Option<Self::CostType> {
            if *state <= 3 {
                Some(3 - *state)
            } else if *state <= 5 {
                Some(0)
            } else {
                None
            }
        }
    }

    #[test]
    fn test_create_root_minimization() {
        let dp = MockDp(OptimizationMode::Minimization);
        let node = DualBoundNode::<_, _, i32>::create_root(&dp, 0, 1, None);

        assert!(node.is_some());
        let node = node.unwrap();
        assert_eq!(node.get_state(&dp), &0);
        assert_eq!(node.get_cost(&dp), 1);
        assert_eq!(node.get_bound(&dp), Some(4));
        assert!(!node.is_closed());
        assert_eq!(node.get_transitions(&dp), vec![]);
    }

    #[test]
    fn test_create_root_none_minimization() {
        let dp = MockDp(OptimizationMode::Minimization);
        let node = DualBoundNode::<_, _, i32>::create_root(&dp, 6, 1, None);

        assert!(node.is_none());
    }

    #[test]
    fn test_create_root_with_primal_bound_minimization() {
        let dp = MockDp(OptimizationMode::Minimization);
        let node = DualBoundNode::<_, _, i32>::create_root(&dp, 0, 1, Some(5));

        assert!(node.is_some());
        let node = node.unwrap();
        assert_eq!(node.get_state(&dp), &0);
        assert_eq!(node.get_cost(&dp), 1);
        assert_eq!(node.get_bound(&dp), Some(4));
        assert!(!node.is_closed());
        assert_eq!(node.get_transitions(&dp), vec![]);
    }

    #[test]
    fn test_create_root_with_primal_bound_none_minimization() {
        let dp = MockDp(OptimizationMode::Minimization);
        let node = DualBoundNode::<_, _, i32>::create_root(&dp, 0, 1, Some(4));

        assert!(node.is_none());
    }

    #[test]
    fn test_create_root_maximization() {
        let dp = MockDp(OptimizationMode::Maximization);
        let node = DualBoundNode::<_, _, i32>::create_root(&dp, 0, 1, None);

        assert!(node.is_some());
        let node = node.unwrap();
        assert_eq!(node.get_state(&dp), &0);
        assert_eq!(node.get_cost(&dp), 1);
        assert_eq!(node.get_bound(&dp), Some(4));
        assert!(!node.is_closed());
        assert_eq!(node.get_transitions(&dp), vec![]);
    }

    #[test]
    fn test_create_root_none_maximization() {
        let dp = MockDp(OptimizationMode::Maximization);
        let node = DualBoundNode::<_, _, i32>::create_root(&dp, 6, 1, None);

        assert!(node.is_none());
    }

    #[test]
    fn test_create_root_with_primal_bound_maximization() {
        let dp = MockDp(OptimizationMode::Maximization);
        let node = DualBoundNode::<_, _, i32>::create_root(&dp, 0, 1, Some(3));

        assert!(node.is_some());
        let node = node.unwrap();
        assert_eq!(node.get_state(&dp), &0);
        assert_eq!(node.get_cost(&dp), 1);
        assert_eq!(node.get_bound(&dp), Some(4));
        assert!(!node.is_closed());
        assert_eq!(node.get_transitions(&dp), vec![]);
    }

    #[test]
    fn test_create_root_with_primal_bound_none_maximization() {
        let dp = MockDp(OptimizationMode::Maximization);
        let node = DualBoundNode::<_, _, i32>::create_root(&dp, 0, 1, Some(4));

        assert!(node.is_none());
    }

    #[test]
    fn test_create_child_minimization() {
        let dp = MockDp(OptimizationMode::Minimization);
        let node = DualBoundNode::<_, _, i32>::create_root(&dp, 0, 1, None);
        assert!(node.is_some());
        let node = node.unwrap();
        let child = node.create_child(&dp, 1, 2, 0, None, None);

        assert!(child.is_some());
        let child = child.unwrap();
        assert_eq!(child.get_state(&dp), &1);
        assert_eq!(child.get_cost(&dp), 2);
        assert_eq!(child.get_bound(&dp), Some(4));
        assert!(!child.is_closed());
        assert_eq!(child.get_transitions(&dp), vec![0]);
    }

    #[test]
    fn test_create_child_none_minimization() {
        let dp = MockDp(OptimizationMode::Minimization);
        let node = DualBoundNode::<_, _, i32>::create_root(&dp, 0, 1, None);
        assert!(node.is_some());
        let node = node.unwrap();
        let child = node.create_child(&dp, 6, 2, 0, None, None);

        assert!(child.is_none());
    }

    #[test]
    fn test_create_child_with_primal_bound_minimization() {
        let dp = MockDp(OptimizationMode::Minimization);
        let node = DualBoundNode::<_, _, i32>::create_root(&dp, 0, 1, None);
        assert!(node.is_some());
        let node = node.unwrap();
        let child = node.create_child(&dp, 1, 2, 0, Some(5), None);

        assert!(child.is_some());
        let child = child.unwrap();
        assert_eq!(child.get_state(&dp), &1);
        assert_eq!(child.get_cost(&dp), 2);
        assert_eq!(child.get_bound(&dp), Some(4));
        assert!(!child.is_closed());
        assert_eq!(child.get_transitions(&dp), vec![0]);
    }

    #[test]
    fn test_create_child_with_primal_bound_none_minimization() {
        let dp = MockDp(OptimizationMode::Minimization);
        let node = DualBoundNode::<_, _, i32>::create_root(&dp, 0, 1, None);
        assert!(node.is_some());
        let node = node.unwrap();
        let child = node.create_child(&dp, 1, 2, 0, Some(4), None);

        assert!(child.is_none());
    }

    #[test]
    fn test_create_child_with_other_minimization() {
        let dp = MockDp(OptimizationMode::Minimization);
        let node = DualBoundNode::<_, _, i32>::create_root(&dp, 0, 1, None);
        assert!(node.is_some());
        let node = node.unwrap();
        let other = DualBoundNode::<_, _, i32>::create_root(&dp, 1, 3, None);
        assert!(other.is_some());
        let other = other.unwrap();
        let child = node.create_child(&dp, 1, 2, 0, None, Some(&other));

        assert!(child.is_some());
        let child = child.unwrap();
        assert_eq!(child.get_state(&dp), &1);
        assert_eq!(child.get_cost(&dp), 2);
        assert_eq!(child.get_bound(&dp), Some(4));
        assert!(!child.is_closed());
        assert_eq!(child.get_transitions(&dp), vec![0]);
    }

    #[test]
    fn test_create_child_maximization() {
        let dp = MockDp(OptimizationMode::Maximization);
        let node = DualBoundNode::<_, _, i32>::create_root(&dp, 0, 1, None);
        assert!(node.is_some());
        let node = node.unwrap();
        let child = node.create_child(&dp, 1, 2, 0, None, None);

        assert!(child.is_some());
        let child = child.unwrap();
        assert_eq!(child.get_state(&dp), &1);
        assert_eq!(child.get_cost(&dp), 2);
        assert_eq!(child.get_bound(&dp), Some(4));
        assert!(!child.is_closed());
        assert_eq!(child.get_transitions(&dp), vec![0]);
    }

    #[test]
    fn test_create_child_none_maximization() {
        let dp = MockDp(OptimizationMode::Maximization);
        let node = DualBoundNode::<_, _, i32>::create_root(&dp, 0, 1, None);
        assert!(node.is_some());
        let node = node.unwrap();
        let child = node.create_child(&dp, 6, 2, 0, None, None);

        assert!(child.is_none());
    }

    #[test]
    fn test_create_child_with_primal_bound_maximization() {
        let dp = MockDp(OptimizationMode::Maximization);
        let node = DualBoundNode::<_, _, i32>::create_root(&dp, 0, 1, None);
        assert!(node.is_some());
        let node = node.unwrap();
        let child = node.create_child(&dp, 1, 2, 0, Some(3), None);

        assert!(child.is_some());
        let child = child.unwrap();
        assert_eq!(child.get_state(&dp), &1);
        assert_eq!(child.get_cost(&dp), 2);
        assert_eq!(child.get_bound(&dp), Some(4));
        assert!(!child.is_closed());
        assert_eq!(child.get_transitions(&dp), vec![0]);
    }

    #[test]
    fn test_create_child_with_primal_bound_none_maximization() {
        let dp = MockDp(OptimizationMode::Maximization);
        let node = DualBoundNode::<_, _, i32>::create_root(&dp, 0, 1, None);
        assert!(node.is_some());
        let node = node.unwrap();
        let child = node.create_child(&dp, 1, 2, 0, Some(4), None);

        assert!(child.is_none());
    }

    #[test]
    fn test_create_child_with_other_maximization() {
        let dp = MockDp(OptimizationMode::Maximization);
        let node = DualBoundNode::<_, _, i32>::create_root(&dp, 0, 1, None);
        assert!(node.is_some());
        let node = node.unwrap();
        let other = DualBoundNode::<_, _, i32>::create_root(&dp, 1, 3, None);
        assert!(other.is_some());
        let other = other.unwrap();
        let child = node.create_child(&dp, 1, 2, 0, None, Some(&other));

        assert!(child.is_some());
        let child = child.unwrap();
        assert_eq!(child.get_state(&dp), &1);
        assert_eq!(child.get_cost(&dp), 2);
        assert_eq!(child.get_bound(&dp), Some(4));
        assert!(!child.is_closed());
        assert_eq!(child.get_transitions(&dp), vec![0]);
    }

    #[test]
    fn test_clone() {
        let dp = MockDp(OptimizationMode::Minimization);
        let node = DualBoundNode::<_, _, i32>::create_root(&dp, 0, 1, None);
        assert!(node.is_some());
        let node = node.unwrap();
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
        let node = DualBoundNode::<_, _, i32>::create_root(&dp, 0, 1, None);
        assert!(node.is_some());
        let mut node = node.unwrap();

        *node.get_state_mut(&dp) = 1;
        assert_eq!(node.get_state(&dp), &1);
    }

    #[test]
    fn test_close() {
        let dp = MockDp(OptimizationMode::Minimization);
        let node = DualBoundNode::<_, _, i32>::create_root(&dp, 0, 1, None);
        assert!(node.is_some());
        let node = node.unwrap();

        assert!(!node.is_closed());
        node.close();
        assert!(node.is_closed());
    }

    #[test]
    fn test_ord_minimization() {
        let dp = MockDp(OptimizationMode::Minimization);
        let node1 = DualBoundNode::<_, _, i32>::create_root(&dp, 3, 2, None);
        assert!(node1.is_some());
        let node1 = node1.unwrap();
        let node2 = DualBoundNode::<_, _, i32>::create_root(&dp, 4, 2, None);
        assert!(node2.is_some());
        let node2 = node2.unwrap();
        let node3 = DualBoundNode::<_, _, i32>::create_root(&dp, 2, 1, None);
        assert!(node3.is_some());
        let node3 = node3.unwrap();
        let node4 = DualBoundNode::<_, _, i32>::create_root(&dp, 2, 0, None);
        assert!(node4.is_some());
        let node4 = node4.unwrap();

        assert!(node1 == node1);
        assert!(node1 == node2);
        assert!(node1 > node3);
        assert!(node1 >= node3);
        assert!(node1 < node4);
        assert!(node1 <= node4);
    }

    #[test]
    fn test_ord_maximization() {
        let dp = MockDp(OptimizationMode::Maximization);
        let node1 = DualBoundNode::<_, _, i32>::create_root(&dp, 3, 2, None);
        assert!(node1.is_some());
        let node1 = node1.unwrap();
        let node2 = DualBoundNode::<_, _, i32>::create_root(&dp, 4, 2, None);
        assert!(node2.is_some());
        let node2 = node2.unwrap();
        let node3 = DualBoundNode::<_, _, i32>::create_root(&dp, 2, 1, None);
        assert!(node3.is_some());
        let node3 = node3.unwrap();
        let node4 = DualBoundNode::<_, _, i32>::create_root(&dp, 2, 0, None);
        assert!(node4.is_some());
        let node4 = node4.unwrap();

        assert!(node1 == node1);
        assert!(node1 == node2);
        assert!(node1 < node3);
        assert!(node1 <= node3);
        assert!(node1 > node4);
        assert!(node1 >= node4);
    }
}
