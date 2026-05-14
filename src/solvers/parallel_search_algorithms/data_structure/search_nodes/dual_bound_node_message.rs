use crate::solvers::parallel_search_algorithms::data_structure::arc_id_tree::ArcIdTree;
use crate::dp::{Bound, Dp, OptimizationMode};
use super::search_node_message::SearchNodeMessage;
use crate::solvers::search_algorithms::{DualBoundNode, Sequence};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::Neg;
use std::sync::Arc;
use rustc_hash::FxHasher;

/// Node ordered by the path dual bound (f-value) computed from the state dual bound (h-value).
///
/// Ties are broken by the h-value.
#[derive(Debug)]
pub struct DualBoundNodeMessage<D, S, C, L> {
    state: S,
    pub g: C,
    h: C,
    f: C,
    transition_tree: Arc<ArcIdTree<L>>,
    _phantom: PhantomData<D>,
}

impl<D, S, C, L> DualBoundNodeMessage<D, S, C, L>
{
    pub fn new(state: S, g: C, h: C, f: C, transition_tree: Arc<ArcIdTree<L>>) -> Self {
        Self {
            state,
            g,
            h,
            f,
            transition_tree,
            _phantom: PhantomData,
        }
    }
}

impl<D, S, C, L> DualBoundNodeMessage<D, S, C, L>
where
    D: Dp<State = S, CostType = C> + Bound<State = S, CostType = C>,
    C: Copy + Neg<Output = C>,
    L: Default + Copy,
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
            transition_tree: Arc::new(ArcIdTree::default()),
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
        transition: L,
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
            transition_tree: Arc::new(ArcIdTree::create_child(
                self.transition_tree.clone(),
                transition,
            )),
            _phantom: PhantomData,
        })
    }
}

impl<D, S, C, L> Clone for DualBoundNodeMessage<D, S, C, L>
where
    S: Clone,
    C: Clone,
    L: Clone,
{
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            g: self.g.clone(),
            h: self.h.clone(),
            f: self.f.clone(),
            transition_tree: self.transition_tree.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<D, S, C, L> PartialEq for DualBoundNodeMessage<D, S, C, L>
where
    C: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f && self.h == other.h
    }
}

impl<D, S, C, L> Eq for DualBoundNodeMessage<D, S, C, L> where C: Eq {}

impl<D, S, C, L> Ord for DualBoundNodeMessage<D, S, C, L>
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

impl<D, S, C, L> PartialOrd for DualBoundNodeMessage<D, S, C, L>
where
    C: Eq + Ord,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<D, S, C, L> SearchNodeMessage for DualBoundNodeMessage<D, S, C, L>
where
    D: Send + Sync,
    S: Hash + Send + Sync,
    C: Send + Sync,
    L: Send + Sync,
{
    fn assign_thread(&self, threads: usize) -> usize {
        const SEED: u32 = 0x5583c24d;

        let mut hasher = FxHasher::default();
        hasher.write_u32(SEED);
        self.state.hash(&mut hasher);
        hasher.finish() as usize % threads
    }
}

impl<D, S, C, L> From<DualBoundNodeMessage<D, S, C, L>> 
    for DualBoundNode<D, S, C, L, ArcIdTree<L>, Arc<ArcIdTree<L>>> {
    fn from(value: DualBoundNodeMessage<D, S, C, L>) -> Self {
        DualBoundNode::new(
            value.state, 
            value.g,
            value.h,
            value.f,
            value.transition_tree
        )
    }
}