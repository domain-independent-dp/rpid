use super::ConcurrentStateRegistry;
use crate::dp::{Dominance, DpMut};
use crate::solvers::search_algorithms::SearchNode;
use std::hash::Hash;
use std::sync::Arc;

/// Iterator returning a node and its applicable transition.
pub struct SendableSuccessorIterator<'a, D, S, C, L, K, N, F>
where
    D: DpMut<State = S, CostType = C, Label = L> + Dominance<State = S, Key = K> + Clone + Send,
    S: Send + Sync,
    C: Ord + Copy + Send + Sync,
    L: Copy + Send + Sync,
    K: Hash + Eq,
    N: Ord + SearchNode<DpData = D, State = S, CostType = C, Label = L> + Send + Sync,
    F: Fn(&mut D, S, C, L, &N, Option<C>) -> Option<N> + Clone + Send,
{
    node: Arc<N>,
    dp: D,
    node_constructor: F,
    registry: &'a ConcurrentStateRegistry<K, N>,
    primal_bound: Option<C>,
    successors: Vec<(S, C, L)>,
}

impl<'a, D, S, C, L, K, N, F> SendableSuccessorIterator<'a, D, S, C, L, K, N, F>
where
    D: DpMut<State = S, CostType = C, Label = L> + Dominance<State = S, Key = K> + Clone + Send,
    S: Send + Sync,
    C: Ord + Copy + Send + Sync,
    L: Copy + Send + Sync,
    K: Hash + Eq,
    N: Ord + SearchNode<DpData = D, State = S, CostType = C, Label = L> + Send + Sync,
    F: Fn(&mut D, S, C, L, &N, Option<C>) -> Option<N> + Clone + Send,
{
    /// Creates a new iterator.
    pub fn new(
        node: Arc<N>,
        mut dp: D,
        node_constructor: F,
        registry: &'a ConcurrentStateRegistry<K, N>,
        primal_bound: Option<C>,
    ) -> Self {
        let mut successors = Vec::new();
        dp.get_successors(node.get_state(&dp), &mut successors);

        Self {
            node,
            dp,
            node_constructor,
            registry,
            primal_bound,
            successors,
        }
    }
}

impl<'a, D, S, C, L, K, N, F> Iterator for SendableSuccessorIterator<'a, D, S, C, L, K, N, F>
where
    D: DpMut<State = S, CostType = C, Label = L> + Dominance<State = S, Key = K> + Clone + Send,
    S: Send + Sync,
    C: Ord + Copy + Send + Sync,
    L: Copy + Send + Sync,
    K: Hash + Eq,
    N: Ord + SearchNode<DpData = D, State = S, CostType = C, Label = L> + Send + Sync,
    F: Fn(&mut D, S, C, L, &N, Option<C>) -> Option<N> + Clone + Send,
{
    type Item = Arc<N>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((successor_state, weight, transition)) = self.successors.pop() {
            let successor_cost = self
                .dp
                .combine_cost_weights(self.node.get_cost(&self.dp), weight);
            let successor_node = (self.node_constructor)(
                &mut self.dp,
                successor_state,
                successor_cost,
                transition,
                &self.node,
                self.primal_bound,
            );

            if let Some(successor) = successor_node {
                let insertion_result = self.registry.insert_if_not_dominated(&self.dp, successor);
                for d in insertion_result.dominated {
                    if !d.is_closed() {
                        d.close();
                    }
                }

                if insertion_result.inserted.is_some() {
                    insertion_result.inserted
                } else {
                    self.next()
                }
            } else {
                self.next()
            }
        } else {
            None
        }
    }
}
