use super::search_node_message::SearchNodeMessage;
use crate::dp::{DpMut, OptimizationMode};
use crate::solvers::parallel_search_algorithms::data_structure::arc_id_tree::ArcIdTree;
use crate::solvers::search_algorithms::CostNode;
use std::cmp::Ordering;
use std::marker::PhantomData;
use std::ops::Neg;
use std::sync::Arc;

/// Node ordered by the cost.
pub struct CostNodeMessage<S, C, L> {
    pub state: S,
    pub cost: C,
    pub transition_tree: Arc<ArcIdTree<L>>,
}

// impl<D, S, C, L> From<CostNodeMessage<S, C, L>> for CostNode<D, S, C, L> {
//     fn from(value: CostNodeMessage<S, C, L>) -> Self {
//         CostNode {
//             state: value.state,
//             cost: value.cost,
//             transition_tree: value.transition_tree,
//             _phantom: PhantomData
//         }
//     }
// }