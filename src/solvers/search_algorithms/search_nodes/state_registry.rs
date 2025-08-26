use super::SearchNode;
use crate::dp::{Dominance, DpMut};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::hash::Hash;
use std::ops::Deref;
use std::rc::Rc;

/// Data structure to store search nodes and remove dominated nodes.
pub struct StateRegistry<K, N> {
    map: FxHashMap<K, SmallVec<[Rc<N>; 1]>>,
}

impl<K, I> Default for StateRegistry<K, I> {
    fn default() -> Self {
        Self {
            map: FxHashMap::default(),
        }
    }
}

/// Result of inserting a node into the registry.
pub struct InsertionResult<N> {
    /// The inserted node.
    pub inserted: Option<Rc<N>>,
    /// The nodes that were dominated by the inserted node.
    pub dominated: SmallVec<[Rc<N>; 1]>,
}

impl<N> Default for InsertionResult<N> {
    #[inline]
    fn default() -> Self {
        Self {
            inserted: None,
            dominated: SmallVec::default(),
        }
    }
}

struct RemoveResult<N> {
    dominated: SmallVec<[Rc<N>; 1]>,
    same_state_index: Option<usize>,
}

impl<K, N, D, S, C> StateRegistry<K, N>
where
    K: Hash + Eq,
    N: SearchNode<DpData = D, State = S, CostType = C>,
    D: DpMut<State = S, CostType = C> + Dominance<State = S, Key = K>,
    C: Ord + Copy,
{
    /// Creates a new state registry with the given capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            map: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
        }
    }

    fn remove_dominated(
        list: &mut SmallVec<[Rc<N>; 1]>,
        dp: &mut D,
        state: &S,
        cost: C,
    ) -> Option<RemoveResult<N>> {
        let mut dominated_indices = SmallVec::<[usize; 1]>::default();
        let mut same_state_index = None;

        for (i, v) in list.iter().enumerate() {
            let other_cost = v.get_cost(dp);
            let other = v.get_state(dp);

            match dp.compare(state, other) {
                Some(Ordering::Less) | Some(Ordering::Equal)
                    if !dp.is_better_cost(cost, other_cost) =>
                {
                    return None;
                }
                Some(Ordering::Equal) => {
                    same_state_index = Some(dominated_indices.len());
                    dominated_indices.push(i);
                }
                Some(Ordering::Greater) if !dp.is_better_cost(other_cost, cost) => {
                    dominated_indices.push(i);
                }
                _ => {}
            }
        }

        let dominated = dominated_indices
            .into_iter()
            .rev()
            .map(|i| list.swap_remove(i))
            .collect::<SmallVec<_>>();
        let same_state_index = same_state_index.map(|i| dominated.len() - i - 1);

        Some(RemoveResult {
            dominated,
            same_state_index,
        })
    }

    /// Inserts a node into the registry if it is not dominated by any other node.
    pub fn insert_if_not_dominated(&mut self, dp: &mut D, mut node: N) -> InsertionResult<N> {
        match self.map.entry(dp.get_key(node.get_state(dp))) {
            Entry::Occupied(entry) => {
                // Update the key of the state by the already stored key to reduce memory usage.
                dp.update_key(node.get_state_mut(dp), entry.key());

                let list = entry.into_mut();
                let result =
                    Self::remove_dominated(list, dp, node.get_state(dp), node.get_cost(dp));

                if result.is_none() {
                    return InsertionResult::default();
                }

                let result = result.unwrap();
                let inserted = Rc::from(node);
                list.push(inserted.clone());

                InsertionResult {
                    inserted: Some(inserted),
                    dominated: result.dominated,
                }
            }
            Entry::Vacant(entry) => {
                let inserted = Rc::new(node);
                entry.insert(SmallVec::from_vec(vec![inserted.clone()]));

                InsertionResult {
                    inserted: Some(inserted),
                    dominated: SmallVec::default(),
                }
            }
        }
    }

    /// Inserts a node created from a state and a cost by a constructor into the registry if it is not dominated by any other node.
    ///
    /// The constructor may use the information of a node that has the same state as the new node.
    /// If the constructor returns `None`, the node is not inserted.
    pub fn insert_with_if_not_dominated(
        &mut self,
        dp: &mut D,
        mut state: S,
        cost: C,
        constructor: impl FnOnce(&mut D, S, C, Option<&N>) -> Option<N>,
    ) -> InsertionResult<N> {
        match self.map.entry(dp.get_key(&state)) {
            Entry::Occupied(entry) => {
                // Update the key of the state by the already stored key to reduce memory usage.
                dp.update_key(&mut state, entry.key());

                let list = entry.into_mut();
                let result = Self::remove_dominated(list, dp, &state, cost);

                if result.is_none() {
                    return InsertionResult::default();
                }

                let result = result.unwrap();
                let same_state_information =
                    result.same_state_index.map(|i| result.dominated[i].deref());
                let node = constructor(dp, state, cost, same_state_information);

                let inserted = if let Some(node) = node {
                    let inserted = Rc::from(node);
                    list.push(inserted.clone());

                    Some(inserted)
                } else {
                    None
                };

                InsertionResult {
                    inserted,
                    dominated: result.dominated,
                }
            }
            Entry::Vacant(entry) => {
                if let Some(node) = constructor(dp, state, cost, None) {
                    let inserted = Rc::new(node);
                    entry.insert(SmallVec::from_vec(vec![inserted.clone()]));

                    InsertionResult {
                        inserted: Some(inserted),
                        dominated: SmallVec::default(),
                    }
                } else {
                    InsertionResult::default()
                }
            }
        }
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dp::{Dominance, Dp};
    use crate::solvers::search_algorithms::CostNode;

    struct MockDp;

    impl Dp for MockDp {
        type State = (i32, i32, i32);
        type CostType = i32;
        type Label = usize;

        fn get_target(&self) -> Self::State {
            (0, 0, 0)
        }

        fn get_successors(
            &self,
            _: &Self::State,
        ) -> impl IntoIterator<Item = (Self::State, Self::CostType, Self::Label)> {
            vec![]
        }

        fn get_base_cost(&self, _: &Self::State) -> Option<Self::CostType> {
            None
        }
    }

    impl Dominance for MockDp {
        type State = (i32, i32, i32);
        type Key = i32;

        fn get_key(&self, state: &Self::State) -> Self::Key {
            state.0
        }

        fn compare(&self, a: &Self::State, b: &Self::State) -> Option<Ordering> {
            if a.1 == b.1 && a.2 == b.2 {
                Some(Ordering::Equal)
            } else if a.1 <= b.1 && a.2 <= b.2 {
                Some(Ordering::Greater)
            } else if a.1 >= b.1 && a.2 >= b.2 {
                Some(Ordering::Less)
            } else {
                None
            }
        }
    }

    #[test]
    fn test_insert_if_not_dominated() {
        let mut registry = StateRegistry::default();
        let mut dp = MockDp;

        let state = (7, 7, 7);
        let cost = 7;
        let node = CostNode::create_root(&dp, state, cost);
        let result = registry.insert_if_not_dominated(&mut dp, node);
        assert!(result.inserted.is_some());
        let node = result.inserted.unwrap();
        assert_eq!(node.get_state(&dp), &(7, 7, 7));
        assert_eq!(node.get_cost(&dp), 7);
        assert!(result.dominated.is_empty());

        // Different key.
        let state = (6, 8, 8);
        let cost = 8;
        let node = CostNode::create_root(&dp, state, cost);
        let result = registry.insert_if_not_dominated(&mut dp, node);
        assert!(result.inserted.is_some());
        let node = result.inserted.unwrap();
        assert_eq!(node.get_state(&dp), &(6, 8, 8));
        assert_eq!(node.get_cost(&dp), 8);
        assert!(result.dominated.is_empty());

        // Incomparable due to the state.
        let state = (7, 6, 6);
        let cost = 8;
        let node = CostNode::create_root(&dp, state, cost);
        let result = registry.insert_if_not_dominated(&mut dp, node);
        assert!(result.inserted.is_some());
        let node = result.inserted.unwrap();
        assert_eq!(node.get_state(&dp), &(7, 6, 6));
        assert_eq!(node.get_cost(&dp), 8);
        assert!(result.dominated.is_empty());

        // Incomparable due to the cost.
        let state = (7, 7, 8);
        let cost = 6;
        let node = CostNode::create_root(&dp, state, cost);
        let result = registry.insert_if_not_dominated(&mut dp, node);
        assert!(result.inserted.is_some());
        let node = result.inserted.unwrap();
        assert_eq!(node.get_state(&dp), &(7, 7, 8));
        assert_eq!(node.get_cost(&dp), 6);
        assert!(result.dominated.is_empty());

        // Dominated by the first node due to the cost.
        let state = (7, 7, 7);
        let cost = 8;
        let node = CostNode::create_root(&dp, state, cost);
        let result = registry.insert_if_not_dominated(&mut dp, node);
        assert!(result.inserted.is_none());
        assert!(result.dominated.is_empty());

        // Dominated by the first node due to the state.
        let state = (7, 8, 7);
        let cost = 7;
        let node = CostNode::create_root(&dp, state, cost);
        let result = registry.insert_if_not_dominated(&mut dp, node);
        assert!(result.inserted.is_none());
        assert!(result.dominated.is_empty());

        // Replaces two nodes.
        let state = (7, 7, 7);
        let cost = 6;
        let node = CostNode::create_root(&dp, state, cost);
        let result = registry.insert_if_not_dominated(&mut dp, node);
        assert!(result.inserted.is_some());
        let node = result.inserted.unwrap();
        assert_eq!(node.get_state(&dp), &(7, 7, 7));
        assert_eq!(node.get_cost(&dp), 6);
        assert_eq!(result.dominated.len(), 2);
        let mut dominated = result.dominated;
        dominated.sort_by_key(|n| n.get_cost(&dp));
        assert_eq!(dominated[0].get_state(&dp), &(7, 7, 8));
        assert_eq!(dominated[0].get_cost(&dp), 6);
        assert_eq!(dominated[1].get_state(&dp), &(7, 7, 7));
        assert_eq!(dominated[1].get_cost(&dp), 7);
    }

    #[test]
    fn test_insert_with_if_not_dominated() {
        let mut registry = StateRegistry::default();
        let mut dp = MockDp;
        let constructor =
            |dp: &mut _, state, cost, _: Option<&_>| Some(CostNode::create_root(dp, state, cost));

        let state = (7, 7, 7);
        let cost = 7;
        let result = registry.insert_with_if_not_dominated(&mut dp, state, cost, constructor);
        assert!(result.inserted.is_some());
        let node = result.inserted.unwrap();
        assert_eq!(node.get_state(&dp), &(7, 7, 7));
        assert_eq!(node.get_cost(&dp), 7);
        assert!(result.dominated.is_empty());

        // Different key.
        let state = (6, 8, 8);
        let cost = 8;
        let result = registry.insert_with_if_not_dominated(&mut dp, state, cost, constructor);
        assert!(result.inserted.is_some());
        let node = result.inserted.unwrap();
        assert_eq!(node.get_state(&dp), &(6, 8, 8));
        assert_eq!(node.get_cost(&dp), 8);
        assert!(result.dominated.is_empty());

        // Incomparable due to the state.
        let state = (7, 6, 6);
        let cost = 8;
        let result = registry.insert_with_if_not_dominated(&mut dp, state, cost, constructor);
        assert!(result.inserted.is_some());
        let node = result.inserted.unwrap();
        assert_eq!(node.get_state(&dp), &(7, 6, 6));
        assert_eq!(node.get_cost(&dp), 8);
        assert!(result.dominated.is_empty());

        // Incomparable due to the cost.
        let state = (7, 7, 8);
        let cost = 6;
        let result = registry.insert_with_if_not_dominated(&mut dp, state, cost, constructor);
        assert!(result.inserted.is_some());
        let node = result.inserted.unwrap();
        assert_eq!(node.get_state(&dp), &(7, 7, 8));
        assert_eq!(node.get_cost(&dp), 6);
        assert!(result.dominated.is_empty());

        // Dominated by the first node due to the cost.
        let state = (7, 7, 7);
        let cost = 8;
        let result = registry.insert_with_if_not_dominated(&mut dp, state, cost, constructor);
        assert!(result.inserted.is_none());
        assert!(result.dominated.is_empty());

        // Dominated by the first node due to the state.
        let state = (7, 8, 7);
        let cost = 7;
        let result = registry.insert_with_if_not_dominated(&mut dp, state, cost, constructor);
        assert!(result.inserted.is_none());
        assert!(result.dominated.is_empty());

        // Replaces two nodes.
        let state = (7, 7, 7);
        let cost = 6;
        let result = registry.insert_with_if_not_dominated(&mut dp, state, cost, constructor);
        assert!(result.inserted.is_some());
        let node = result.inserted.unwrap();
        assert_eq!(node.get_state(&dp), &(7, 7, 7));
        assert_eq!(node.get_cost(&dp), 6);
        assert_eq!(result.dominated.len(), 2);
        let mut dominated = result.dominated;
        dominated.sort_by_key(|n| n.get_cost(&dp));
        assert_eq!(dominated[0].get_state(&dp), &(7, 7, 8));
        assert_eq!(dominated[0].get_cost(&dp), 6);
        assert_eq!(dominated[1].get_state(&dp), &(7, 7, 7));
        assert_eq!(dominated[1].get_cost(&dp), 7);
    }
}
