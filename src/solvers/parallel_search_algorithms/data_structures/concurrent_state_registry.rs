use crate::solvers::search_algorithms::SearchNode;
use crate::dp::{Dominance, DpMut};
use dashmap::DashMap;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::hash::{BuildHasherDefault, Hash};
use rustc_hash::FxHasher;
use std::ops::Deref;
use std::sync::Arc;

/// Result of inserting a node into the concurrent registry.
pub struct ConcurrentInsertionResult<N> {
    /// The inserted node.
    pub inserted: Option<Arc<N>>,
    /// The nodes that were dominated by the inserted node.
    pub dominated: SmallVec<[Arc<N>; 1]>,
}

impl<N> Default for ConcurrentInsertionResult<N> {
    #[inline]
    fn default() -> Self {
        Self {
            inserted: None,
            dominated: SmallVec::default(),
        }
    }
}

struct ConcurrentRemoveResult<N> {
    dominated: SmallVec<[Arc<N>; 1]>,
    same_state_index: Option<usize>,
}

pub struct ConcurrentStateRegistry<K, N>
{
    map: DashMap<K, SmallVec<[Arc<N>; 1]>, BuildHasherDefault<FxHasher>>,
}

impl<K, N> Default for ConcurrentStateRegistry<K, N>
where 
    K: Eq + Hash,
{
    /// Creates a new state registry.
    #[inline]
    fn default() -> Self {
        ConcurrentStateRegistry {
            map: DashMap::default(),
        }
    }
}

impl<K, N, D, S, C> ConcurrentStateRegistry<K, N>
where 
    K: Eq + Hash,
    N: SearchNode<DpData = D, State = S, CostType = C>,
    D: DpMut<State = S, CostType = C> + Dominance<State = S, Key = K>,
    C: Ord + Copy,
{
    #[inline]
    pub fn with_capacity_and_shard_amount(
        capacity: usize,
        shard_amount: usize,
    ) -> ConcurrentStateRegistry<K, N> {
        ConcurrentStateRegistry {
            map: DashMap::with_capacity_and_hasher_and_shard_amount(
                capacity,
                BuildHasherDefault::<FxHasher>::default(),
                shard_amount,
            ),
        }
    }

    // /// Tries to reserve capacity.
    // #[inline]
    // pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
    //     self.map.try_reserve(additional)
    // }

    fn remove_dominated(
        list: &mut SmallVec<[Arc<N>; 1]>,
        dp: &D,
        state: &S,
        cost: C,
    ) -> Option<ConcurrentRemoveResult<N>> {
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

        Some(ConcurrentRemoveResult {
            dominated,
            same_state_index,
        })
    }

    /// Inserts a node into the registry if it is not dominated by any other node.
    pub fn insert_if_not_dominated(&self, dp: &D, mut node: N) -> ConcurrentInsertionResult<N> {
        let entry = self.map.entry(dp.get_key(node.get_state(dp)));
        match entry{
            dashmap::mapref::entry::Entry::Occupied(entry) => {
                // Update the key of the state by the already stored key to reduce memory usage.
                dp.update_key(node.get_state_mut(dp), entry.key());

                let mut list = entry.into_ref();
                let mut_list = list.value_mut();
                let result =
                    Self::remove_dominated(mut_list, dp, node.get_state(dp), node.get_cost(dp));

                if result.is_none() {
                    return ConcurrentInsertionResult::default();
                }

                let result = result.unwrap();
                let inserted = Arc::from(node);
                list.push(inserted.clone());

                ConcurrentInsertionResult {
                    inserted: Some(inserted),
                    dominated: result.dominated,
                }
            }
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                let inserted = Arc::new(node);
                entry.insert(SmallVec::from_vec(vec![inserted.clone()]));

                ConcurrentInsertionResult {
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
        &self,
        dp: &mut D,
        mut state: S,
        cost: C,
        constructor: impl FnOnce(&mut D, S, C, Option<&N>) -> Option<N>,
    ) -> ConcurrentInsertionResult<N> {
        let entry = self.map.entry(dp.get_key(&state));
        match entry{
            dashmap::mapref::entry::Entry::Occupied(entry) => {
                // Update the key of the state by the already stored key to reduce memory usage.
                dp.update_key(&mut state, entry.key());

                let mut list = entry.into_ref();
                let mut_list = list.value_mut();
                let result = Self::remove_dominated(mut_list, dp, &state, cost);

                if result.is_none() {
                    return ConcurrentInsertionResult::default();
                }

                let result = result.unwrap();
                let same_state_information =
                    result.same_state_index.map(|i| result.dominated[i].deref());
                let node = constructor(dp, state, cost, same_state_information);

                let inserted = if let Some(node) = node {
                    let inserted = Arc::from(node);
                    list.push(inserted.clone());

                    Some(inserted)
                } else {
                    None
                };

                ConcurrentInsertionResult {
                    inserted,
                    dominated: result.dominated,
                }
            }
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                if let Some(node) = constructor(dp, state, cost, None) {
                    let inserted = Arc::new(node);
                    entry.insert(SmallVec::from_vec(vec![inserted.clone()]));

                    ConcurrentInsertionResult {
                        inserted: Some(inserted),
                        dominated: SmallVec::default(),
                    }
                } else {
                    ConcurrentInsertionResult::default()
                }
            }
        }
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }
}