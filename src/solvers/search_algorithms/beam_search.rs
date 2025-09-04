use super::SearchParameters;
use super::search::Solution;
use super::search_nodes::{SearchNode, StateRegistry};
use crate::dp::{Dominance, DpMut};
use crate::timer::Timer;
use smallvec::SmallVec;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, binary_heap};
use std::fmt::Display;
use std::hash::Hash;
use std::mem;
use std::rc::Rc;

/// Beam that keeps the best nodes.
pub struct Beam<N> {
    beam_width: usize,
    size: usize,
    queue: BinaryHeap<Reverse<Rc<N>>>,
}

/// Result of the beam insertion.
#[derive(Debug, Clone)]
pub struct BeamInsertionResult<N> {
    /// The given node is inserted into the beam.
    pub is_inserted: bool,
    /// The given node is newly registered to the state registry.
    pub is_newly_registered: bool,
    /// The given node is not inserted into the beam due to the beam width.
    pub is_pruned: bool,
    /// A node dominated by the given node.
    pub dominated: SmallVec<[Rc<N>; 1]>,
    /// A node removed from the beam.
    pub removed: Option<Rc<N>>,
}

/// Iterator over the beam.
pub struct BeamDrain<'a, N> {
    iter: binary_heap::Drain<'a, Reverse<Rc<N>>>,
}

impl<N, D, S, C> Iterator for BeamDrain<'_, N>
where
    N: Ord + SearchNode<DpData = D, State = S, CostType = C>,
{
    type Item = Rc<N>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            Some(node) if node.0.is_closed() => self.next(),
            Some(node) => {
                node.0.close();
                Some(node.0)
            }
            None => None,
        }
    }
}

impl<N, K, D, S, C> Beam<N>
where
    N: Ord + SearchNode<DpData = D, State = S, CostType = C>,
    K: Hash + Eq,
    D: DpMut<State = S, CostType = C> + Dominance<State = S, Key = K>,
    C: Ord + Copy,
{
    /// Creates a new beam with the given beam width.
    #[inline]
    pub fn new(beam_width: usize) -> Self {
        Self {
            beam_width,
            size: 0,
            queue: BinaryHeap::with_capacity(beam_width),
        }
    }

    /// Returns `true` if the beam is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    fn clean_garbage(&mut self) {
        let mut peek = self.queue.peek();

        while peek.is_some_and(|node| node.0.is_closed()) {
            self.queue.pop();
            peek = self.queue.peek();
        }
    }

    /// Pops the worst node from the beam.
    pub fn pop(&mut self) -> Option<Rc<N>> {
        self.queue.pop().map(|node| {
            node.0.close();
            self.size -= 1;
            self.clean_garbage();

            node.0
        })
    }

    /// Drains the beam.
    pub fn drain(&mut self) -> BeamDrain<'_, N> {
        self.size = 0;

        BeamDrain {
            iter: self.queue.drain(),
        }
    }

    /// Inserts the given node into the beam.
    pub fn insert(
        &mut self,
        dp: &mut D,
        node: N,
        registry: &mut StateRegistry<K, N>,
    ) -> BeamInsertionResult<N> {
        let mut result = BeamInsertionResult {
            is_inserted: false,
            is_newly_registered: false,
            is_pruned: false,
            dominated: SmallVec::default(),
            removed: None,
        };

        if self.size < self.beam_width || self.queue.peek().is_none_or(|peek| node > *peek.0) {
            let insertion_result = registry.insert_if_not_dominated(dp, node);

            for d in insertion_result.dominated.iter() {
                if !d.is_closed() {
                    d.close();
                    self.size -= 1;
                    self.clean_garbage();
                }
            }

            result.dominated = insertion_result.dominated;

            if let Some(node) = insertion_result.inserted {
                if result.dominated.is_empty() {
                    result.is_newly_registered = true;
                }

                if self.size == self.beam_width {
                    result.removed = self.pop();
                }

                if self.size < self.beam_width {
                    self.queue.push(Reverse(node));
                    self.size += 1;
                    result.is_inserted = true;
                } else {
                    result.is_pruned = true;
                }
            }
        } else {
            result.is_pruned = true;
        }

        result
    }
}

/// Parameters for beam search.
pub struct BeamSearchParameters<C> {
    /// Beam width.
    pub beam_width: usize,
    /// Whether to keep all search layers in memory.
    pub keep_all_layers: bool,
    /// Search parameters.
    pub search_parameters: SearchParameters<C>,
}

impl<C> Default for BeamSearchParameters<C>
where
    SearchParameters<C>: Default,
{
    fn default() -> Self {
        Self {
            beam_width: 1,
            keep_all_layers: false,
            search_parameters: Default::default(),
        }
    }
}

/// Performs beam search.
///
/// Beam search expands the best `n` nodes at each layer, where `n` is the beam width.
///
/// `node_constructor` is a function that constructs a new search node from the given state,
/// cost, transition, parent node, and primal bound.
///
/// `solution_checker` is a function that checks whether the given node is a solution
/// and returns the cost and transitions if it is.
pub fn beam_search<D, S, C, L, K, N, F, G>(
    dp: &mut D,
    root_node: N,
    mut node_constructor: F,
    mut solution_checker: G,
    parameters: &BeamSearchParameters<C>,
) -> Solution<C, L>
where
    D: DpMut<State = S, CostType = C, Label = L> + Dominance<State = S, Key = K>,
    C: Ord + Copy + Display,
    L: Copy,
    K: Hash + Eq,
    N: Ord + SearchNode<DpData = D, State = S, CostType = C, Label = L>,
    F: FnMut(&mut D, S, C, L, &N, Option<C>) -> Option<N>,
    G: FnMut(&mut D, &N) -> Option<(C, Vec<L>)>,
{
    let timer = parameters
        .search_parameters
        .time_limit
        .map(Timer::with_time_limit)
        .unwrap_or_default();
    let quiet = parameters.search_parameters.quiet;
    let expansion_limit = parameters.search_parameters.expansion_limit;

    let mut solution = Solution {
        best_bound: parameters.search_parameters.dual_bound,
        generated: 1,
        ..Solution::default()
    };

    let mut current_beam = Beam::new(parameters.beam_width);
    let mut next_beam = Beam::new(parameters.beam_width);
    let mut registry = parameters
        .search_parameters
        .initial_registry_capacity
        .map(StateRegistry::with_capacity)
        .unwrap_or(StateRegistry::with_capacity(parameters.beam_width));
    current_beam.insert(dp, root_node, &mut registry);
    let mut successors = Vec::new();

    let mut primal_bound = parameters.search_parameters.primal_bound;
    let mut is_pruned = false;
    let mut removed_dual_bound = None;
    let mut layer_index = 0;

    while !current_beam.is_empty() {
        let mut layer_dual_bound = removed_dual_bound;

        for node in current_beam.drain() {
            if timer.check_time_limit() {
                if !quiet {
                    println!("Time limit reached.");
                }

                solution.time = timer.get_elapsed_time();
                solution.is_time_limit_reached = true;

                return solution;
            }

            if let (Some(dual_bound), Some(primal_bound)) = (node.get_bound(dp), primal_bound) {
                if !dp.is_better_cost(dual_bound, primal_bound) {
                    continue;
                }
            }

            if let Some((solution_cost, transitions)) = solution_checker(dp, &node) {
                if primal_bound.is_none_or(|bound| dp.is_better_cost(solution_cost, bound)) {
                    primal_bound = Some(solution_cost);
                    solution.cost = Some(solution_cost);
                    solution.transitions = transitions;

                    dp.notify_primal_bound(solution_cost);

                    if !quiet {
                        println!(
                            "New primal bound: {solution_cost}, expanded: {expanded}, generated: {generated}, elapsed time: {time}s.",
                            solution_cost = solution_cost,
                            expanded = solution.expanded,
                            generated = solution.generated,
                            time = timer.get_elapsed_time()
                        );
                    }
                }

                continue;
            }

            let state = node.get_state(dp);
            let cost = node.get_cost(dp);
            dp.get_successors(state, &mut successors);

            successors
                .drain(..)
                .for_each(|(successor_state, weight, transition)| {
                    let successor_cost = dp.combine_cost_weights(cost, weight);

                    if let Some(successor_node) = node_constructor(
                        dp,
                        successor_state,
                        successor_cost,
                        transition,
                        &node,
                        primal_bound,
                    ) {
                        let successor_bound = successor_node.get_bound(dp);
                        let result = next_beam.insert(dp, successor_node, &mut registry);

                        if !is_pruned && (result.is_pruned || result.removed.is_some()) {
                            is_pruned = true;
                        }

                        if let Some(bound) = successor_bound {
                            if layer_dual_bound
                                .is_none_or(|layer_bound| dp.is_better_cost(bound, layer_bound))
                            {
                                layer_dual_bound = Some(bound);
                            }

                            if result.is_pruned
                                && removed_dual_bound.is_none_or(|removed_bound| {
                                    dp.is_better_cost(bound, removed_bound)
                                })
                            {
                                removed_dual_bound = Some(bound);
                            }
                        }

                        if let Some(bound) =
                            result.removed.and_then(|removed| removed.get_bound(dp))
                        {
                            if removed_dual_bound
                                .is_none_or(|removed_bound| dp.is_better_cost(bound, removed_bound))
                            {
                                removed_dual_bound = Some(bound);
                            }
                        }

                        if result.is_newly_registered {
                            solution.generated += 1;
                        }
                    }
                });

            solution.expanded += 1;

            if expansion_limit.is_some_and(|limit| solution.expanded >= limit) {
                if !quiet {
                    println!("Expansion limit reached.");
                }

                solution.is_expansion_limit_reached = true;
                solution.time = timer.get_elapsed_time();

                return solution;
            }
        }

        if !quiet {
            println!(
                "Layer: {layer_index}, expanded: {expanded}, generated: {generated}, elapsed time: {time}s",
                expanded = solution.expanded,
                generated = solution.generated,
                time = timer.get_elapsed_time()
            );
        }

        if let Some(bound) = layer_dual_bound {
            if primal_bound.is_some_and(|primal_bound| dp.is_better_cost(primal_bound, bound)) {
                solution.best_bound = primal_bound;
                solution.is_optimal = solution.cost.is_some();
                solution.is_infeasible = solution.cost.is_none();
                solution.time = timer.get_elapsed_time();

                return solution;
            } else if solution
                .best_bound
                .is_none_or(|best_bound| dp.is_better_cost(best_bound, bound))
            {
                solution.best_bound = Some(bound);

                if !quiet {
                    println!("New dual bound: {bound}");
                }
            }
        }

        if solution.cost.is_some() {
            if solution.cost == solution.best_bound || (!is_pruned && next_beam.is_empty()) {
                solution.is_optimal = true;
                solution.best_bound = solution.cost;
            }

            solution.time = timer.get_elapsed_time();

            return solution;
        }

        mem::swap(&mut current_beam, &mut next_beam);

        if !parameters.keep_all_layers {
            registry.clear();
        }

        layer_index += 1;
    }

    if !is_pruned {
        solution.is_infeasible = true;
        solution.best_bound = None;
    }

    solution.time = timer.get_elapsed_time();

    solution
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dp::Dp;
    use std::cell::Cell;
    use std::cmp::Ordering;

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

    struct MockNode(i32, i32, Cell<bool>, Vec<usize>);

    impl SearchNode for MockNode {
        type DpData = MockDp;
        type State = i32;
        type CostType = i32;
        type Label = usize;

        fn get_state(&self, _: &Self::DpData) -> &Self::State {
            &self.0
        }

        fn get_state_mut(&mut self, _: &Self::DpData) -> &mut Self::State {
            &mut self.0
        }

        fn get_cost(&self, _: &Self::DpData) -> Self::CostType {
            self.1
        }

        fn get_bound(&self, _: &Self::DpData) -> Option<Self::CostType> {
            None
        }

        fn close(&self) {
            self.2.set(true)
        }

        fn is_closed(&self) -> bool {
            self.2.get()
        }

        fn get_transitions(&self, _: &Self::DpData) -> Vec<Self::Label> {
            self.3.clone()
        }
    }

    impl PartialEq for MockNode {
        fn eq(&self, other: &Self) -> bool {
            self.1 == other.1
        }
    }

    impl Eq for MockNode {}

    impl Ord for MockNode {
        fn cmp(&self, other: &Self) -> Ordering {
            other.1.cmp(&self.1)
        }
    }

    impl PartialOrd for MockNode {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    #[test]
    fn test_beam() {
        let mut dp = MockDp(2);
        let mut beam = Beam::new(2);
        assert!(beam.is_empty());
        let mut registry = StateRegistry::default();

        let node = MockNode(4, 6, Cell::new(false), vec![]);
        let result = beam.insert(&mut dp, node, &mut registry);
        assert!(result.is_inserted);
        assert!(result.is_newly_registered);
        assert!(!result.is_pruned);
        assert_eq!(result.dominated.len(), 0);
        assert!(result.removed.is_none());
        assert!(!beam.is_empty());

        // Dominated
        let node = MockNode(4, 8, Cell::new(false), vec![]);
        let result = beam.insert(&mut dp, node, &mut registry);
        assert!(!result.is_inserted);
        assert!(!result.is_newly_registered);
        assert!(!result.is_pruned);
        assert_eq!(result.dominated.len(), 0);
        assert!(result.removed.is_none());
        assert!(!beam.is_empty());

        // Incomparable
        let node = MockNode(2, 2, Cell::new(false), vec![]);
        let result = beam.insert(&mut dp, node, &mut registry);
        assert!(result.is_inserted);
        assert!(result.is_newly_registered);
        assert!(!result.is_pruned);
        assert_eq!(result.dominated.len(), 0);
        assert!(result.removed.is_none());
        assert!(!beam.is_empty());

        // Dominating
        let node = MockNode(4, 4, Cell::new(false), vec![]);
        let result = beam.insert(&mut dp, node, &mut registry);
        assert!(result.is_inserted);
        assert!(!result.is_newly_registered);
        assert!(!result.is_pruned);
        assert_eq!(result.dominated.len(), 1);
        let dominated = result.dominated.first().unwrap();
        assert_eq!(dominated.get_state(&dp), &4);
        assert_eq!(dominated.get_cost(&dp), 6);
        assert!(dominated.is_closed());
        assert!(result.removed.is_none());
        assert!(!beam.is_empty());

        // Pruned by size
        let node = MockNode(5, 5, Cell::new(false), vec![]);
        let result = beam.insert(&mut dp, node, &mut registry);
        assert!(!result.is_inserted);
        assert!(!result.is_newly_registered);
        assert!(result.is_pruned);
        assert_eq!(result.dominated.len(), 0);
        assert!(result.removed.is_none());
        assert!(!beam.is_empty());

        // Push out the worst node
        let node = MockNode(3, 3, Cell::new(false), vec![]);
        let result = beam.insert(&mut dp, node, &mut registry);
        assert!(result.is_inserted);
        assert!(result.is_newly_registered);
        assert!(!result.is_pruned);
        assert_eq!(result.dominated.len(), 0);
        assert!(result.removed.is_some());
        let removed = result.removed.unwrap();
        assert_eq!(removed.get_state(&dp), &4);
        assert_eq!(removed.get_cost(&dp), 4);
        assert!(removed.is_closed());
        assert!(!beam.is_empty());

        let node = beam.pop();
        assert!(node.is_some());
        let node = node.unwrap();
        assert_eq!(node.get_state(&dp), &3);
        assert_eq!(node.get_cost(&dp), 3);
        assert!(node.is_closed());
        assert!(!beam.is_empty());

        let node = beam.pop();
        assert!(node.is_some());
        let node = node.unwrap();
        assert_eq!(node.get_state(&dp), &2);
        assert_eq!(node.get_cost(&dp), 2);
        assert!(node.is_closed());
        assert!(beam.is_empty());
    }

    #[test]
    fn test_beam_drain() {
        let mut dp = MockDp(2);
        let mut beam = Beam::new(2);
        assert!(beam.is_empty());
        let mut registry = StateRegistry::default();

        let node = MockNode(4, 6, Cell::new(false), vec![]);
        beam.insert(&mut dp, node, &mut registry);

        let node = MockNode(2, 2, Cell::new(false), vec![]);
        beam.insert(&mut dp, node, &mut registry);

        assert!(!beam.is_empty());

        let mut nodes = beam.drain().collect::<Vec<_>>();
        assert_eq!(nodes.len(), 2);
        nodes.sort();
        assert_eq!(nodes[0].get_state(&dp), &4);
        assert_eq!(nodes[0].get_cost(&dp), 6);
        assert!(nodes[0].is_closed());
        assert_eq!(nodes[1].get_state(&dp), &2);
        assert_eq!(nodes[1].get_cost(&dp), 2);
        assert!(nodes[1].is_closed());

        assert!(beam.is_empty());
    }

    #[test]
    fn test_beam_search() {
        let mut dp = MockDp(2);
        let root_node = MockNode(
            Dp::get_target(&dp),
            Dp::get_identity_weight(&dp),
            Cell::new(false),
            Vec::new(),
        );
        let node_constructor = |_: &mut _, state, cost, transition, parent: &MockNode, _| {
            let mut transitions = parent.3.clone();
            transitions.push(transition);
            Some(MockNode(state, cost, Cell::new(false), transitions))
        };
        let solution_checker = |dp: &mut MockDp, node: &MockNode| {
            dp.get_base_cost(node.get_state(dp)).map(|cost| {
                (
                    Dp::combine_cost_weights(dp, node.get_cost(dp), cost),
                    node.3.clone(),
                )
            })
        };
        let parameters = BeamSearchParameters {
            beam_width: 2,
            search_parameters: SearchParameters {
                quiet: true,
                ..Default::default()
            },
            ..Default::default()
        };

        let solution = beam_search(
            &mut dp,
            root_node,
            &node_constructor,
            solution_checker,
            &parameters,
        );
        assert_eq!(solution.cost, Some(2));
        assert_eq!(solution.transitions, vec![1, 1]);
        assert!(!solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);
    }

    #[test]
    fn test_beam_search_infeasible() {
        let mut dp = MockDp(2);
        let root_node = MockNode(
            Dp::get_target(&dp),
            Dp::get_identity_weight(&dp),
            Cell::new(false),
            Vec::new(),
        );
        let node_constructor = |_: &mut _, state, cost, transition, parent: &MockNode, _| {
            let mut transitions = parent.3.clone();
            transitions.push(transition);
            Some(MockNode(state, cost, Cell::new(false), transitions))
        };
        let solution_checker = |dp: &mut MockDp, node: &MockNode| {
            dp.get_base_cost(node.get_state(dp)).map(|cost| {
                (
                    Dp::combine_cost_weights(dp, node.get_cost(dp), cost),
                    node.3.clone(),
                )
            })
        };
        let parameters = BeamSearchParameters {
            beam_width: 2,
            search_parameters: SearchParameters {
                primal_bound: Some(2),
                quiet: true,
                ..Default::default()
            },
            ..Default::default()
        };

        let solution = beam_search(
            &mut dp,
            root_node,
            &node_constructor,
            solution_checker,
            &parameters,
        );
        assert_eq!(solution.cost, None);
        assert_eq!(solution.transitions, vec![]);
        assert!(!solution.is_optimal);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);
    }
}
