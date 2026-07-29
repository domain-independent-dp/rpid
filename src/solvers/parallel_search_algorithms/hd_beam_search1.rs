use super::super::search_algorithms::{
    Beam, BeamSearchParameters, SearchNode, Solution, StateRegistry,
};
use super::hd_search_statistics::{HdSearchResult, HdSearchStatistics};
use crate::dp::{Dominance, DpMut};
use crate::timer::Timer;
use bus::{Bus, BusReader};
use std::error::Error;
use std::fmt::Display;
use std::hash::Hash;
use std::sync::mpsc::{Receiver, Sender, SyncSender, channel, sync_channel};
use std::{cmp, iter, mem, thread};

/// Performs hash distributed beam search 1 (HDBS1).
///
/// It keeps the best `beam_size` nodes at each layer.
/// 
/// Type parameter `N` is a node type that implements `SearchNode`, and type parameter `M` is a node message type that is sendable
/// and can be transformed into a `N` node.
/// 
/// `node_constructor` is a function that constructs a new search node from the given state,
/// cost, transition, parent node, and primal bound.
///
/// `solution_checker` is a function that checks whether the given node is a solution
/// and returns the cost and transitions if it is.
///
/// # References
///
/// Ryo Kuroiwa and J. Christopher Beck. "Parallel Beam Search Algorithms for Domain-Independent Dynamic Programming,"
/// Proceedings of the 38th Annual AAAI Conference on Artificial Intelligence (AAAI), 2024.
///
/// # Panics
///
/// If it fails to create a thread pool or reserve memory for the state registry.
///
pub fn hd_beam_search1<D, S, C, L, K, N, M, F, G, A>(
    dp: &D,
    root_node: M,
    node_constructor: F,
    solution_checker: G,
    thread_assigner: A,
    parameters: &BeamSearchParameters<C>,
    threads: usize,
) -> Result<HdSearchResult<C, L>, Box<dyn Error>>
where
    D: DpMut<State = S, CostType = C, Label = L> + Dominance<State = S, Key = K> + Clone + Send,
    C: Ord + Copy + Display + Send + Sync,
    L: Copy + Send + Sync,
    K: Hash + Eq,
    N: Ord + SearchNode<DpData = D, State = S, CostType = C, Label = L> + From<M> + Send,
    M: Clone + Send,
    F: Fn(&mut D, S, C, L, &N, Option<C>) -> Option<M> + Clone + Send,
    G: Fn(&mut D, &N) -> Option<(C, Vec<L>)> + Clone + Send,
    A: Fn(&D, &M, usize) -> usize + Clone + Send,
{
    let threads = cmp::min(threads, parameters.beam_width);
    if !parameters.search_parameters.quiet {
        println!("HD1 Beam Search with {threads} threads.");
    }
    let base_beam_size = parameters.beam_width / threads;
    let modulo = parameters.beam_width % threads;

    let (node_txs, node_rxs): (Vec<_>, Vec<_>) = (0..threads).map(|_| channel()).unzip();
    let (solution_tx, solution_rx) = sync_channel(1);
    let (optimality_tx, optimality_rx) = sync_channel(1);
    let (statistics_tx, statistics_rx) = sync_channel(threads);

    let (local_layer_tx, local_layer_rx) = sync_channel(threads - 1);
    let mut global_layer_tx = Bus::new(1);
    let follower_channels = (0..threads - 1)
        .map(|_| LayerChannel::Follower(local_layer_tx.clone(), global_layer_tx.add_rx()))
        .collect::<Vec<_>>();
    let leader_channel = LayerChannel::Leader(local_layer_rx, global_layer_tx);
    let layer_channels = iter::once(leader_channel).chain(follower_channels);

    let parameters = *parameters;
    thread::scope(|s| {
        for (id, (node_rx, layer_channel)) in node_rxs.into_iter().zip(layer_channels).enumerate() {
            let node_txs = node_txs.clone();

            let solution_tx = solution_tx.clone();
            let optimality_tx = optimality_tx.clone();
            let statistics_tx = statistics_tx.clone();
            let channels = Channels {
                id,
                node_txs,
                node_rx,
                layer_channel,
                solution_tx,
                optimality_tx,
                statistics_tx,
            };

            let mut dp = dp.clone();
            let mut parameters = parameters;
            parameters.beam_width = base_beam_size + if id < modulo { 1 } else { 0 };
            let node_constructor = node_constructor.clone();
            let solution_checker = solution_checker.clone();
            let thread_assigner = thread_assigner.clone();
            let root_node = root_node.clone();

            s.spawn(move || {
                single_sync_beam_search1(
                    &mut dp,
                    root_node,
                    node_constructor,
                    solution_checker,
                    thread_assigner,
                    &parameters,
                    channels,
                )
            });
        }
    });

    let mut solution = Solution::default();

    if let Some((cost, transitions)) = solution_rx.recv()? {
        solution.cost = Some(cost);
        solution.transitions = transitions;
    }

    let optimality_message = optimality_rx.recv()?;

    if optimality_message.proved {
        solution.is_optimal = solution.cost.is_some();
        solution.is_infeasible = solution.cost.is_none();
    }

    solution.best_bound = optimality_message.bound;
    solution.is_time_limit_reached = optimality_message.time_out;

    let mut statistics = HdSearchStatistics {
        expanded: Vec::with_capacity(threads),
        generated: Vec::with_capacity(threads),
        kept: Vec::with_capacity(threads),
        sent: Vec::with_capacity(threads),
    };

    for _ in 0..threads {
        let information = statistics_rx.recv()?;
        solution.expanded += information.expanded;
        solution.generated += information.generated;
        statistics.expanded.push(information.expanded);
        statistics.generated.push(information.generated);
        statistics.kept.push(information.kept);
        statistics.sent.push(information.sent);
    }

    Ok((solution, statistics))
}

#[derive(Default, Clone)]
struct LocalLayerMessage<T> {
    id: usize,
    pruned: bool,
    is_empty: bool,
    bound: Option<T>,
    cost: Option<T>,
}

#[derive(Clone)]
enum GlobalLayerMessage<T> {
    Terminate(Option<usize>),
    Bound(Option<T>),
}

#[derive(Default)]
struct OptimalityMessage<T> {
    bound: Option<T>,
    proved: bool,
    time_out: bool,
}

#[derive(Default)]
struct Statistics {
    expanded: usize,
    generated: usize,
    kept: usize,
    sent: usize,
}

enum LayerChannel<T> {
    Leader(Receiver<LocalLayerMessage<T>>, Bus<GlobalLayerMessage<T>>),
    Follower(
        SyncSender<LocalLayerMessage<T>>,
        BusReader<GlobalLayerMessage<T>>,
    ),
}

struct Channels<C, M, L> {
    id: usize,
    node_txs: Vec<Sender<Option<M>>>,
    node_rx: Receiver<Option<M>>,
    layer_channel: LayerChannel<C>,
    solution_tx: SyncSender<Option<(C, Vec<L>)>>,
    optimality_tx: SyncSender<OptimalityMessage<C>>,
    statistics_tx: SyncSender<Statistics>,
}

fn single_sync_beam_search1<D, S, C, L, K, N, M, F, G, A>(
    dp: &mut D,
    root_node: M,
    mut node_constructor: F,
    mut solution_checker: G,
    thread_assigner: A,
    parameters: &BeamSearchParameters<C>,
    mut channels: Channels<C, M, L>,
) where
    D: DpMut<State = S, CostType = C, Label = L> + Dominance<State = S, Key = K>,
    C: Ord + Copy + Display + Sync,
    L: Copy,
    K: Hash + Eq,
    N: Ord + SearchNode<DpData = D, State = S, CostType = C, Label = L> + From<M>,
    M: Send,
    F: FnMut(&mut D, S, C, L, &N, Option<C>) -> Option<M>,
    G: FnMut(&mut D, &N) -> Option<(C, Vec<L>)>,
    A: Fn(&D, &M, usize) -> usize,
{
    let id = channels.id;
    let timer = if let LayerChannel::Leader(..) = &channels.layer_channel {
        Some(
            parameters
                .search_parameters
                .time_limit
                .map(Timer::with_time_limit)
                .unwrap_or_default(),
        )
    } else {
        None
    };
    let quiet = id != 0 || parameters.search_parameters.quiet;
    let mut primal_bound = parameters.search_parameters.primal_bound;
    let threads = channels.node_txs.len();

    let mut current_beam = Beam::new(parameters.beam_width);
    let mut next_beam = Beam::new(parameters.beam_width);
    let mut registry = parameters
        .search_parameters
        .initial_registry_capacity
        .map(StateRegistry::with_capacity)
        .unwrap_or(StateRegistry::with_capacity(parameters.beam_width));

    let mut sent = 0;
    let mut kept = 0;
    let mut generated = 0;

    if id == thread_assigner(dp, &root_node, threads) {
        current_beam.insert(dp, N::from(root_node), &mut registry);
        generated += 1;

        if !parameters.keep_all_layers {
            registry.clear();
        }
    }

    let mut successors = Vec::new();

    let mut expanded = 0;
    let mut is_pruned = false;
    let mut best_dual_bound = None;
    let mut removed_dual_bound = None;
    let mut layer_index = 0;

    loop {
        let mut incumbent = None;
        let mut layer_dual_bound = removed_dual_bound;

        {
            let mut expanded_all = false;
            let mut sent_all = false;
            let mut received_all = 0;

            let mut iter = current_beam.drain();

            while !sent_all || received_all < threads - 1 {
                if !expanded_all {
                    // Expands a node.
                    if let Some(node) = iter.next() {
                        if let (Some(dual_bound), Some(primal_bound)) =
                            (node.get_bound(dp), primal_bound)
                        {
                            if !dp.is_better_cost(dual_bound, primal_bound) {
                                continue;
                            }
                        }

                        if let Some((solution_cost, transitions)) = solution_checker(dp, &node) {
                            if primal_bound
                                .is_none_or(|bound| dp.is_better_cost(solution_cost, bound))
                            {
                                primal_bound = Some(solution_cost);
                                incumbent = Some((solution_cost, transitions));

                                dp.notify_primal_bound(solution_cost);

                                // Optimal solution, ignore remaining open nodes.
                                if Some(solution_cost) == best_dual_bound {
                                    expanded_all = true;
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
                                    let sent_to = thread_assigner(dp, &successor_node, threads);

                                    if sent_to == id {
                                        kept += 1;
                                        let successor_node = N::from(successor_node);
                                        let successor_bound = successor_node.get_bound(dp);
                                        let result =
                                            next_beam.insert(dp, successor_node, &mut registry);

                                        if !is_pruned
                                            && (result.is_pruned || result.removed.is_some())
                                        {
                                            is_pruned = true;
                                        }

                                        if let Some(bound) = successor_bound {
                                            if layer_dual_bound.is_none_or(|layer_bound| {
                                                dp.is_better_cost(bound, layer_bound)
                                            }) {
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
                                            if removed_dual_bound.is_none_or(|removed_bound| {
                                                dp.is_better_cost(bound, removed_bound)
                                            }) {
                                                removed_dual_bound = Some(bound);
                                            }
                                        }

                                        if result.is_newly_registered {
                                            generated += 1;
                                        }
                                    } else {
                                        channels.node_txs[sent_to]
                                            .send(Some(successor_node))
                                            .unwrap();
                                        sent += 1;
                                    }
                                }
                            });

                        expanded += 1;
                    } else {
                        expanded_all = true;
                    }
                }

                // Notifies the other threads that the current thread sent all nodes
                if expanded_all && !sent_all {
                    sent_all = true;
                    channels.node_txs.iter().enumerate().for_each(|(i, tx)| {
                        if i != id {
                            tx.send(None).unwrap()
                        }
                    });
                }

                if received_all < threads - 1 {
                    // Receives a node.
                    while let Ok(node) = channels.node_rx.try_recv() {
                        if let Some(node) = node {
                            let node = N::from(node);
                            let node_bound = node.get_bound(dp);
                            let result = next_beam.insert(dp, node, &mut registry);

                            if !is_pruned && (result.is_pruned || result.removed.is_some()) {
                                is_pruned = true;
                            }

                            if let Some(bound) = node_bound {
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
                                if removed_dual_bound.is_none_or(|removed_bound| {
                                    dp.is_better_cost(bound, removed_bound)
                                }) {
                                    removed_dual_bound = Some(bound);
                                }
                            }

                            if result.is_newly_registered {
                                generated += 1;
                            }
                        } else {
                            received_all += 1;
                        }
                    }
                }
            }
        }

        // Aggregates the information of the current layer.
        match &mut channels.layer_channel {
            LayerChannel::Follower(tx, rx) => {
                // Sends the information to the leader.
                let information = LocalLayerMessage {
                    id,
                    pruned: is_pruned,
                    is_empty: next_beam.is_empty(),
                    bound: layer_dual_bound,
                    cost: incumbent.as_ref().map(|(cost, _)| *cost),
                };
                tx.send(information).unwrap();

                // Receives the aggregated information from the leader.
                match rx.recv().unwrap() {
                    // Termination.
                    GlobalLayerMessage::Terminate(goal_id) => {
                        if Some(id) == goal_id {
                            let (cost, transitions) = incumbent.unwrap();

                            channels
                                .solution_tx
                                .send(Some((cost, transitions)))
                                .unwrap()
                        }

                        // Sends the statistics to the original thread.
                        channels
                            .statistics_tx
                            .send(Statistics {
                                expanded,
                                generated,
                                kept,
                                sent,
                            })
                            .unwrap();

                        return;
                    }
                    // Updates the dual bound.
                    GlobalLayerMessage::Bound(bound) => {
                        best_dual_bound = bound;
                    }
                }
            }
            LayerChannel::Leader(rx, tx) => {
                // Aggregates the information from all threads.
                let mut is_empty = next_beam.is_empty();
                let mut cost = incumbent.as_ref().map(|(cost, _)| *cost);
                let mut goal_id = if cost.is_some() { Some(id) } else { None };

                // Receives and aggregates the information from each follower.
                for _ in 0..threads - 1 {
                    let information = rx.recv().unwrap();
                    is_pruned |= information.pruned;
                    is_empty &= information.is_empty;

                    if let Some(bound) = information.bound {
                        if layer_dual_bound
                            .is_none_or(|layer_bound| dp.is_better_cost(bound, layer_bound))
                        {
                            layer_dual_bound = Some(bound);
                        }
                    }

                    if let Some(other_cost) = information.cost {
                        if cost.is_none_or(|incumbent_cost| {
                            dp.is_better_cost(other_cost, incumbent_cost)
                        }) || (Some(other_cost) == cost && information.id < goal_id.unwrap())
                        {
                            cost = Some(other_cost);
                            goal_id = Some(information.id);
                        }
                    }
                }

                if let Some(value) = layer_dual_bound {
                    if cost.is_some_and(|incumbent_cost| !dp.is_better_cost(value, incumbent_cost))
                    {
                        best_dual_bound = cost
                    } else if best_dual_bound
                        .is_none_or(|best_bound| dp.is_better_cost(best_bound, value))
                    {
                        best_dual_bound = Some(value);
                    }
                }

                let time_out = timer.as_ref().unwrap().check_time_limit();

                if !quiet {
                    println!(
                        "Searched layer: {}, elapsed time: {}",
                        layer_index,
                        timer.as_ref().unwrap().get_elapsed_time()
                    );
                }

                if is_empty || time_out || goal_id.is_some() {
                    let mut proved = !is_pruned && is_empty;

                    if let Some(goal_id) = goal_id {
                        if threads > 1 {
                            // Sends the termination signal to all followers
                            // with the id of the thread that finds the best solution.
                            tx.broadcast(GlobalLayerMessage::Terminate(Some(goal_id)));
                        }

                        if cost == best_dual_bound {
                            proved = true;
                        }

                        if goal_id == id {
                            let (cost, transitions) = incumbent.unwrap();

                            channels
                                .solution_tx
                                .send(Some((cost, transitions)))
                                .unwrap()
                        }
                    } else {
                        if threads > 1 {
                            // Sends the termination signal to all followers without a solution.
                            tx.broadcast(GlobalLayerMessage::Terminate(None));
                        }

                        // Sends no solution to the original thread.
                        channels.solution_tx.send(None).unwrap();

                        if proved {
                            best_dual_bound = None;
                        }
                    }

                    // Sends the statistics to the original thread.
                    channels
                        .statistics_tx
                        .send(Statistics {
                            expanded,
                            generated,
                            kept,
                            sent,
                        })
                        .unwrap();
                    // Sends the optimality information to the original thread.
                    channels
                        .optimality_tx
                        .send(OptimalityMessage {
                            bound: best_dual_bound,
                            proved,
                            time_out,
                        })
                        .unwrap();

                    return;
                } else if threads > 1 {
                    // Sends the dual bound to all followers.
                    tx.broadcast(GlobalLayerMessage::Bound(best_dual_bound));
                }
            }
        }

        mem::swap(&mut current_beam, &mut next_beam);

        if !parameters.keep_all_layers {
            registry.clear();
        }

        layer_index += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::super::hash_distribution;
    use super::*;
    use crate::dp::Dp;
    use crate::solvers::search_algorithms::*;
    use std::cell::Cell;
    use std::cmp::Ordering;

    #[derive(PartialEq, Eq, Clone)]
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

    #[derive(Clone)]
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
    fn test_hd_beam_search1() {
        let dp = MockDp(2);
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
        let thread_assigner = |dp: &MockDp, message: &MockNode, threads: usize| {
            hash_distribution::fx_hash_assign_thread(&dp.get_key(message.get_state(dp)), threads, 0)
        };
        let parameters = BeamSearchParameters {
            beam_width: 8,
            search_parameters: SearchParameters {
                quiet: true,
                ..Default::default()
            },
            ..Default::default()
        };

        let result = hd_beam_search1(
            &dp,
            root_node,
            node_constructor,
            solution_checker,
            thread_assigner,
            &parameters,
            8,
        );
        assert!(result.is_ok());
        let (solution, statistic) = result.unwrap();
        assert_eq!(solution.cost, Some(2));
        assert_eq!(solution.transitions, vec![1, 1]);
        assert!(!solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);

        assert_eq!(statistic.expanded.len(), 8);
        assert_eq!(statistic.generated.len(), 8);
        assert_eq!(statistic.kept.len(), 8);
        assert_eq!(statistic.sent.len(), 8);
    }

    #[test]
    fn test_hd_beam_search1_infeasible() {
        let dp = MockDp(2);
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
        let thread_assigner = |dp: &MockDp, message: &MockNode, threads: usize| {
            hash_distribution::fx_hash_assign_thread(&dp.get_key(message.get_state(dp)), threads, 0)
        };
        let parameters = BeamSearchParameters {
            beam_width: 8,
            search_parameters: SearchParameters {
                primal_bound: Some(2),
                quiet: true,
                ..Default::default()
            },
            ..Default::default()
        };

        let result = hd_beam_search1(
            &dp,
            root_node,
            node_constructor,
            solution_checker,
            thread_assigner,
            &parameters,
            8,
        );
        assert!(result.is_ok());
        let (solution, statistic) = result.unwrap();
        assert_eq!(solution.cost, None);
        assert_eq!(solution.transitions, vec![]);
        assert!(!solution.is_optimal);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);

        assert_eq!(statistic.expanded.len(), 8);
        assert_eq!(statistic.generated.len(), 8);
        assert_eq!(statistic.kept.len(), 8);
        assert_eq!(statistic.sent.len(), 8);
    }
}
