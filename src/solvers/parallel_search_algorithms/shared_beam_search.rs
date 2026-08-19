use super::super::search_algorithms::{
    BeamSearchParameters, SearchNode, Solution,
};
use super::data_structures::{ConcurrentStateRegistry, SendableSuccessorIterator};
use crate::OptimizationMode;
use crate::dp::{Dominance, DpMut};
use crate::timer::Timer;
use std::error::Error;
use std::fmt::Display;
use std::hash::Hash;
use rayon::prelude::*;

/// Performs shared beam search (SBS).
///
/// This function uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
///
/// It keeps the best `beam_size` nodes at each layer.
///
/// Type parameter `N` is a node type that implements `BfsNode`.
/// Type parameter `E` is a type of a function that evaluates a transition and generate a successor node.
/// The last argument of the function is the primal bound of the solution cost.
/// Type parameter `B` is a type of a function that combines the g-value (the cost to a state) and the base cost.
/// It should be the same function as the cost expression, e.g., `cost + base_cost` for `cost + w`.
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
pub fn shared_beam_search<D, S, C, L, K, N, F, G>(
    dp: &D,
    root_node: N,
    node_constructor: F,
    solution_checker: G,
    parameters: &BeamSearchParameters<C>,
    threads: usize,
) -> Result<Solution<C, L>, Box<dyn Error>>
where
    D: DpMut<State = S, CostType = C, Label = L> + Dominance<State = S, Key = K> + Clone + Send + Sync,
    S: Send + Sync,
    C: Ord + Copy + Display + Send + Sync,
    L: Copy + Send + Sync,
    K: Hash + Eq + Send + Sync,
    N: Ord + SearchNode<DpData = D, State = S, CostType = C, Label = L> + Send + Sync,
    F: Fn(&mut D, S, C, L, &N, Option<C>) -> Option<N> + Clone + Send + Sync,
    G: Fn(&mut D, &N) -> Option<(C, Vec<L>)> + Clone + Send + Sync,
{
    if !parameters.search_parameters.quiet {
        println!("HD2 Beam Search with {threads} threads.");
    }

    let timer = parameters
        .search_parameters
        .time_limit
        .map(Timer::with_time_limit)
        .unwrap_or_default();
    let beam_size = parameters.beam_width;
    let quiet = parameters.search_parameters.quiet;
    let mut primal_bound = parameters.search_parameters.primal_bound;

    let mut beam = Vec::with_capacity(beam_size);
    let capacity = parameters
        .search_parameters
        .initial_registry_capacity
        .unwrap_or_else(|| beam.capacity());
    let shard_amount = (threads * 4).next_power_of_two();
    let mut registry = ConcurrentStateRegistry::with_capacity_and_shard_amount(
        capacity,
        shard_amount,
    );

    let mut best_dual_bound = root_node.get_bound(dp);
    let insertion_result = registry.insert_if_not_dominated(dp, root_node);
    beam.push(insertion_result.inserted.unwrap());

    if !parameters.keep_all_layers {
        registry.clear();
    }

    let mut removed_dual_bound: Option<C> = None;

    let mut successors = Vec::with_capacity(beam_size);
    let mut non_dominated_successors = Vec::with_capacity(beam_size);
    let mut goal_information = Vec::with_capacity(beam_size);

    let mut expanded = 0;
    let mut generated = 1;
    let mut pruned = false;
    let mut layer_index = 0;

    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()?;

    while !beam.is_empty() {
        let beam_len = beam.len();

        let goal = thread_pool.install(|| {
            goal_information.par_extend(beam.par_drain(..).filter_map(|node| {
                node.close();

                if let Some((solution_cost, transitions)) = solution_checker(&mut dp.clone(), &node) {
                    if primal_bound
                        .is_none_or(|bound| dp.is_better_cost(solution_cost, bound))
                    {
                        Some((node, Some((solution_cost, transitions))))
                    } else {
                        None
                    }
                } else {
                    Some((node, None))
                }
            }));

            let filtered_goals = goal_information
                .par_iter()
                .filter_map(|(_, result)| {
                    if let Some((cost, transitions)) = result {
                        Some((cost, transitions))
                    } else {
                        None
                    }
                });

            if let Some((cost, transitions)) = 
                if dp.get_optimization_mode() == OptimizationMode::Minimization {
                    filtered_goals.min_by_key(|goal| goal.0)
                } else {
                    filtered_goals.max_by_key(|goal| goal.0)
                } 
            {
                primal_bound = Some(*cost);
                Some((*cost, transitions.clone()))
            } else {
                None
            }
        });

        if let (Some(best_dual_bound), Some(cost)) =
            (best_dual_bound, goal.as_ref().map(|(cost, _)| *cost))
        {
            if best_dual_bound == cost {
                return Ok(Solution {
                    cost: Some(cost),
                    best_bound: Some(best_dual_bound),
                    is_optimal: true,
                    transitions: goal.unwrap().1,
                    expanded,
                    generated,
                    time: timer.get_elapsed_time(),
                    ..Default::default()
                });
            }
        }

        if timer.check_time_limit() {
            return Ok(goal.map_or_else(
                || Solution {
                    expanded,
                    generated,
                    time: timer.get_elapsed_time(),
                    is_time_limit_reached: true,
                    ..Default::default()
                },
                |(cost, transitions)| Solution {
                    cost: Some(cost),
                    best_bound: best_dual_bound,
                    transitions,
                    expanded,
                    generated,
                    time: timer.get_elapsed_time(),
                    is_expansion_limit_reached: true,
                    ..Default::default()
                },
            ));
        }

        if !pruned || goal.is_none() {
            if goal.is_none() {
                expanded += beam_len;
            } else {
                thread_pool.install(|| {
                    expanded += goal_information.len();
                });
            }

            thread_pool.install(|| {
                successors.par_extend(
                    goal_information
                        .par_drain(..)
                        .filter_map(|(node, result)| {
                            if result.is_none() {
                                Some(SendableSuccessorIterator::new(
                                    node,
                                    dp.clone(),
                                    node_constructor.clone(),
                                    &registry,
                                    primal_bound
                                ))
                            } else {
                                None
                            }   
                        })
                        .flatten_iter(),
                );
                non_dominated_successors
                    .par_extend(successors.par_drain(..).filter(|node| !node.is_closed()));

                generated += non_dominated_successors.len();

                if non_dominated_successors.len() > beam_size {
                    non_dominated_successors.par_sort_unstable_by(|a, b| b.cmp(a));

                    if N::ordered_by_bound() {
                        if let Some(mut bound) = non_dominated_successors[0].get_bound(dp) {
                            if removed_dual_bound.is_some_and(|removed_bound| {
                                !dp.is_better_cost(bound, removed_bound)
                            }) {
                                bound = removed_dual_bound.unwrap();
                            }

                            if primal_bound.is_some_and(|primal_bound| {
                                !dp.is_better_cost(bound, primal_bound)
                            }) {
                                best_dual_bound = primal_bound;
                            } else if best_dual_bound.is_none_or(|best_bound| {
                                dp.is_better_cost(best_bound, bound)
                            }) {
                                best_dual_bound = Some(bound);
                            }
                        }

                        if let Some(bound) = non_dominated_successors[beam_size].get_bound(dp) {
                            if removed_dual_bound.is_none_or(|removed_bound| {
                                dp.is_better_cost(bound, removed_bound)
                            }) {
                                removed_dual_bound = Some(bound);
                            }
                        }
                    }

                    if !pruned {
                        pruned = true;
                    }

                    beam.par_extend(non_dominated_successors.par_drain(..beam_size));
                    non_dominated_successors.clear();
                } else {
                    beam.par_extend(non_dominated_successors.par_drain(..));
                }
            })
        }

        if !quiet {
            println!(
                "Searched layer: {}, expanded: {}, elapsed time: {}",
                layer_index,
                expanded,
                timer.get_elapsed_time()
            );
        }

        if let Some((cost, transitions)) = goal {
            let is_optimal = best_dual_bound == Some(cost) || (!pruned && beam.is_empty());

            return Ok(Solution {
                cost: Some(cost),
                best_bound: if is_optimal {
                    Some(cost)
                } else {
                    best_dual_bound
                },
                transitions,
                expanded,
                generated,
                is_optimal,
                time: timer.get_elapsed_time(),
                ..Default::default()
            });
        }

        if !parameters.keep_all_layers {
            registry.clear();
        }

        layer_index += 1;
    }

    Ok(Solution {
        is_infeasible: !pruned,
        best_bound: if pruned { best_dual_bound } else { None },
        expanded,
        generated,
        time: timer.get_elapsed_time(),
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dp::Dp;
    use crate::solvers::search_algorithms::*;
    use std::cmp::Ordering;
    use std::sync::atomic;

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

    // #[derive(Clone)]
    struct MockNode(i32, i32, atomic::AtomicBool, Vec<usize>);

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
            self.2.store(true, atomic::Ordering::Relaxed);
        }

        fn is_closed(&self) -> bool {
            self.2.load(atomic::Ordering::Relaxed)
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
    fn test_shared_beam_search() {
        let dp = MockDp(2);
        let root_node = MockNode(
            Dp::get_target(&dp),
            Dp::get_identity_weight(&dp),
            atomic::AtomicBool::new(false),
            Vec::new(),
        );
        let node_constructor = |_: &mut _, state, cost, transition, parent: &MockNode, _| {
            let mut transitions = parent.3.clone();
            transitions.push(transition);
            Some(MockNode(state, cost, atomic::AtomicBool::new(false), transitions))
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
            beam_width: 8,
            search_parameters: SearchParameters {
                quiet: true,
                ..Default::default()
            },
            ..Default::default()
        };

        let result = shared_beam_search(
            &dp,
            root_node, 
            node_constructor, 
            solution_checker, 
            &parameters, 
            8
        );
        assert!(result.is_ok());
        let solution = result.unwrap();
        assert_eq!(solution.cost, Some(2));
        assert_eq!(solution.transitions, vec![1, 1]);
        assert!(!solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);
    }

    #[test]
    fn test_shared_beam_search_infeasible() {
        let dp = MockDp(2);
        let root_node = MockNode(
            Dp::get_target(&dp),
            Dp::get_identity_weight(&dp),
            atomic::AtomicBool::new(false),
            Vec::new(),
        );
        let node_constructor = |_: &mut _, state, cost, transition, parent: &MockNode, _| {
            let mut transitions = parent.3.clone();
            transitions.push(transition);
            Some(MockNode(state, cost, atomic::AtomicBool::new(false), transitions))
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
            beam_width: 8,
            search_parameters: SearchParameters {
                primal_bound: Some(2),
                quiet: true,
                ..Default::default()
            },
            ..Default::default()
        };

        let result = shared_beam_search(
            &dp,
            root_node, 
            node_constructor, 
            solution_checker, 
            &parameters, 
            8
        );
        assert!(result.is_ok());
        let solution = result.unwrap();
        assert_eq!(solution.cost, None);
        assert_eq!(solution.transitions, vec![]);
        assert!(!solution.is_optimal);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);
    }
}
