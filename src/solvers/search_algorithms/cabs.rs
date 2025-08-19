use super::beam_search::BeamSearchParameters;
use super::search::{Search, SearchParameters, Solution};
use super::search_nodes::SearchNode;
use crate::dp::{Dominance, Dp};
use crate::timer::Timer;
use std::fmt::Display;
use std::hash::Hash;

/// Parameters for CABS.
pub struct CabsParameters {
    /// Initial beam width.
    pub initial_beam_width: usize,
    /// Maximum beam width.
    pub max_beam_width: Option<usize>,
    /// Whether to keep all layers.
    pub keep_all_layers: bool,
}

impl Default for CabsParameters {
    fn default() -> Self {
        Self {
            initial_beam_width: 1,
            max_beam_width: None,
            keep_all_layers: false,
        }
    }
}

/// Complete anytime beam search (CABS).
///
/// CABS repeats beam search while doubling the beam width until a termination condition is met.
pub struct Cabs<D, C, L, R, B> {
    dp: D,
    root_node_constructor: R,
    beam_search: B,
    parameters: SearchParameters<C>,
    cabs_parameters: CabsParameters,
    beam_width: usize,
    primal_bound: Option<C>,
    solution: Solution<C, L>,
    is_terminated: bool,
    timer: Timer,
}

impl<D, C, L, N, R, B, S, K> Cabs<D, C, L, R, B>
where
    D: Dp<State = S, CostType = C, Label = L> + Dominance<State = S, Key = K>,
    L: Clone,
    R: FnMut(&D, Option<C>) -> Option<N>,
    N: SearchNode<DpData = D, State = S, CostType = C, Label = L>,
    B: FnMut(&D, N, &BeamSearchParameters<C>) -> Solution<C, L>,
    C: Ord + Copy + Display,
    K: Hash + Eq,
{
    /// Creates a CABS solver.
    ///
    /// `root_node_constructor` is a function that constructs a root search node from the given state and primal bound.
    ///
    /// `beam_search` is a function that performs beam search from the given root node and beam search parameters.
    pub fn new(
        dp: D,
        root_node_constructor: R,
        beam_search: B,
        parameters: SearchParameters<C>,
        cabs_parameters: CabsParameters,
    ) -> Self {
        let mut timer = parameters
            .time_limit
            .map(Timer::with_time_limit)
            .unwrap_or_default();
        let beam_width = cabs_parameters.initial_beam_width;
        let primal_bound = parameters.primal_bound;
        let solution = Solution {
            best_bound: parameters.dual_bound,
            ..Default::default()
        };

        timer.stop();

        Self {
            dp,
            root_node_constructor,
            beam_search,
            parameters,
            cabs_parameters,
            beam_width,
            primal_bound,
            solution,
            is_terminated: false,
            timer,
        }
    }

    fn update_dual_bound_if_better(&mut self, dual_bound: C) -> bool {
        if self.solution.best_bound.is_none()
            || self
                .dp
                .is_better_cost(self.solution.best_bound.unwrap(), dual_bound)
        {
            if !self.parameters.quiet {
                println!(
                    "New dual bound: {}, expanded: {}, generated: {}, elapsed time: {}s.",
                    dual_bound,
                    self.solution.expanded,
                    self.solution.generated,
                    self.timer.get_elapsed_time()
                );
            }

            self.solution.best_bound = Some(dual_bound);

            true
        } else {
            false
        }
    }

    fn stop_timer_and_return_solution(&mut self) -> Solution<C, L> {
        self.solution.time = self.timer.get_elapsed_time();
        self.timer.stop();

        self.solution.clone()
    }
}

impl<D, C, L, N, R, B, S, K> Search for Cabs<D, C, L, R, B>
where
    D: Dp<State = S, CostType = C, Label = L> + Dominance<State = S, Key = K>,
    L: Clone,
    R: FnMut(&D, Option<C>) -> Option<N>,
    N: SearchNode<DpData = D, State = S, CostType = C, Label = L>,
    B: FnMut(&D, N, &BeamSearchParameters<C>) -> Solution<C, L>,
    C: Ord + Copy + Display,
    K: Hash + Eq,
{
    type CostType = C;
    type Label = L;

    fn search_next(&mut self) -> (Solution<Self::CostType, Self::Label>, bool) {
        self.timer.start();

        if self.is_terminated {
            return (self.stop_timer_and_return_solution(), true);
        }

        loop {
            let is_max_beam_width_reached =
                if let Some(max_beam_width) = self.cabs_parameters.max_beam_width {
                    if self.beam_width >= max_beam_width {
                        self.beam_width = max_beam_width;

                        true
                    } else {
                        false
                    }
                } else {
                    false
                };

            let root_node = (self.root_node_constructor)(&self.dp, self.primal_bound);

            if root_node.is_none() {
                self.solution.is_infeasible = true;
                self.is_terminated = true;

                return (self.stop_timer_and_return_solution(), true);
            }

            let root_node = root_node.unwrap();

            let expansion_limit = self
                .parameters
                .expansion_limit
                .map(|limit| limit - self.solution.expanded);

            let beam_search_parameters = BeamSearchParameters {
                beam_width: self.beam_width,
                keep_all_layers: self.cabs_parameters.keep_all_layers,
                search_parameters: SearchParameters {
                    primal_bound: self.primal_bound,
                    dual_bound: self.solution.best_bound,
                    get_all_solutions: self.parameters.get_all_solutions,
                    quiet: true,
                    time_limit: self.timer.get_remaining_time_limit(),
                    expansion_limit,
                    initial_registry_capacity: self.parameters.initial_registry_capacity,
                },
            };

            let solution = (self.beam_search)(&self.dp, root_node, &beam_search_parameters);

            self.solution.expanded += solution.expanded;
            self.solution.generated += solution.generated;

            if !self.parameters.quiet {
                println!(
                    "Searched with beam width {beam_width}, expanded: {expanded}, generated: {generated}, elapsed time: {time}s.",
                    beam_width = self.beam_width,
                    expanded = self.solution.expanded,
                    generated = self.solution.generated,
                    time = self.timer.get_elapsed_time()
                );
            }

            if let Some(bound) = solution.best_bound {
                self.update_dual_bound_if_better(bound);
            }

            if let Some(cost) = solution.cost {
                self.primal_bound = Some(cost);
                self.solution.cost = Some(cost);
                self.solution.transitions = solution.transitions;

                if solution.is_optimal {
                    self.solution.is_optimal = true;
                    self.is_terminated = true;
                } else {
                    self.beam_width *= 2;
                }

                if !self.parameters.quiet {
                    println!(
                        "New primal bound: {solution_cost}, expanded: {expanded}, generated: {generated}, elapsed time: {time}s.",
                        solution_cost = cost,
                        expanded = self.solution.expanded,
                        generated = self.solution.generated,
                        time = self.timer.get_elapsed_time()
                    );
                }

                return (self.stop_timer_and_return_solution(), self.is_terminated);
            } else if solution.is_infeasible {
                self.solution.is_optimal = self.solution.cost.is_some();
                self.solution.is_infeasible = self.solution.cost.is_none();
                self.solution.best_bound = self.solution.cost;
                self.is_terminated = true;

                return (self.stop_timer_and_return_solution(), true);
            }

            if solution.is_time_limit_reached {
                if !self.parameters.quiet {
                    println!("Time limit reached.",);
                }

                self.solution.is_time_limit_reached = true;
                self.is_terminated = true;

                return (self.stop_timer_and_return_solution(), true);
            }

            if solution.is_expansion_limit_reached {
                if !self.parameters.quiet {
                    println!("Expansion limit reached.",);
                }

                self.solution.is_expansion_limit_reached = true;
                self.is_terminated = true;

                return (self.stop_timer_and_return_solution(), true);
            }

            if is_max_beam_width_reached {
                if !self.parameters.quiet {
                    println!("Max beam width reached.",);
                }

                self.is_terminated = true;

                return (self.stop_timer_and_return_solution(), true);
            }

            self.beam_width *= 2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solvers::search_algorithms::beam_search;
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
    fn test_search() {
        let dp = MockDp(2);
        let root_node_constructor = |dp: &MockDp, _| {
            Some(MockNode(
                dp.get_target(),
                dp.get_identity_weight(),
                Cell::new(false),
                Vec::new(),
            ))
        };
        let node_constructor = |_: &_, state, cost, transition, parent: &MockNode, _| {
            let mut transitions = parent.3.clone();
            transitions.push(transition);
            Some(MockNode(state, cost, Cell::new(false), transitions))
        };
        let solution_checker = |dp: &MockDp, node: &MockNode| {
            dp.get_base_cost(node.get_state(dp)).map(|cost| {
                (
                    dp.combine_cost_weights(node.get_cost(dp), cost),
                    node.3.clone(),
                )
            })
        };
        let beam_search_closure = move |dp: &_, root_node, parameters: &_| {
            beam_search(
                dp,
                root_node,
                &node_constructor,
                &solution_checker,
                parameters,
            )
        };

        let parameters = SearchParameters {
            quiet: true,
            ..Default::default()
        };
        let cabs_parameters = CabsParameters::default();

        let mut search = Cabs::new(
            dp,
            root_node_constructor,
            beam_search_closure,
            parameters,
            cabs_parameters,
        );

        let solution = search.search();
        assert_eq!(solution.cost, Some(2));
        assert_eq!(solution.transitions, vec![1, 1]);
        assert_eq!(solution.best_bound, Some(2));
        assert!(solution.is_optimal);
        assert!(!solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);
    }

    #[test]
    fn test_search_infeasible() {
        let dp = MockDp(2);
        let root_node_constructor = |dp: &MockDp, _| {
            Some(MockNode(
                dp.get_target(),
                dp.get_identity_weight(),
                Cell::new(false),
                Vec::new(),
            ))
        };
        let node_constructor = |_: &_, state, cost, transition, parent: &MockNode, _| {
            let mut transitions = parent.3.clone();
            transitions.push(transition);
            Some(MockNode(state, cost, Cell::new(false), transitions))
        };
        let solution_checker = |dp: &MockDp, node: &MockNode| {
            dp.get_base_cost(node.get_state(dp)).map(|cost| {
                (
                    dp.combine_cost_weights(node.get_cost(dp), cost),
                    node.3.clone(),
                )
            })
        };
        let beam_search_closure = move |dp: &_, root_node, parameters: &_| {
            beam_search(
                dp,
                root_node,
                &node_constructor,
                &solution_checker,
                parameters,
            )
        };

        let parameters = SearchParameters {
            primal_bound: Some(2),
            quiet: true,
            ..Default::default()
        };
        let cabs_parameters = CabsParameters::default();

        let mut search = Cabs::new(
            dp,
            root_node_constructor,
            beam_search_closure,
            parameters,
            cabs_parameters,
        );

        let solution = search.search();
        assert_eq!(solution.cost, None);
        assert_eq!(solution.transitions, vec![]);
        assert_eq!(solution.best_bound, None);
        assert!(!solution.is_optimal);
        assert!(solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);
    }
}
