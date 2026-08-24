use super::search_algorithms::CabsParameters;
use crate::solvers::parallel_search_algorithms::{
    self, ArcIdTree, SendableCostNode, SendableDualBoundNode, hash_distribution,
};
use crate::solvers::search_algorithms::{self, Cabs, CostNode, DualBoundNode, SearchNode};
use crate::solvers::{Search, SearchParameters};
use crate::{BoundMut, Dominance, DpMut};
use core::panic;
use num_traits::Signed;
use std::fmt::Display;
use std::hash::Hash;
use std::sync::Arc;

/// Creates complete anytime beam search (CABS) solver without dual bound guidance.
///
/// Search nodes are ordered by the cost.
///
/// The DP model must implement the `Dominance` trait.
///
/// # Examples
///
/// ```
/// use rpid::prelude::*;
/// use rpid::solvers;
/// use fixedbitset::FixedBitSet;
///
/// struct Tsp {
///     c: Vec<Vec<i32>>,
/// }
///
/// struct TspState {
///     unvisited: FixedBitSet,
///     current: usize,
/// }
///
/// impl Dp for Tsp {
///     type State = TspState;
///     type CostType = i32;
///     type Label = usize;
///
///     fn get_target(&self) -> TspState {
///         let mut unvisited = FixedBitSet::with_capacity(self.c.len());
///         unvisited.insert_range(1..);
///
///         TspState {
///             unvisited,
///             current: 0,
///        }
///     }
///
///     fn get_successors(&self, state: &TspState) -> impl IntoIterator<Item = (TspState, i32, usize)> {
///         state.unvisited.ones().map(|next| {
///             let mut unvisited = state.unvisited.clone();
///             unvisited.remove(next);
///
///             let successor = TspState {
///                 unvisited,
///                 current: next,
///             };
///             let weight = self.c[state.current][next];
///             
///             (successor, weight, next)
///         })
///     }
///
///     fn get_base_cost(&self, state: &TspState) -> Option<i32> {
///         if state.unvisited.is_clear() {
///             Some(self.c[state.current][0])
///         } else {
///             None
///         }
///     }
/// }
///
/// impl Dominance for Tsp {
///     type State = TspState;
///     type Key = (FixedBitSet, usize);
///
///     fn get_key(&self, state: &TspState) -> Self::Key {
///         (state.unvisited.clone(), state.current)
///     }
/// }
///
/// let tsp = Tsp { c: vec![vec![0, 1, 2], vec![1, 0, 3], vec![2, 3, 0]] };
/// let parameters = SearchParameters {
///     quiet: true,
///     ..Default::default()
/// };
/// let cabs_parameters = CabsParameters::default();
/// let mut solver = solvers::create_blind_cabs(tsp, parameters, cabs_parameters);
/// let solution = solver.search();
/// assert_eq!(solution.cost, Some(6));
/// assert_eq!(solution.transitions, vec![1, 2]);
/// assert!(solution.is_optimal);
/// assert!(!solution.is_infeasible);
/// assert_eq!(solution.best_bound, Some(6));
/// ```
pub fn create_blind_cabs<'a, D, S, C, L, K>(
    dp: D,
    parameters: SearchParameters<C>,
    cabs_parameters: CabsParameters,
) -> Box<dyn Search<CostType = C, Label = L> + 'a>
where
    D: DpMut<State = S, CostType = C, Label = L> + Dominance<State = S, Key = K> + 'a,
    C: Ord + Copy + Signed + Display + 'a,
    L: Default + Copy + 'a,
    K: Hash + Eq,
{
    if !parameters.quiet {
        println!("Created Complete Anytime Beam Search (CABS) without guidance.");
    }

    let root_node_constructor = |dp: &mut D, _| {
        Some(CostNode::create_root(
            dp,
            dp.get_target(),
            dp.get_identity_weight(),
        ))
    };
    let node_constructor = {
        |dp: &mut _, state, cost, transition, parent: &CostNode<_, _, _, _>, _| {
            Some(parent.create_child(dp, state, cost, transition))
        }
    };
    let solution_checker = { |dp: &mut _, node: &CostNode<_, _, _, _>| node.check_solution(dp) };
    let beam_search_closure = {
        move |dp: &mut _, root_node, parameters: &_| {
            search_algorithms::beam_search(
                dp,
                root_node,
                node_constructor,
                solution_checker,
                parameters,
            )
        }
    };

    Box::new(Cabs::new(
        dp,
        root_node_constructor,
        beam_search_closure,
        parameters,
        cabs_parameters,
    ))
}

/// Creates complete anytime beam search (CABS) solver.
///
/// Search nodes are ordered by the f-value, which is the combination of the cost and the dual bound.
///
/// The DP model must implement the `Dominance` and `DualBound` traits.
///
/// # Panic
/// When `threads` argument takes 0 value.
///
/// # Examples
///
/// ```
/// use rpid::prelude::*;
/// use rpid::solvers;
/// use fixedbitset::FixedBitSet;
///
/// #[derive(Clone)]
/// struct Tsp {
///     c: Vec<Vec<i32>>,
/// }
///
/// #[derive(Clone, Hash)]
/// struct TspState {
///     unvisited: FixedBitSet,
///     current: usize,
/// }
///
/// impl Dp for Tsp {
///     type State = TspState;
///     type CostType = i32;
///     type Label = usize;
///
///     fn get_target(&self) -> Self::State {
///         let mut unvisited = FixedBitSet::with_capacity(self.c.len());
///         unvisited.insert_range(1..);
///
///         TspState {
///             unvisited,
///             current: 0,
///        }
///     }
///
///     fn get_successors(
///         &self,
///         state: &Self::State,
///     ) -> impl IntoIterator<Item = (Self::State, Self::CostType, Self::Label)> {
///         state.unvisited.ones().map(|next| {
///             let mut unvisited = state.unvisited.clone();
///             unvisited.remove(next);
///
///             let successor = TspState {
///                 unvisited,
///                 current: next,
///             };
///             let weight = self.c[state.current][next];
///             
///             (successor, weight, next)
///         })
///     }
///
///     fn get_base_cost(&self, state: &Self::State) -> Option<Self::CostType> {
///         if state.unvisited.is_clear() {
///             Some(self.c[state.current][0])
///         } else {
///             None
///         }
///     }
/// }
///
/// impl Dominance for Tsp {
///     type State = TspState;
///     type Key = (FixedBitSet, usize);
///
///     fn get_key(&self, state: &Self::State) -> Self::Key {
///         (state.unvisited.clone(), state.current)
///     }
/// }
///
/// impl Bound for Tsp {
///     type State = TspState;
///     type CostType = i32;
///
///     fn get_dual_bound(&self, state: &Self::State) -> Option<Self::CostType> {
///         Some(0)
///     }
/// }
///
/// let tsp = Tsp { c: vec![vec![0, 1, 2], vec![1, 0, 3], vec![2, 3, 0]] };
/// let parameters = SearchParameters {
///     quiet: true,
///     ..Default::default()
/// };
/// let cabs_parameters = CabsParameters::default();
/// let mut solver = solvers::create_cabs(tsp, parameters, cabs_parameters);
/// let solution = solver.search();
/// assert_eq!(solution.cost, Some(6));
/// assert_eq!(solution.transitions, vec![1, 2]);
/// assert!(solution.is_optimal);
/// assert!(!solution.is_infeasible);
/// assert_eq!(solution.best_bound, Some(6));
/// ```
pub fn create_cabs<'a, D, S, C, L, K>(
    dp: D,
    mut parameters: SearchParameters<C>,
    cabs_parameters: CabsParameters,
) -> Box<dyn Search<CostType = C, Label = L> + 'a>
where
    D: DpMut<State = S, CostType = C, Label = L>
        + Dominance<State = S, Key = K>
        + BoundMut<State = S, CostType = C>
        + 'a,
    C: Ord + Copy + Signed + Display + 'a,
    L: Default + Copy + 'a,
    K: Hash + Eq,
{
    if !parameters.quiet {
        println!("Created Complete Anytime Beam Search (CABS).");
    }

    let root_node_constructor = |dp: &mut D, bound| {
        DualBoundNode::create_root(dp, dp.get_target(), dp.get_identity_weight(), bound)
    };
    let node_constructor = {
        |dp: &mut D,
         state,
         cost,
         transition,
         parent: &DualBoundNode<_, _, _, _, ArcIdTree<L>, Arc<_>>,
         primal_bound| {
            parent.create_child(dp, state, cost, transition, primal_bound, None)
        }
    };
    let solution_checker =
        { |dp: &mut D, node: &DualBoundNode<_, _, _, _, _, _>| node.check_solution(dp) };

    let beam_search_closure = {
        move |dp: &mut _, root_node, parameters: &_| {
            search_algorithms::beam_search(
                dp,
                root_node,
                node_constructor,
                solution_checker,
                parameters,
            )
        }
    };
    parameters.update_bounds(&dp);

    Box::new(Cabs::new(
        dp,
        root_node_constructor,
        beam_search_closure,
        parameters,
        cabs_parameters,
    ))
}

/// Creates complete anytime hash distributed beam search 2 (CAHDBS2) solver.
/// This is the default parallelization version of CABS, since it was experimented with the best performance in previous research.
///
/// Search nodes are ordered by the f-value, which is the combination of the cost and the dual bound.
///
/// The DP model must implement the `Dominance` and `DualBound` traits.
///
/// # References
///
/// Ryo Kuroiwa and J. Christopher Beck. "Parallel Beam Search Algorithms for Domain-Independent Dynamic Programming,"
/// Proceedings of the 38th Annual AAAI Conference on Artificial Intelligence (AAAI), 2024.
///
/// # Panic
/// When `threads` argument takes 0 value.
///
/// # Examples
///
/// ```
/// use rpid::prelude::*;
/// use rpid::solvers;
/// use fixedbitset::FixedBitSet;
///
/// #[derive(Clone)]
/// struct Tsp {
///     c: Vec<Vec<i32>>,
/// }
///
/// #[derive(Clone, Hash)]
/// struct TspState {
///     unvisited: FixedBitSet,
///     current: usize,
/// }
///
/// impl Dp for Tsp {
///     type State = TspState;
///     type CostType = i32;
///     type Label = usize;
///
///     fn get_target(&self) -> Self::State {
///         let mut unvisited = FixedBitSet::with_capacity(self.c.len());
///         unvisited.insert_range(1..);
///
///         TspState {
///             unvisited,
///             current: 0,
///        }
///     }
///
///     fn get_successors(
///         &self,
///         state: &Self::State,
///     ) -> impl IntoIterator<Item = (Self::State, Self::CostType, Self::Label)> {
///         state.unvisited.ones().map(|next| {
///             let mut unvisited = state.unvisited.clone();
///             unvisited.remove(next);
///
///             let successor = TspState {
///                 unvisited,
///                 current: next,
///             };
///             let weight = self.c[state.current][next];
///             
///             (successor, weight, next)
///         })
///     }
///
///     fn get_base_cost(&self, state: &Self::State) -> Option<Self::CostType> {
///         if state.unvisited.is_clear() {
///             Some(self.c[state.current][0])
///         } else {
///             None
///         }
///     }
/// }
///
/// impl Dominance for Tsp {
///     type State = TspState;
///     type Key = (FixedBitSet, usize);
///
///     fn get_key(&self, state: &Self::State) -> Self::Key {
///         (state.unvisited.clone(), state.current)
///     }
/// }
///
/// impl Bound for Tsp {
///     type State = TspState;
///     type CostType = i32;
///
///     fn get_dual_bound(&self, state: &Self::State) -> Option<Self::CostType> {
///         Some(0)
///     }
/// }
///
/// let tsp = Tsp { c: vec![vec![0, 1, 2], vec![1, 0, 3], vec![2, 3, 0]] };
/// let parameters = SearchParameters {
///     quiet: true,
///     ..Default::default()
/// };
///
/// let cabs_parameters = CabsParameters::default();
/// let mut solver = solvers::create_parallel_cabs(tsp, parameters, cabs_parameters, 8);
/// let solution = solver.search();
/// assert_eq!(solution.cost, Some(6));
/// assert_eq!(solution.transitions, vec![1, 2]);
/// assert!(solution.is_optimal);
/// assert!(!solution.is_infeasible);
/// assert_eq!(solution.best_bound, Some(6));
/// ```
pub fn create_parallel_cabs<'a, D, S, C, L, K>(
    dp: D,
    mut parameters: SearchParameters<C>,
    cabs_parameters: CabsParameters,
    threads: usize,
) -> Box<dyn Search<CostType = C, Label = L> + 'a>
where
    D: DpMut<State = S, CostType = C, Label = L>
        + Dominance<State = S, Key = K>
        + BoundMut<State = S, CostType = C>
        + Clone
        + Send
        + 'a,
    S: Clone + Send,
    C: Ord + Copy + Signed + Display + Send + Sync + 'a,
    L: Default + Copy + Send + Sync + 'a,
    K: Hash + Eq,
{
    if !parameters.quiet {
        println!(
            "Created Complete Anytime Hash Distributed Beam Search 2 (CAHDBS2) with {} threads.",
            threads
        );
    }

    const THREAD_ASSIGNER_SEED: u32 = 0x5583c24d;

    let root_node_constructor = |dp: &mut D, bound| {
        DualBoundNode::create_root(dp, dp.get_target(), dp.get_identity_weight(), bound)
    };
    let node_constructor = {
        |dp: &mut D,
         state,
         cost,
         transition,
         parent: &DualBoundNode<_, _, _, _, ArcIdTree<L>, Arc<_>>,
         primal_bound| {
            parent.create_child(dp, state, cost, transition, primal_bound, None)
        }
    };
    let solution_checker =
        { |dp: &mut D, node: &DualBoundNode<_, _, _, _, _, _>| node.check_solution(dp) };
    let thread_assigner =
        move |dp: &D, message: &DualBoundNode<_, _, _, _, _, _>, threads: usize| {
            hash_distribution::fx_hash_assign_thread(
                &dp.get_key(message.get_state(dp)),
                threads,
                THREAD_ASSIGNER_SEED,
            )
        };

    let print_statistics = !parameters.quiet;

    if threads == 0 {
        panic!("Creating multi-thread CABS solver with 0 thread");
    }

    let beam_search_closure = move |dp: &mut _, root_node, parameters: &_| {
        let (solution, statistics) = parallel_search_algorithms::hd_beam_search2(
            dp,
            root_node,
            node_constructor,
            solution_checker,
            thread_assigner,
            parameters,
            threads,
        )
        .unwrap();

        if print_statistics {
            println!(
                "HD2 Beam Search searched with beam size: {}, threads: {}, kept: {}, sent: {}",
                parameters.beam_width,
                threads,
                statistics.kept.iter().sum::<usize>(),
                statistics.sent.iter().sum::<usize>(),
            );
        }

        solution
    };

    parameters.update_bounds(&dp);

    Box::new(Cabs::new(
        dp,
        root_node_constructor,
        beam_search_closure,
        parameters,
        cabs_parameters,
    ))
}

/// Creates complete anytime hash distributed beam search 2 (CAHDBS2) solver without guidance.
/// This is the default parallelization version of CABS without guidance.
///
/// Search nodes are ordered by the cost.
///
/// # References
///
/// Ryo Kuroiwa and J. Christopher Beck. "Parallel Beam Search Algorithms for Domain-Independent Dynamic Programming,"
/// Proceedings of the 38th Annual AAAI Conference on Artificial Intelligence (AAAI), 2024.
///
/// # Panic
/// When `threads` argument takes 0 value.
///
/// # Examples
///
/// ```
/// use rpid::prelude::*;
/// use rpid::solvers;
/// use fixedbitset::FixedBitSet;
///
/// #[derive(Clone)]
/// struct Tsp {
///     c: Vec<Vec<i32>>,
/// }
///
/// #[derive(Clone, Hash)]
/// struct TspState {
///     unvisited: FixedBitSet,
///     current: usize,
/// }
///
/// impl Dp for Tsp {
///     type State = TspState;
///     type CostType = i32;
///     type Label = usize;
///
///     fn get_target(&self) -> Self::State {
///         let mut unvisited = FixedBitSet::with_capacity(self.c.len());
///         unvisited.insert_range(1..);
///
///         TspState {
///             unvisited,
///             current: 0,
///        }
///     }
///
///     fn get_successors(
///         &self,
///         state: &Self::State,
///     ) -> impl IntoIterator<Item = (Self::State, Self::CostType, Self::Label)> {
///         state.unvisited.ones().map(|next| {
///             let mut unvisited = state.unvisited.clone();
///             unvisited.remove(next);
///
///             let successor = TspState {
///                 unvisited,
///                 current: next,
///             };
///             let weight = self.c[state.current][next];
///             
///             (successor, weight, next)
///         })
///     }
///
///     fn get_base_cost(&self, state: &Self::State) -> Option<Self::CostType> {
///         if state.unvisited.is_clear() {
///             Some(self.c[state.current][0])
///         } else {
///             None
///         }
///     }
/// }
///
/// impl Dominance for Tsp {
///     type State = TspState;
///     type Key = (FixedBitSet, usize);
///
///     fn get_key(&self, state: &Self::State) -> Self::Key {
///         (state.unvisited.clone(), state.current)
///     }
/// }
///
/// let tsp = Tsp { c: vec![vec![0, 1, 2], vec![1, 0, 3], vec![2, 3, 0]] };
/// let parameters = SearchParameters {
///     quiet: true,
///     ..Default::default()
/// };
///
/// let cabs_parameters = CabsParameters::default();
/// let mut solver = solvers::create_blind_parallel_cabs(tsp, parameters, cabs_parameters, 8);
/// let solution = solver.search();
/// assert_eq!(solution.cost, Some(6));
/// assert_eq!(solution.transitions, vec![1, 2]);
/// assert!(solution.is_optimal);
/// assert!(!solution.is_infeasible);
/// assert_eq!(solution.best_bound, Some(6));
/// ```
pub fn create_blind_parallel_cabs<'a, D, S, C, L, K>(
    dp: D,
    parameters: SearchParameters<C>,
    cabs_parameters: CabsParameters,
    threads: usize,
) -> Box<dyn Search<CostType = C, Label = L> + 'a>
where
    D: DpMut<State = S, CostType = C, Label = L>
        + Dominance<State = S, Key = K>
        + Clone
        + Send
        + 'a,
    S: Clone + Send,
    C: Ord + Copy + Signed + Display + Send + Sync + 'a,
    L: Default + Copy + Send + Sync + 'a,
    K: Hash + Eq,
{
    if !parameters.quiet {
        println!(
            "Created Complete Anytime Hash Distributed Beam Search 2 (CAHDBS2) with {} threads without guidance.",
            threads
        );
    }

    const THREAD_ASSIGNER_SEED: u32 = 0x5583c24d;

    let root_node_constructor = |dp: &mut D, _| {
        Some(CostNode::create_root(
            dp,
            dp.get_target(),
            dp.get_identity_weight(),
        ))
    };
    let node_constructor = {
        |dp: &mut D,
         state,
         cost,
         transition,
         parent: &CostNode<_, _, _, _, ArcIdTree<L>, Arc<_>>,
         _| { Some(parent.create_child(dp, state, cost, transition)) }
    };
    let solution_checker =
        { |dp: &mut D, node: &CostNode<_, _, _, _, _, _>| node.check_solution(dp) };
    let thread_assigner = move |dp: &D, message: &CostNode<_, _, _, _, _, _>, threads: usize| {
        hash_distribution::fx_hash_assign_thread(
            &dp.get_key(message.get_state(dp)),
            threads,
            THREAD_ASSIGNER_SEED,
        )
    };

    let print_statistics = !parameters.quiet;

    if threads == 0 {
        panic!("Creating multi-thread CABS solver with 0 thread");
    }

    let beam_search_closure = move |dp: &mut _, root_node, parameters: &_| {
        let (solution, statistics) = parallel_search_algorithms::hd_beam_search2(
            dp,
            root_node,
            node_constructor,
            solution_checker,
            thread_assigner,
            parameters,
            threads,
        )
        .unwrap();

        if print_statistics {
            println!(
                "HD2 Beam Search searched with beam size: {}, threads: {}, kept: {}, sent: {}",
                parameters.beam_width,
                threads,
                statistics.kept.iter().sum::<usize>(),
                statistics.sent.iter().sum::<usize>(),
            );
        }

        solution
    };

    Box::new(Cabs::new(
        dp,
        root_node_constructor,
        beam_search_closure,
        parameters,
        cabs_parameters,
    ))
}

/// Creates complete anytime hash distributed beam search 1 (CAHDBS1) solver.
///
/// Search nodes are ordered by the f-value, which is the combination of the cost and the dual bound.
///
/// The DP model must implement the `Dominance` and `DualBound` traits.
///
/// # References
///
/// Ryo Kuroiwa and J. Christopher Beck. "Parallel Beam Search Algorithms for Domain-Independent Dynamic Programming,"
/// Proceedings of the 38th Annual AAAI Conference on Artificial Intelligence (AAAI), 2024.
///
/// # Panic
/// When `threads` argument takes 0 value.
///
/// # Examples
///
/// ```
/// use rpid::prelude::*;
/// use rpid::solvers;
/// use fixedbitset::FixedBitSet;
///
/// #[derive(Clone)]
/// struct Tsp {
///     c: Vec<Vec<i32>>,
/// }
///
/// #[derive(Clone, Hash)]
/// struct TspState {
///     unvisited: FixedBitSet,
///     current: usize,
/// }
///
/// impl Dp for Tsp {
///     type State = TspState;
///     type CostType = i32;
///     type Label = usize;
///
///     fn get_target(&self) -> Self::State {
///         let mut unvisited = FixedBitSet::with_capacity(self.c.len());
///         unvisited.insert_range(1..);
///
///         TspState {
///             unvisited,
///             current: 0,
///        }
///     }
///
///     fn get_successors(
///         &self,
///         state: &Self::State,
///     ) -> impl IntoIterator<Item = (Self::State, Self::CostType, Self::Label)> {
///         state.unvisited.ones().map(|next| {
///             let mut unvisited = state.unvisited.clone();
///             unvisited.remove(next);
///
///             let successor = TspState {
///                 unvisited,
///                 current: next,
///             };
///             let weight = self.c[state.current][next];
///             
///             (successor, weight, next)
///         })
///     }
///
///     fn get_base_cost(&self, state: &Self::State) -> Option<Self::CostType> {
///         if state.unvisited.is_clear() {
///             Some(self.c[state.current][0])
///         } else {
///             None
///         }
///     }
/// }
///
/// impl Dominance for Tsp {
///     type State = TspState;
///     type Key = (FixedBitSet, usize);
///
///     fn get_key(&self, state: &Self::State) -> Self::Key {
///         (state.unvisited.clone(), state.current)
///     }
/// }
///
/// impl Bound for Tsp {
///     type State = TspState;
///     type CostType = i32;
///
///     fn get_dual_bound(&self, state: &Self::State) -> Option<Self::CostType> {
///         Some(0)
///     }
/// }
///
/// let tsp = Tsp { c: vec![vec![0, 1, 2], vec![1, 0, 3], vec![2, 3, 0]] };
/// let parameters = SearchParameters {
///     quiet: true,
///     ..Default::default()
/// };
///
/// let cabs_parameters = CabsParameters::default();
/// let mut solver = solvers::create_cahdbs1(tsp, parameters, cabs_parameters, 8);
/// let solution = solver.search();
/// assert_eq!(solution.cost, Some(6));
/// assert_eq!(solution.transitions, vec![1, 2]);
/// assert!(solution.is_optimal);
/// assert!(!solution.is_infeasible);
/// assert_eq!(solution.best_bound, Some(6));
/// ```
pub fn create_cahdbs1<'a, D, S, C, L, K>(
    dp: D,
    mut parameters: SearchParameters<C>,
    cabs_parameters: CabsParameters,
    threads: usize,
) -> Box<dyn Search<CostType = C, Label = L> + 'a>
where
    D: DpMut<State = S, CostType = C, Label = L>
        + Dominance<State = S, Key = K>
        + BoundMut<State = S, CostType = C>
        + Clone
        + Send
        + 'a,
    S: Clone + Send,
    C: Ord + Copy + Signed + Display + Send + Sync + 'a,
    L: Default + Copy + Send + Sync + 'a,
    K: Hash + Eq,
{
    if !parameters.quiet {
        println!(
            "Created Complete Anytime Hash Distributed Beam Search 1 (CAHDBS1) with {} threads.",
            threads
        );
    }

    const THREAD_ASSIGNER_SEED: u32 = 0x5583c24d;

    let root_node_constructor = |dp: &mut D, bound| {
        DualBoundNode::create_root(dp, dp.get_target(), dp.get_identity_weight(), bound)
    };
    let node_constructor = {
        |dp: &mut D,
         state,
         cost,
         transition,
         parent: &DualBoundNode<_, _, _, _, ArcIdTree<L>, Arc<_>>,
         primal_bound| {
            parent.create_child(dp, state, cost, transition, primal_bound, None)
        }
    };
    let solution_checker =
        { |dp: &mut D, node: &DualBoundNode<_, _, _, _, _, _>| node.check_solution(dp) };
    let thread_assigner =
        move |dp: &D, message: &DualBoundNode<_, _, _, _, _, _>, threads: usize| {
            hash_distribution::fx_hash_assign_thread(
                &dp.get_key(message.get_state(dp)),
                threads,
                THREAD_ASSIGNER_SEED,
            )
        };

    let print_statistics = !parameters.quiet;

    if threads == 0 {
        panic!("Creating multi-thread CABS solver with 0 thread");
    }
    let beam_search_closure = move |dp: &mut _, root_node, parameters: &_| {
        let (solution, statistics) = parallel_search_algorithms::hd_beam_search1(
            dp,
            root_node,
            node_constructor,
            solution_checker,
            thread_assigner,
            parameters,
            threads,
        )
        .unwrap();

        if print_statistics {
            println!(
                "HD1 Beam Search searched with beam size: {}, threads: {}, kept: {}, sent: {}",
                parameters.beam_width,
                threads,
                statistics.kept.iter().sum::<usize>(),
                statistics.sent.iter().sum::<usize>(),
            );
        }

        solution
    };

    parameters.update_bounds(&dp);

    Box::new(Cabs::new(
        dp,
        root_node_constructor,
        beam_search_closure,
        parameters,
        cabs_parameters,
    ))
}

/// Creates complete anytime hash distributed beam search 1 (CAHDBS2) solver without guidance.
///
/// Search nodes are ordered by the cost.
///
/// # References
///
/// Ryo Kuroiwa and J. Christopher Beck. "Parallel Beam Search Algorithms for Domain-Independent Dynamic Programming,"
/// Proceedings of the 38th Annual AAAI Conference on Artificial Intelligence (AAAI), 2024.
///
/// # Panic
/// When `threads` argument takes 0 value.
///
/// # Examples
///
/// ```
/// use rpid::prelude::*;
/// use rpid::solvers;
/// use fixedbitset::FixedBitSet;
///
/// #[derive(Clone)]
/// struct Tsp {
///     c: Vec<Vec<i32>>,
/// }
///
/// #[derive(Clone, Hash)]
/// struct TspState {
///     unvisited: FixedBitSet,
///     current: usize,
/// }
///
/// impl Dp for Tsp {
///     type State = TspState;
///     type CostType = i32;
///     type Label = usize;
///
///     fn get_target(&self) -> Self::State {
///         let mut unvisited = FixedBitSet::with_capacity(self.c.len());
///         unvisited.insert_range(1..);
///
///         TspState {
///             unvisited,
///             current: 0,
///        }
///     }
///
///     fn get_successors(
///         &self,
///         state: &Self::State,
///     ) -> impl IntoIterator<Item = (Self::State, Self::CostType, Self::Label)> {
///         state.unvisited.ones().map(|next| {
///             let mut unvisited = state.unvisited.clone();
///             unvisited.remove(next);
///
///             let successor = TspState {
///                 unvisited,
///                 current: next,
///             };
///             let weight = self.c[state.current][next];
///             
///             (successor, weight, next)
///         })
///     }
///
///     fn get_base_cost(&self, state: &Self::State) -> Option<Self::CostType> {
///         if state.unvisited.is_clear() {
///             Some(self.c[state.current][0])
///         } else {
///             None
///         }
///     }
/// }
///
/// impl Dominance for Tsp {
///     type State = TspState;
///     type Key = (FixedBitSet, usize);
///
///     fn get_key(&self, state: &Self::State) -> Self::Key {
///         (state.unvisited.clone(), state.current)
///     }
/// }
///
/// let tsp = Tsp { c: vec![vec![0, 1, 2], vec![1, 0, 3], vec![2, 3, 0]] };
/// let parameters = SearchParameters {
///     quiet: true,
///     ..Default::default()
/// };
///
/// let cabs_parameters = CabsParameters::default();
/// let mut solver = solvers::create_blind_cahdbs1(tsp, parameters, cabs_parameters, 8);
/// let solution = solver.search();
/// assert_eq!(solution.cost, Some(6));
/// assert_eq!(solution.transitions, vec![1, 2]);
/// assert!(solution.is_optimal);
/// assert!(!solution.is_infeasible);
/// assert_eq!(solution.best_bound, Some(6));
/// ```
pub fn create_blind_cahdbs1<'a, D, S, C, L, K>(
    dp: D,
    parameters: SearchParameters<C>,
    cabs_parameters: CabsParameters,
    threads: usize,
) -> Box<dyn Search<CostType = C, Label = L> + 'a>
where
    D: DpMut<State = S, CostType = C, Label = L>
        + Dominance<State = S, Key = K>
        + Clone
        + Send
        + 'a,
    S: Clone + Send,
    C: Ord + Copy + Signed + Display + Send + Sync + 'a,
    L: Default + Copy + Send + Sync + 'a,
    K: Hash + Eq,
{
    if !parameters.quiet {
        println!(
            "Created Complete Anytime Hash Distributed Beam Search 1 (CAHDBS1) with {} threads without guidance.",
            threads
        );
    }

    const THREAD_ASSIGNER_SEED: u32 = 0x5583c24d;

    let root_node_constructor = |dp: &mut D, _| {
        Some(CostNode::create_root(
            dp,
            dp.get_target(),
            dp.get_identity_weight(),
        ))
    };
    let node_constructor = {
        |dp: &mut D,
         state,
         cost,
         transition,
         parent: &CostNode<_, _, _, _, ArcIdTree<L>, Arc<_>>,
         _| { Some(parent.create_child(dp, state, cost, transition)) }
    };
    let solution_checker =
        { |dp: &mut D, node: &CostNode<_, _, _, _, _, _>| node.check_solution(dp) };
    let thread_assigner = move |dp: &D, message: &CostNode<_, _, _, _, _, _>, threads: usize| {
        hash_distribution::fx_hash_assign_thread(
            &dp.get_key(message.get_state(dp)),
            threads,
            THREAD_ASSIGNER_SEED,
        )
    };

    let print_statistics = !parameters.quiet;

    if threads == 0 {
        panic!("Creating multi-thread CABS solver with 0 thread");
    }

    let beam_search_closure = move |dp: &mut _, root_node, parameters: &_| {
        let (solution, statistics) = parallel_search_algorithms::hd_beam_search1(
            dp,
            root_node,
            node_constructor,
            solution_checker,
            thread_assigner,
            parameters,
            threads,
        )
        .unwrap();

        if print_statistics {
            println!(
                "HD2 Beam Search searched with beam size: {}, threads: {}, kept: {}, sent: {}",
                parameters.beam_width,
                threads,
                statistics.kept.iter().sum::<usize>(),
                statistics.sent.iter().sum::<usize>(),
            );
        }

        solution
    };

    Box::new(Cabs::new(
        dp,
        root_node_constructor,
        beam_search_closure,
        parameters,
        cabs_parameters,
    ))
}

/// Creates complete anytime shared-beam search (CASBS) solver.
///
/// Search nodes are ordered by the f-value, which is the combination of the cost and the dual bound.
///
/// The DP model must implement the `Dominance` and `DualBound` traits.
///
/// # References
///
/// Ryo Kuroiwa and J. Christopher Beck. "Parallel Beam Search Algorithms for Domain-Independent Dynamic Programming,"
/// Proceedings of the 38th Annual AAAI Conference on Artificial Intelligence (AAAI), 2024.
///
/// # Panic
/// When `threads` argument takes 0 value.
///
/// # Examples
///
/// ```
/// use rpid::prelude::*;
/// use rpid::solvers;
/// use fixedbitset::FixedBitSet;
///
/// #[derive(Clone)]
/// struct Tsp {
///     c: Vec<Vec<i32>>,
/// }
///
/// #[derive(Clone, Hash)]
/// struct TspState {
///     unvisited: FixedBitSet,
///     current: usize,
/// }
///
/// impl Dp for Tsp {
///     type State = TspState;
///     type CostType = i32;
///     type Label = usize;
///
///     fn get_target(&self) -> Self::State {
///         let mut unvisited = FixedBitSet::with_capacity(self.c.len());
///         unvisited.insert_range(1..);
///
///         TspState {
///             unvisited,
///             current: 0,
///        }
///     }
///
///     fn get_successors(
///         &self,
///         state: &Self::State,
///     ) -> impl IntoIterator<Item = (Self::State, Self::CostType, Self::Label)> {
///         state.unvisited.ones().map(|next| {
///             let mut unvisited = state.unvisited.clone();
///             unvisited.remove(next);
///
///             let successor = TspState {
///                 unvisited,
///                 current: next,
///             };
///             let weight = self.c[state.current][next];
///             
///             (successor, weight, next)
///         })
///     }
///
///     fn get_base_cost(&self, state: &Self::State) -> Option<Self::CostType> {
///         if state.unvisited.is_clear() {
///             Some(self.c[state.current][0])
///         } else {
///             None
///         }
///     }
/// }
///
/// impl Dominance for Tsp {
///     type State = TspState;
///     type Key = (FixedBitSet, usize);
///
///     fn get_key(&self, state: &Self::State) -> Self::Key {
///         (state.unvisited.clone(), state.current)
///     }
/// }
///
/// impl Bound for Tsp {
///     type State = TspState;
///     type CostType = i32;
///
///     fn get_dual_bound(&self, state: &Self::State) -> Option<Self::CostType> {
///         Some(0)
///     }
/// }
///
/// let tsp = Tsp { c: vec![vec![0, 1, 2], vec![1, 0, 3], vec![2, 3, 0]] };
/// let parameters = SearchParameters {
///     quiet: true,
///     ..Default::default()
/// };
///
/// let cabs_parameters = CabsParameters::default();
/// let mut solver = solvers::create_casbs(tsp, parameters, cabs_parameters, 8);
/// let solution = solver.search();
/// assert_eq!(solution.cost, Some(6));
/// assert_eq!(solution.transitions, vec![1, 2]);
/// assert!(solution.is_optimal);
/// assert!(!solution.is_infeasible);
/// assert_eq!(solution.best_bound, Some(6));
/// ```
pub fn create_casbs<'a, D, S, C, L, K>(
    dp: D,
    mut parameters: SearchParameters<C>,
    cabs_parameters: CabsParameters,
    threads: usize,
) -> Box<dyn Search<CostType = C, Label = L> + 'a>
where
    D: DpMut<State = S, CostType = C, Label = L>
        + Dominance<State = S, Key = K>
        + BoundMut<State = S, CostType = C>
        + Clone
        + Send
        + Sync
        + 'a,
    S: Clone + Send + Sync,
    C: Ord + Copy + Signed + Display + Send + Sync + 'a,
    L: Default + Copy + Send + Sync + 'a,
    K: Hash + Eq + Send + Sync,
{
    if !parameters.quiet {
        println!(
            "Created Complete Anytime Shared-Beam Search (CASBS) with {} threads.",
            threads
        );
    }

    let root_node_constructor = |dp: &mut D, bound| {
        SendableDualBoundNode::create_root(dp, dp.get_target(), dp.get_identity_weight(), bound)
    };
    let node_constructor = {
        |dp: &mut D,
         state,
         cost,
         transition,
         parent: &SendableDualBoundNode<_, _, _, _>,
         primal_bound| {
            parent.create_child(dp, state, cost, transition, primal_bound, None)
        }
    };
    let solution_checker =
        { |dp: &mut D, node: &SendableDualBoundNode<_, _, _, _, _, _>| node.check_solution(dp) };

    if threads == 0 {
        panic!("Creating multi-thread CABS solver with 0 thread");
    }
    let beam_search_closure = move |dp: &mut _, root_node, parameters: &_| {
        parallel_search_algorithms::shared_beam_search(
            dp,
            root_node,
            node_constructor,
            solution_checker,
            parameters,
            threads,
        )
        .unwrap()
    };

    parameters.update_bounds(&dp);

    Box::new(Cabs::new(
        dp,
        root_node_constructor,
        beam_search_closure,
        parameters,
        cabs_parameters,
    ))
}

/// Creates complete anytime shared-beam search (CASBS) solver without guidance.
///
/// Search nodes are ordered by the cost.
///
/// The DP model must implement the `Dominance` traits.
///
/// # References
///
/// Ryo Kuroiwa and J. Christopher Beck. "Parallel Beam Search Algorithms for Domain-Independent Dynamic Programming,"
/// Proceedings of the 38th Annual AAAI Conference on Artificial Intelligence (AAAI), 2024.
///
/// # Panic
/// When `threads` argument takes 0 value.
///
/// # Examples
///
/// ```
/// use rpid::prelude::*;
/// use rpid::solvers;
/// use fixedbitset::FixedBitSet;
///
/// #[derive(Clone)]
/// struct Tsp {
///     c: Vec<Vec<i32>>,
/// }
///
/// #[derive(Clone, Hash)]
/// struct TspState {
///     unvisited: FixedBitSet,
///     current: usize,
/// }
///
/// impl Dp for Tsp {
///     type State = TspState;
///     type CostType = i32;
///     type Label = usize;
///
///     fn get_target(&self) -> Self::State {
///         let mut unvisited = FixedBitSet::with_capacity(self.c.len());
///         unvisited.insert_range(1..);
///
///         TspState {
///             unvisited,
///             current: 0,
///        }
///     }
///
///     fn get_successors(
///         &self,
///         state: &Self::State,
///     ) -> impl IntoIterator<Item = (Self::State, Self::CostType, Self::Label)> {
///         state.unvisited.ones().map(|next| {
///             let mut unvisited = state.unvisited.clone();
///             unvisited.remove(next);
///
///             let successor = TspState {
///                 unvisited,
///                 current: next,
///             };
///             let weight = self.c[state.current][next];
///             
///             (successor, weight, next)
///         })
///     }
///
///     fn get_base_cost(&self, state: &Self::State) -> Option<Self::CostType> {
///         if state.unvisited.is_clear() {
///             Some(self.c[state.current][0])
///         } else {
///             None
///         }
///     }
/// }
///
/// impl Dominance for Tsp {
///     type State = TspState;
///     type Key = (FixedBitSet, usize);
///
///     fn get_key(&self, state: &Self::State) -> Self::Key {
///         (state.unvisited.clone(), state.current)
///     }
/// }
///
/// let tsp = Tsp { c: vec![vec![0, 1, 2], vec![1, 0, 3], vec![2, 3, 0]] };
/// let parameters = SearchParameters {
///     quiet: true,
///     ..Default::default()
/// };
///
/// let cabs_parameters = CabsParameters::default();
/// let mut solver = solvers::create_blind_casbs(tsp, parameters, cabs_parameters, 8);
/// let solution = solver.search();
/// assert_eq!(solution.cost, Some(6));
/// assert_eq!(solution.transitions, vec![1, 2]);
/// assert!(solution.is_optimal);
/// assert!(!solution.is_infeasible);
/// assert_eq!(solution.best_bound, Some(6));
/// ```
pub fn create_blind_casbs<'a, D, S, C, L, K>(
    dp: D,
    parameters: SearchParameters<C>,
    cabs_parameters: CabsParameters,
    threads: usize,
) -> Box<dyn Search<CostType = C, Label = L> + 'a>
where
    D: DpMut<State = S, CostType = C, Label = L>
        + Dominance<State = S, Key = K>
        + Clone
        + Send
        + Sync
        + 'a,
    S: Clone + Send + Sync,
    C: Ord + Copy + Signed + Display + Send + Sync + 'a,
    L: Default + Copy + Send + Sync + 'a,
    K: Hash + Eq + Send + Sync,
{
    if !parameters.quiet {
        println!(
            "Created Complete Anytime Shared-Beam Search (CASBS) with {} threads without guidance.",
            threads
        );
    }

    let root_node_constructor = |dp: &mut D, _| {
        Some(SendableCostNode::create_root(
            dp,
            dp.get_target(),
            dp.get_identity_weight(),
        ))
    };
    let node_constructor = {
        |dp: &mut D, state, cost, transition, parent: &SendableCostNode<_, _, _, _>, _| {
            Some(parent.create_child(dp, state, cost, transition))
        }
    };
    let solution_checker =
        { |dp: &mut D, node: &SendableCostNode<_, _, _, _, _, _>| node.check_solution(dp) };

    if threads == 0 {
        panic!("Creating multi-thread CABS solver with 0 thread");
    }
    let beam_search_closure = move |dp: &mut _, root_node, parameters: &_| {
        parallel_search_algorithms::shared_beam_search(
            dp,
            root_node,
            node_constructor,
            solution_checker,
            parameters,
            threads,
        )
        .unwrap()
    };

    Box::new(Cabs::new(
        dp,
        root_node_constructor,
        beam_search_closure,
        parameters,
        cabs_parameters,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dp::{Bound, Dp};

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

    impl Bound for MockDp {
        type State = i32;
        type CostType = i32;

        fn get_dual_bound(&self, _: &Self::State) -> Option<Self::CostType> {
            Some(0)
        }
    }

    #[test]
    fn test_cabs() {
        let dp = MockDp(2);
        let parameters = SearchParameters {
            quiet: true,
            ..Default::default()
        };
        let cabs_parameters = CabsParameters::default();
        let mut search = create_cabs(dp, parameters, cabs_parameters);

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
    fn test_cabs_infeasible() {
        let dp = MockDp(2);
        let parameters = SearchParameters {
            primal_bound: Some(2),
            quiet: true,
            ..Default::default()
        };
        let cabs_parameters = CabsParameters::default();
        let mut search = create_cabs(dp, parameters, cabs_parameters);

        let solution = search.search();
        assert_eq!(solution.cost, None);
        assert_eq!(solution.transitions, vec![]);
        assert_eq!(solution.best_bound, None);
        assert!(!solution.is_optimal);
        assert!(solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);
    }

    #[test]
    fn test_blind_cabs() {
        let dp = MockDp(2);
        let parameters = SearchParameters {
            quiet: true,
            ..Default::default()
        };
        let cabs_parameters = CabsParameters::default();
        let mut search = create_blind_cabs(dp, parameters, cabs_parameters);

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
    fn test_blind_cabs_infeasible() {
        let dp = MockDp(2);
        let parameters = SearchParameters {
            primal_bound: Some(2),
            quiet: true,
            ..Default::default()
        };
        let cabs_parameters = CabsParameters::default();
        let mut search = create_blind_cabs(dp, parameters, cabs_parameters);

        let solution = search.search();
        assert_eq!(solution.cost, None);
        assert_eq!(solution.transitions, vec![]);
        assert_eq!(solution.best_bound, None);
        assert!(!solution.is_optimal);
        assert!(solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);
    }

    #[test]
    fn test_hd1_parallel_cabs() {
        let dp = MockDp(2);
        let parameters = SearchParameters {
            quiet: true,
            ..Default::default()
        };
        let cabs_parameters = CabsParameters::default();
        let mut search = create_cahdbs1(dp, parameters, cabs_parameters, 8);

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
    fn test_hd1_parallel_cabs_infeasible() {
        let dp = MockDp(2);
        let parameters = SearchParameters {
            primal_bound: Some(2),
            quiet: true,
            ..Default::default()
        };
        let cabs_parameters = CabsParameters::default();
        let mut search = create_cahdbs1(dp, parameters, cabs_parameters, 8);

        let solution = search.search();
        assert_eq!(solution.cost, None);
        assert_eq!(solution.transitions, vec![]);
        assert_eq!(solution.best_bound, None);
        assert!(!solution.is_optimal);
        assert!(solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);
    }

    #[test]
    fn test_blind_hd1_parallel_cabs() {
        let dp = MockDp(2);
        let parameters = SearchParameters {
            quiet: true,
            ..Default::default()
        };
        let cabs_parameters = CabsParameters::default();
        let mut search = create_blind_cahdbs1(dp, parameters, cabs_parameters, 8);

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
    fn test_blind_hd1_parallel_cabs_infeasible() {
        let dp = MockDp(2);
        let parameters = SearchParameters {
            primal_bound: Some(2),
            quiet: true,
            ..Default::default()
        };
        let cabs_parameters = CabsParameters::default();
        let mut search = create_blind_cahdbs1(dp, parameters, cabs_parameters, 8);

        let solution = search.search();
        assert_eq!(solution.cost, None);
        assert_eq!(solution.transitions, vec![]);
        assert_eq!(solution.best_bound, None);
        assert!(!solution.is_optimal);
        assert!(solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);
    }

    #[test]
    fn test_hd2_parallel_cabs() {
        let dp = MockDp(2);
        let parameters = SearchParameters {
            quiet: true,
            ..Default::default()
        };
        let cabs_parameters = CabsParameters::default();
        let mut search = create_parallel_cabs(dp, parameters, cabs_parameters, 8);

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
    fn test_hd2_parallel_cabs_infeasible() {
        let dp = MockDp(2);
        let parameters = SearchParameters {
            primal_bound: Some(2),
            quiet: true,
            ..Default::default()
        };
        let cabs_parameters = CabsParameters::default();
        let mut search = create_parallel_cabs(dp, parameters, cabs_parameters, 8);

        let solution = search.search();
        assert_eq!(solution.cost, None);
        assert_eq!(solution.transitions, vec![]);
        assert_eq!(solution.best_bound, None);
        assert!(!solution.is_optimal);
        assert!(solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);
    }

    #[test]
    fn test_blind_hd2_parallel_cabs() {
        let dp = MockDp(2);
        let parameters = SearchParameters {
            quiet: true,
            ..Default::default()
        };
        let cabs_parameters = CabsParameters::default();
        let mut search = create_blind_parallel_cabs(dp, parameters, cabs_parameters, 8);

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
    fn test_blind_hd2_parallel_cabs_infeasible() {
        let dp = MockDp(2);
        let parameters = SearchParameters {
            primal_bound: Some(2),
            quiet: true,
            ..Default::default()
        };
        let cabs_parameters = CabsParameters::default();
        let mut search = create_blind_parallel_cabs(dp, parameters, cabs_parameters, 8);

        let solution = search.search();
        assert_eq!(solution.cost, None);
        assert_eq!(solution.transitions, vec![]);
        assert_eq!(solution.best_bound, None);
        assert!(!solution.is_optimal);
        assert!(solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);
    }

    #[test]
    fn test_casbs() {
        let dp = MockDp(2);
        let parameters = SearchParameters {
            quiet: true,
            ..Default::default()
        };
        let cabs_parameters = CabsParameters::default();
        let mut search = create_casbs(dp, parameters, cabs_parameters, 8);

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
    fn test_casbs_infeasible() {
        let dp = MockDp(2);
        let parameters = SearchParameters {
            primal_bound: Some(2),
            quiet: true,
            ..Default::default()
        };
        let cabs_parameters = CabsParameters::default();
        let mut search = create_casbs(dp, parameters, cabs_parameters, 8);

        let solution = search.search();
        assert_eq!(solution.cost, None);
        assert_eq!(solution.transitions, vec![]);
        assert_eq!(solution.best_bound, None);
        assert!(!solution.is_optimal);
        assert!(solution.is_infeasible);
        assert!(!solution.is_time_limit_reached);
        assert!(!solution.is_expansion_limit_reached);
    }

    #[test]
    fn test_blind_casbs() {
        let dp = MockDp(2);
        let parameters = SearchParameters {
            quiet: true,
            ..Default::default()
        };
        let cabs_parameters = CabsParameters::default();
        let mut search = create_blind_casbs(dp, parameters, cabs_parameters, 8);

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
    fn test_blind_casbs_infeasible() {
        let dp = MockDp(2);
        let parameters = SearchParameters {
            primal_bound: Some(2),
            quiet: true,
            ..Default::default()
        };
        let cabs_parameters = CabsParameters::default();
        let mut search = create_blind_casbs(dp, parameters, cabs_parameters, 8);

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
