use num_traits::Zero;
use std::cmp::Ordering;
use std::hash::Hash;
use std::ops::Add;

/// Optimization mode for dynamic programming problems.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationMode {
    /// Minimization: the goal is to minimize the cost.
    Minimization,
    /// Maximization: the goal is to maximize the cost.
    Maximization,
}

/// Trait for dynamic programming problems.
///
/// This trait defines the methods that a dynamic programming problem must implement.
/// Data necessary for the problem should be stored in the struct that implements this trait.
///
/// By default, the solution cost is computed by summing the cost weights of the transitions,
/// and minimization is assumed.
/// Override the methods to change this behavior.
///
/// # Examples
///
/// ```
/// use rpid::prelude::*;
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
/// let tsp = Tsp { c: vec![vec![0, 1, 2], vec![1, 0, 3], vec![2, 3, 0]] };
///
/// let target = tsp.get_target();
/// assert_eq!(target.current, 0);
/// let mut unvisited = FixedBitSet::with_capacity(3);
/// unvisited.insert_range(1..);
/// assert_eq!(target.unvisited, unvisited);
///
/// let successors: Vec<_> = tsp.get_successors(&target).into_iter().collect();
/// assert_eq!(successors.len(), 2);
///
/// assert_eq!(successors[0].0.current, 1);
/// let mut unvisited_1 = unvisited.clone();
/// unvisited_1.remove(1);
/// assert_eq!(successors[0].0.unvisited, unvisited_1);
/// assert_eq!(successors[0].1, 1);
/// assert_eq!(successors[0].2, 1);
///
/// assert_eq!(successors[1].0.current, 2);
/// let mut unvisited_2 = unvisited.clone();
/// unvisited_2.remove(2);
/// assert_eq!(successors[1].0.unvisited, unvisited_2);
/// assert_eq!(successors[1].1, 2);
/// assert_eq!(successors[1].2, 2);
///
/// assert_eq!(tsp.get_base_cost(&target), None);
/// ```
pub trait Dp {
    /// Type of the state.
    type State;
    /// Type of the cost. Usually, `i32` or `f64`.
    type CostType: PartialOrd + Add<Output = Self::CostType> + Zero;
    /// Type of the transition label. Usually, an unsigned integer type or `bool` (e.g., the knapsack problem).
    type Label;

    /// Gets the target (initial) state.
    fn get_target(&self) -> Self::State;

    /// Gets the successors of a state.
    ///
    /// The easiest way to implement this method is to return a vector (`Vec<(Self::State, Self::CostType, Self::Label)>`)
    /// or an array ([`(Self::State, Self::CostType, Self::Label); N]`), where the first element of a tuple is the successor state,
    /// the second element is the cost weight of the transition, and the third element is the label of the transition.
    /// However, by returning an iterator, you can avoid allocating memory for the successors.
    fn get_successors(
        &self,
        state: &Self::State,
    ) -> impl IntoIterator<Item = (Self::State, Self::CostType, Self::Label)>;

    /// Checks if a state is a base (goal) state and returns the base cost if it is.
    fn get_base_cost(&self, state: &Self::State) -> Option<Self::CostType>;

    /// Combines two cost weights.
    ///
    /// This method is used to combine the cost weights of two transitions.
    /// Addition is used by default, but you can override this method to use a different operation.
    /// However, the operation must be associative and isotone, e.g.,
    /// (a + b) + c = a + (b + c) and a ≤ b → a + c ≤ b + c.
    fn combine_cost_weights(&self, a: Self::CostType, b: Self::CostType) -> Self::CostType {
        a + b
    }

    /// Gets the identity weight.
    ///
    /// This method returns the identity element of the `combine_cost_weights` operation, e.g.,
    /// a + 0 = 0 + a = a.
    /// The default implementation returns `Self::CostType::zero()`.
    fn get_identity_weight(&self) -> Self::CostType {
        Self::CostType::zero()
    }

    /// Returns the optimization mode of the problem.
    ///
    /// The default implementation returns `OptimizationMode::Minimization`.
    fn get_optimization_mode(&self) -> OptimizationMode {
        OptimizationMode::Minimization
    }

    /// Returns whether the cost is better than the old cost.
    ///
    /// By default, this method follows the maximization flag.
    fn is_better_cost(&self, new_cost: Self::CostType, old_cost: Self::CostType) -> bool {
        match self.get_optimization_mode() {
            OptimizationMode::Minimization => new_cost < old_cost,
            OptimizationMode::Maximization => new_cost > old_cost,
        }
    }
}

/// Trait for dominance relations.
///
/// # Examples
///
/// ```
/// use rpid::prelude::*;
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
///
/// let unvisited = FixedBitSet::with_capacity(3);
/// let current = 0;
/// let key = (unvisited.clone(), current);
/// let state = TspState { unvisited, current };
/// assert_eq!(tsp.get_key(&state), key);
/// ```
pub trait Dominance {
    /// Type of the state.
    type State;
    /// Type of the key.
    type Key: Hash + Eq;

    /// Gets the key of a state.
    ///
    /// A state possibly dominates or is dominated by another state if their keys are equal.
    /// If only this method is implemented, the states are the same if their keys are equal.
    fn get_key(&self, state: &Self::State) -> Self::Key;

    /// Compares two states with the same key.
    ///
    /// Returns `Some(Ordering::Greater)` if the first state dominates the second state,
    /// `Some(Ordering::Less)` if the second state dominates the first state,
    /// `Some(Ordering::Equal)` if they are the same, and `None` if they are incomparable.
    /// By default, this method returns `Some(Ordering::Equal)`, which means that
    /// the states are the same if their keys are equal.
    fn compare(&self, _a: &Self::State, _b: &Self::State) -> Option<Ordering> {
        Some(Ordering::Equal)
    }

    /// Updates the key of a state.
    ///
    /// This method is not necessarily implemented.
    /// It is useful to reduce memory usage when the key of a state is stored as a shared pointer.
    fn update_key(&self, _state: &mut Self::State, _key: &Self::Key) {}
}

/// Trait for computing dual bounds depending on a state.
///
/// A dual bound is a lower/upper bound on the cost of the optimal solution
/// in a minimization/maximization problem.
///
/// ```
/// use rpid::prelude::*;
/// use rpid::algorithms;
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
/// impl Bound for Tsp {
///     type State = TspState;
///     type CostType = i32;
///
///     fn get_dual_bound(&self, _: &Self::State) -> Option<Self::CostType> {
///         Some(0)
///     }
/// }
///
/// let tsp = Tsp { c: vec![vec![0, 1, 2], vec![1, 0, 3], vec![2, 3, 0]] };
///
/// let unvisited = FixedBitSet::with_capacity(3);
/// let current = 0;
/// let state = TspState { unvisited, current };
/// assert_eq!(tsp.get_dual_bound(&state), Some(0));
/// ```
pub trait Bound {
    /// Type of the state.
    type State;
    /// Type of the cost.
    type CostType;

    /// Gets the dual bound of a state.
    ///
    /// Returns `None` if the state is not feasible.
    fn get_dual_bound(&self, state: &Self::State) -> Option<Self::CostType>;

    /// Gets the global primal bound.
    ///
    /// The default implementation returns `None`, which means that the global primal bound is not used.
    /// The global primal bound is an upper/lower bound on the cost of the optimal solution in a minimization/maximization problem.
    fn get_global_primal_bound(&self) -> Option<Self::CostType> {
        None
    }

    /// Gets the global dual bound.
    ///
    /// The default implementation returns `None`, which means that the global dual bound is not used.
    /// The global dual bound is an lower/upper bound on the cost of the optimal solution in a minimization/maximization problem.
    fn get_global_dual_bound(&self) -> Option<Self::CostType> {
        None
    }
}

/// Trait for dynamic programming problems.
///
/// This trait is similar to the `Dp` trait, but it allows mutable access to the problem struct
/// when generating successors and get the base cost.
/// This is useful when the problem struct contains data that needs to be updated during the search.
///
/// This trait defines the methods that a dynamic programming problem must implement.
/// Data necessary for the problem should be stored in the struct that implements this trait, e.g.,
/// some caching.
///
/// By default, the solution cost is computed by summing the cost weights of the transitions,
/// and minimization is assumed.
/// Override the methods to change this behavior.
pub trait DpMut {
    /// Type of the state.
    type State;
    /// Type of the cost. Usually, `i32` or `f64`.
    type CostType: PartialOrd + Add<Output = Self::CostType> + Zero;
    /// Type of the transition label. Usually, an unsigned integer type or `bool` (e.g., the knapsack problem).
    type Label;

    /// Gets the target (initial) state.
    fn get_target(&self) -> Self::State;

    /// Gets the successors of a state.
    ///
    /// The easiest way to implement this method is to return a vector (`Vec<(Self::State, Self::CostType, Self::Label)>`)
    /// or an array ([`(Self::State, Self::CostType, Self::Label); N]`), where the first element of a tuple is the successor state,
    /// the second element is the cost weight of the transition, and the third element is the label of the transition.
    /// However, by returning an iterator, you can avoid allocating memory for the successors.
    fn get_successors(
        &mut self,
        state: &Self::State,
        successors: &mut Vec<(Self::State, Self::CostType, Self::Label)>,
    );

    /// Checks if a state is a base (goal) state and returns the base cost if it is.
    fn get_base_cost(&mut self, state: &Self::State) -> Option<Self::CostType>;

    /// Combines two cost weights.
    ///
    /// This method is used to combine the cost weights of two transitions.
    /// Addition is used by default, but you can override this method to use a different operation.
    /// However, the operation must be associative and isotone, e.g.,
    /// (a + b) + c = a + (b + c) and a ≤ b → a + c ≤ b + c.
    fn combine_cost_weights(&self, a: Self::CostType, b: Self::CostType) -> Self::CostType {
        a + b
    }

    /// Gets the identity weight.
    ///
    /// This method returns the identity element of the `combine_cost_weights` operation, e.g.,
    /// a + 0 = 0 + a = a.
    /// The default implementation returns `Self::CostType::zero()`.
    fn get_identity_weight(&self) -> Self::CostType {
        Self::CostType::zero()
    }

    /// Returns the optimization mode of the problem.
    ///
    /// The default implementation returns `OptimizationMode::Minimization`.
    fn get_optimization_mode(&self) -> OptimizationMode {
        OptimizationMode::Minimization
    }

    /// Returns whether the cost is better than the old cost.
    ///
    /// By default, this method follows the maximization flag.
    fn is_better_cost(&self, new_cost: Self::CostType, old_cost: Self::CostType) -> bool {
        match self.get_optimization_mode() {
            OptimizationMode::Minimization => new_cost < old_cost,
            OptimizationMode::Maximization => new_cost > old_cost,
        }
    }

    /// Notifies information of a new primal bound.
    fn notify_primal_bound(&mut self, _primal_bound: Self::CostType) {}
}

impl<T: Dp> DpMut for T {
    type State = T::State;
    type CostType = T::CostType;
    type Label = T::Label;

    fn get_target(&self) -> Self::State {
        Dp::get_target(self)
    }

    fn get_successors(
        &mut self,
        state: &Self::State,
        successors: &mut Vec<(Self::State, Self::CostType, Self::Label)>,
    ) {
        successors.extend(Dp::get_successors(self, state));
    }

    fn get_base_cost(&mut self, state: &Self::State) -> Option<Self::CostType> {
        Dp::get_base_cost(self, state)
    }

    fn combine_cost_weights(&self, a: Self::CostType, b: Self::CostType) -> Self::CostType {
        Dp::combine_cost_weights(self, a, b)
    }

    fn get_identity_weight(&self) -> Self::CostType {
        Dp::get_identity_weight(self)
    }

    fn get_optimization_mode(&self) -> OptimizationMode {
        Dp::get_optimization_mode(self)
    }

    fn is_better_cost(&self, new_cost: Self::CostType, old_cost: Self::CostType) -> bool {
        Dp::is_better_cost(self, new_cost, old_cost)
    }
}

/// Trait for computing dual bounds depending on a state.
///
/// This trait is similar to the `Bound` trait, but it allows mutable access to the problem struct
/// when evaluating the dual bound function.
/// This is useful when the problem struct contains data that needs to be updated during the search, e.g.,
/// some caching.
///
/// A dual bound is a lower/upper bound on the cost of the optimal solution
/// in a minimization/maximization problem.
pub trait BoundMut {
    /// Type of the state.
    type State;
    /// Type of the cost.
    type CostType;

    /// Gets the dual bound of a state.
    ///
    /// Returns `None` if the state is not feasible.
    fn get_dual_bound(&mut self, state: &Self::State) -> Option<Self::CostType>;

    /// Gets the global primal bound.
    ///
    /// The default implementation returns `None`, which means that the global primal bound is not used.
    /// The global primal bound is an upper/lower bound on the cost of the optimal solution in a minimization/maximization problem.
    fn get_global_primal_bound(&self) -> Option<Self::CostType> {
        None
    }

    /// Gets the global dual bound.
    ///
    /// The default implementation returns `None`, which means that the global dual bound is not used.
    /// The global dual bound is an lower/upper bound on the cost of the optimal solution in a minimization/maximization problem.
    fn get_global_dual_bound(&self) -> Option<Self::CostType> {
        None
    }
}

impl<T: Bound> BoundMut for T {
    type State = T::State;
    type CostType = T::CostType;

    fn get_dual_bound(&mut self, state: &Self::State) -> Option<Self::CostType> {
        Bound::get_dual_bound(self, state)
    }

    fn get_global_primal_bound(&self) -> Option<Self::CostType> {
        Bound::get_global_primal_bound(self)
    }

    fn get_global_dual_bound(&self) -> Option<Self::CostType> {
        Bound::get_global_dual_bound(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockDp;

    impl Dp for MockDp {
        type State = i32;
        type CostType = i32;
        type Label = usize;

        fn get_target(&self) -> Self::State {
            0
        }

        fn get_successors(
            &self,
            _: &Self::State,
        ) -> impl IntoIterator<Item = (Self::State, Self::CostType, Self::Label)> {
            vec![]
        }

        fn get_base_cost(&self, _state: &Self::State) -> Option<Self::CostType> {
            None
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
    fn test_get_target_mut() {
        let dp = MockDp;
        assert_eq!(DpMut::get_target(&dp), 0);
    }

    #[test]
    fn test_get_successors_mut() {
        let mut dp = MockDp;
        let mut successors = Vec::new();
        DpMut::get_successors(&mut dp, &0, &mut successors);
        assert!(successors.is_empty());
    }

    #[test]
    fn get_base_cost_mut() {
        let mut dp = MockDp;
        assert_eq!(DpMut::get_base_cost(&mut dp, &0), None);
    }

    #[test]
    fn test_combine_cost_weights_default() {
        let dp = MockDp;
        assert_eq!(Dp::combine_cost_weights(&dp, 1, 2), 3);
    }

    #[test]
    fn test_combine_cost_weights_mut() {
        let dp = MockDp;
        assert_eq!(DpMut::combine_cost_weights(&dp, 1, 2), 3);
    }

    #[test]
    fn test_get_identity_weight_default() {
        let dp = MockDp;
        assert_eq!(Dp::get_identity_weight(&dp), 0);
    }

    #[test]
    fn test_get_identity_weight_mut() {
        let dp = MockDp;
        assert_eq!(DpMut::get_identity_weight(&dp), 0);
    }

    #[test]
    fn test_get_optimization_mode_default() {
        let dp = MockDp;
        assert_eq!(
            Dp::get_optimization_mode(&dp),
            OptimizationMode::Minimization
        );
    }

    #[test]
    fn test_get_optimization_mode_mut() {
        let dp = MockDp;
        assert_eq!(
            DpMut::get_optimization_mode(&dp),
            OptimizationMode::Minimization
        );
    }

    #[test]
    fn test_is_better_cost_default() {
        let dp = MockDp;
        assert!(Dp::is_better_cost(&dp, 1, 2));
    }

    #[test]
    fn test_is_better_cost_mut() {
        let dp = MockDp;
        assert!(DpMut::is_better_cost(&dp, 1, 2));
    }

    #[test]
    fn test_compare_default() {
        let dp = MockDp;
        assert_eq!(dp.compare(&0, &0), Some(Ordering::Equal));
    }

    #[test]
    fn test_get_global_primal_bound_default() {
        let dp = MockDp;
        assert_eq!(Bound::get_global_primal_bound(&dp), None);
    }

    #[test]
    fn test_get_global_primal_bound_mut() {
        let dp = MockDp;
        assert_eq!(Bound::get_global_primal_bound(&dp), None);
    }

    #[test]
    fn test_get_dual_bound_mut() {
        let mut dp = MockDp;
        assert_eq!(BoundMut::get_dual_bound(&mut dp, &0), Some(0));
    }

    #[test]
    fn test_get_global_dual_bound_default() {
        let dp = MockDp;
        assert_eq!(Bound::get_global_dual_bound(&dp), None);
    }

    #[test]
    fn test_get_global_dual_bound_mut() {
        let dp = MockDp;
        assert_eq!(BoundMut::get_global_dual_bound(&dp), None);
    }

    struct MockDpMut;

    impl DpMut for MockDpMut {
        type State = i32;
        type CostType = i32;
        type Label = usize;

        fn get_target(&self) -> Self::State {
            0
        }

        fn get_successors(
            &mut self,
            _: &Self::State,
            _: &mut Vec<(Self::State, Self::CostType, Self::Label)>,
        ) {
        }

        fn get_base_cost(&mut self, _state: &Self::State) -> Option<Self::CostType> {
            None
        }
    }

    impl BoundMut for MockDpMut {
        type State = i32;
        type CostType = i32;

        fn get_dual_bound(&mut self, _: &Self::State) -> Option<Self::CostType> {
            Some(0)
        }
    }

    #[test]
    fn test_combine_cost_weights_mut_default() {
        let dp = MockDpMut;
        assert_eq!(dp.combine_cost_weights(1, 2), 3);
    }

    #[test]
    fn test_get_identity_weight_mut_default() {
        let dp = MockDpMut;
        assert_eq!(dp.get_identity_weight(), 0);
    }

    #[test]
    fn test_get_optimization_mode_mut_default() {
        let dp = MockDpMut;
        assert_eq!(dp.get_optimization_mode(), OptimizationMode::Minimization);
    }

    #[test]
    fn test_is_better_cost_mut_default() {
        let dp = MockDpMut;
        assert!(dp.is_better_cost(1, 2));
    }

    #[test]
    fn test_get_global_primal_bound_mut_default() {
        let dp = MockDpMut;
        assert_eq!(dp.get_global_primal_bound(), None);
    }

    #[test]
    fn test_get_global_dual_bound_mut_default() {
        let dp = MockDpMut;
        assert_eq!(dp.get_global_dual_bound(), None);
    }

    #[test]
    fn test_notify_primal_bound_mut_default() {
        let mut dp = MockDpMut;
        dp.notify_primal_bound(42);
    }
}
