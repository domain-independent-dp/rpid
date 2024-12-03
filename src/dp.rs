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

    /// Gets the target (initial) state.
    fn get_target(&self) -> Self::State;

    /// Gets the successors of a state.
    ///
    /// The easiest way to implement this method is to return a vector (`Vec<(Self::State, Self::CostType, usize)>`)
    /// or an array ([`(Self::State, Self::CostType, usize); N]`), where the first element of a tuple is the successor state,
    /// the second element is the cost weight of the transition, and the third element is the index of the transition.
    /// However, by returning an iterator, you can avoid allocating memory for the successors.
    fn get_successors(
        &self,
        state: &Self::State,
    ) -> impl IntoIterator<Item = (Self::State, Self::CostType, usize)>;

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
///     fn get_key(&self, state: &TspState) -> Self::Key {
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
///     fn get_dual_bound(&self, _: &TspState) -> Option<i32> {
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

#[cfg(test)]
mod tests {
    use super::*;

    struct MockDp;

    impl Dp for MockDp {
        type State = i32;
        type CostType = i32;

        fn get_target(&self) -> Self::State {
            0
        }

        fn get_successors(
            &self,
            _: &Self::State,
        ) -> impl IntoIterator<Item = (Self::State, Self::CostType, usize)> {
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
    fn test_combine_cost_weights() {
        let dp = MockDp;
        assert_eq!(dp.combine_cost_weights(1, 2), 3);
    }

    #[test]
    fn test_get_identity_weight() {
        let dp = MockDp;
        assert_eq!(dp.get_identity_weight(), 0);
    }

    #[test]
    fn test_get_optimization_mode() {
        let dp = MockDp;
        assert_eq!(dp.get_optimization_mode(), OptimizationMode::Minimization);
    }

    #[test]
    fn test_is_better_cost() {
        let dp = MockDp;
        assert!(dp.is_better_cost(1, 2));
    }

    #[test]
    fn test_compare() {
        let dp = MockDp;
        assert_eq!(dp.compare(&0, &0), Some(Ordering::Equal));
    }

    #[test]
    fn test_get_global_primal_bound() {
        let dp = MockDp;
        assert_eq!(dp.get_global_primal_bound(), None);
    }

    #[test]
    fn test_get_global_dual_bound() {
        let dp = MockDp;
        assert_eq!(dp.get_global_dual_bound(), None);
    }
}
