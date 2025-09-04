//! Rust Programmable Interface for Domain-Independent Dynamic Programming (RPID)
//!
//! With RPID, you can formulate dynamic programming by implementing traits and
//! solve it by calling a solver.
//! You need to implement the [Dp] trait to define a dynamic programming model,
//! and a solver may require the [Dominance] and [Bound] traits to enable dominance checking and pruning with bounds.
//!
//! ## Example
//!
//! Solving the traveling salesperson problem (TSP) with the CABS solver.
//!
//! ```
//! use fixedbitset::FixedBitSet;
//! use rpid::prelude::*;
//! use rpid::solvers;
//!
//! struct Tsp {
//!     c: Vec<Vec<i32>>,
//! }
//!
//! struct TspState {
//!     unvisited: FixedBitSet,
//!     current: usize,
//! }
//!
//! impl Dp for Tsp {
//!     type State = TspState;
//!     type CostType = i32;
//!     type Label = usize;
//!
//!     fn get_target(&self) -> Self::State {
//!         let mut unvisited = FixedBitSet::with_capacity(self.c.len());
//!         unvisited.insert_range(1..);
//!
//!         TspState {
//!             unvisited,
//!             current: 0,
//!         }
//!     }
//!
//!     fn get_successors(
//!         &self,
//!         state: &Self::State,
//!     ) -> impl IntoIterator<Item = (Self::State, Self::CostType, Self::Label)> {
//!         state.unvisited.ones().map(|next| {
//!             let mut unvisited = state.unvisited.clone();
//!             unvisited.remove(next);
//!
//!             let successor = TspState {
//!                 unvisited,
//!                 current: next,
//!             };
//!             let weight = self.c[state.current][next];
//!
//!             (successor, weight, next)
//!         })
//!     }
//!
//!     fn get_base_cost(&self, state: &Self::State) -> Option<Self::CostType> {
//!         if state.unvisited.is_clear() {
//!             Some(self.c[state.current][0])
//!         } else {
//!             None
//!         }
//!     }
//! }
//!
//! impl Dominance for Tsp {
//!     type State = TspState;
//!     type Key = (FixedBitSet, usize);
//!
//!     fn get_key(&self, state: &Self::State) -> Self::Key {
//!         (state.unvisited.clone(), state.current)
//!     }
//! }
//!
//! impl Bound for Tsp {
//!     type State = TspState;
//!     type CostType = i32;
//!
//!     fn get_dual_bound(&self, _: &Self::State) -> Option<Self::CostType> {
//!         Some(0)
//!     }
//! }
//!
//! let tsp = Tsp { c: vec![vec![0, 1, 2], vec![1, 0, 3], vec![2, 3, 0]] };
//! let mut solver
//!     = solvers::create_cabs(tsp, SearchParameters::default(), CabsParameters::default());
//! let solution = solver.search();
//! assert_eq!(solution.cost, Some(6));
//! assert_eq!(solution.transitions, vec![1, 2]);
//! assert!(solution.is_optimal);
//! assert!(!solution.is_infeasible);
//! assert_eq!(solution.best_bound, Some(6));
//! ```
//!
//! For more examples, see <https://github.com/Kurorororo/didp-rust-models>.
//!
//! ## References
//!
//! Ryo Kuroiwa and J. Christopher Beck. RPID: Rust Programmable Interface for Domain-Independent Dynamic Programming.
//! In *31st International Conference on Principles and Practice of Constraint Programming (CP 2025)*,
//! volume 340 of Leibniz International Proceedings in Informatics (LIPIcs), pages 23:1-23:21.
//! Schloss Dagstuhl - Leibniz-Zentrum für Informatik, 2025.
//! [doi:10.4230/LIPIcs.CP.2025.23](https://doi.org/10.4230/LIPIcs.CP.2025.23)

pub mod algorithms;
mod dp;
pub mod io;
pub mod solvers;
pub mod timer;

pub use dp::{Bound, BoundMut, Dominance, Dp, DpMut, OptimizationMode};
pub use solvers::Solution;

pub mod prelude {
    //! Prelude to import commonly used items.
    pub use super::solvers::{CabsParameters, Search, SearchParameters};
    pub use super::{Bound, Dominance, Dp, OptimizationMode, Solution};
}
