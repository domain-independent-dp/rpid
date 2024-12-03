# RPID -- Rust Programmable Interface for Domain-Independent Dynamic Programming

[![Actions Status](https://img.shields.io/github/actions/workflow/status/domain-independent-dp/rpid/test.yaml?branch=main&logo=github&style=flat-square)](https://github.com/domain-independent-dp/rpid/actions)
[![crates.io](https://img.shields.io/crates/v/rpid)](https://crates.io/crates/rpid)
[![minimum rustc 1.76](https://img.shields.io/badge/rustc-1.76+-blue.svg)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Example

```rust
use fixedbitset::FixedBitSet;
use rpid::prelude::*;
use rpid::solvers;

struct Tsp {
    c: Vec<Vec<i32>>,
}

struct TspState {
    unvisited: FixedBitSet,
    current: usize,
}

impl Dp for Tsp {
    type State = TspState;
    type CostType = i32;

    fn get_target(&self) -> TspState {
        let mut unvisited = FixedBitSet::with_capacity(self.c.len());
        unvisited.insert_range(1..);

        TspState {
            unvisited,
            current: 0,
        }
    }

    fn get_successors(&self, state: &TspState) -> impl IntoIterator<Item = (TspState, i32, usize)> {
        state.unvisited.ones().map(|next| {
            let mut unvisited = state.unvisited.clone();
            unvisited.remove(next);

            let successor = TspState {
                unvisited,
                current: next,
            };
            let weight = self.c[state.current][next];

            (successor, weight, next)
        })
    }

    fn get_base_cost(&self, state: &TspState) -> Option<i32> {
        if state.unvisited.is_clear() {
            Some(self.c[state.current][0])
        } else {
            None
        }
    }
}

impl Dominance for Tsp {
    type State = TspState;
    type Key = (FixedBitSet, usize);

    fn get_key(&self, state: &TspState) -> Self::Key {
        (state.unvisited.clone(), state.current)
    }
}

impl Bound for Tsp {
    type State = TspState;
    type CostType = i32;

    fn get_dual_bound(&self, _: &TspState) -> Option<i32> {
        Some(0)
    }
}

fn main() {
    let tsp = Tsp {
        c: vec![vec![0, 1, 2], vec![1, 0, 3], vec![2, 3, 0]],
    };
    let mut solver =
        solvers::create_cabs(tsp, SearchParameters::default(), CabsParameters::default());
    let solution = solver.search();
    assert_eq!(solution.cost, Some(6));
    assert_eq!(solution.transitions, vec![1, 2]);
    assert!(solution.is_optimal);
    assert!(!solution.is_infeasible);
    assert_eq!(solution.best_bound, Some(6));
}
```
