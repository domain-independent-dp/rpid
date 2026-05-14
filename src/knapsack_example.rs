use crate::{Bound, Dominance, Dp, OptimizationMode};


struct Knapsack {n: usize, c: i32, p: Vec<i32>, w: Vec<i32>}

impl Dp for Knapsack {
    type State = (i32, usize);
    type CostType = i32;
    type Label = bool;

    fn get_target(&self) -> Self::State { (self.c, 0) }
    fn get_successors(
            &self,
            &(r, i): &Self::State,
        ) -> impl IntoIterator<Item = (Self::State, Self::CostType, Self::Label)> {
        if r >= self.w[i] {
            vec![((r - self.w[i], i + 1), self.p[i], true), ((r, i + 1), 0, false)]
        }
        else {
            vec![((r, i + 1), 0, false)]
        }
    }
    fn get_base_cost(&self, &(_, i): &Self::State) -> Option<Self::CostType> {
        if i == self.n { Some(0) } else { None }
    }
    fn get_optimization_mode(&self) -> crate::OptimizationMode { 
        OptimizationMode::Maximization
    }
}

impl Dominance for Knapsack {
    type State = (i32, usize);
    type Key = usize;

    fn get_key(&self, &(_, i): &Self::State) -> Self::Key {
        i
    }
    fn compare(&self, (r, _): &Self::State, (q, _): &Self::State) 
            -> Option<std::cmp::Ordering> {
        Some(r.cmp(q))
    }
}

impl Bound for Knapsack {
    type State = (i32, usize);
    type CostType = i32;

    fn get_dual_bound(&self, &(r, i): &Self::State) 
            -> Option<Self::CostType> {
        let mut bound = 0;
        let mut r = r;
        for j in i..self.n {
            if r >= self.w[j] {
                bound += self.p[j];
                r -= self.w[j]
            } else {
                bound += (((r * self.p[j]) as f64) / (self.w[j] as f64))
                            .floor() as i32;
            }
        }
        Some(bound)
    }
}