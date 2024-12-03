use crate::solvers::{Search, Solution};
use itertools::Itertools;
use std::error::Error;
use std::fmt::Display;
use std::fs::OpenOptions;
use std::io::Write;
use std::str::FromStr;

/// Reads a vector of length n from an input str stream.
///
/// # Examples
///
/// ```
/// use rpid::io;
/// use std::fs;
///
/// let string = "1 2 3 4 5 6 7 8 9 10";
///
/// let mut lines = string.split_whitespace();
/// let vector = io::read_vector::<i32>(&mut lines, 10).unwrap();
/// assert_eq!(vector, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
/// ```
pub fn read_vector<'a, T>(
    input: &mut impl Iterator<Item = &'a str>,
    n: usize,
) -> Result<Vec<T>, <T as FromStr>::Err>
where
    T: FromStr,
{
    input.take(n).map(|x| x.parse::<T>()).collect()
}

/// Reads an m x n matrix from an input str stream.
///
/// # Examples
///
/// ```
/// use rpid::io;
/// use std::fs;
///
/// let string = "1 2 3 \n 4 5 6 \n 7 8 9";
///
/// let mut lines = string.split_whitespace();
/// let matrix = io::read_matrix::<i32>(&mut lines, 3, 3).unwrap();
/// assert_eq!(matrix, vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]);
/// ```
pub fn read_matrix<'a, T>(
    input: &mut impl Iterator<Item = &'a str>,
    m: usize,
    n: usize,
) -> Result<Vec<Vec<T>>, <T as FromStr>::Err>
where
    T: FromStr,
{
    input
        .take(m * n)
        .chunks(n)
        .into_iter()
        .map(|chunk| chunk.into_iter().map(|x| x.parse::<T>()).collect())
        .collect()
}

/// Run a solver and dump the solution history to a CSV file.
///
/// The first field is the time, second is the cost, third is the bound, fourth is the transitions,
/// fifth is the expanded, and sixth is the generated.
pub fn run_solver_and_dump_solution_history<S, C>(
    solver: &mut S,
    filename: &str,
) -> Result<Solution<C>, Box<dyn Error>>
where
    S: Search<CostType = C>,
    C: Display + Copy,
{
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(filename)?;

    loop {
        let (solution, terminated) = solver.search_next();

        if let Some(cost) = solution.cost {
            let transitions = solution
                .transitions
                .iter()
                .map(|t| format!("{}", t))
                .collect::<Vec<_>>()
                .join(" ");

            let line = if let Some(bound) = solution.best_bound {
                format!(
                    "{}, {}, {}, {}, {}, {}\n",
                    solution.time, cost, bound, transitions, solution.expanded, solution.generated
                )
            } else {
                format!(
                    "{}, {}, , {}, {}, {}\n",
                    solution.time, cost, transitions, solution.expanded, solution.generated
                )
            };
            file.write_all(line.as_bytes())?;
            file.flush()?;
        }

        if terminated {
            return Ok(solution);
        }
    }
}

/// Print the cost, bound, and statistics of a solution.
pub fn print_solution_statistics<C>(solution: &Solution<C>)
where
    C: Copy + Display,
{
    if let Some(cost) = solution.cost {
        println!("cost: {}", cost);

        if solution.is_optimal {
            println!("optimal cost: {}", cost);
        }
    } else {
        println!("No solution is found.");

        if solution.is_infeasible {
            println!("The problem is infeasible.");
        }
    }

    if let Some(bound) = solution.best_bound {
        println!("best bound: {}", bound);
    }

    println!("Search time: {}s", solution.time);
    println!("Expanded: {}", solution.expanded);
    println!("Generated: {}", solution.generated);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_vector() {
        let input = "1 2 3 4 5 6 \n 7 8 \n 9 10 11 12";
        let mut lines = input.split_whitespace();

        let matrix = read_vector::<i32>(&mut lines, 7);
        assert!(matrix.is_ok());
        assert_eq!(matrix.unwrap(), vec![1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_read_matrix() {
        let input = "1 2 3 4 5 6 \n 7 8 \n 9 10 11 12";
        let mut lines = input.split_whitespace();

        let matrix = read_matrix::<i32>(&mut lines, 3, 3);
        assert!(matrix.is_ok());
        assert_eq!(
            matrix.unwrap(),
            vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]
        );
    }
}
