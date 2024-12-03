//! Algorithms for solving optimization problems.
//!
//! The algorithms in this module are useful for preprocessing and relaxation.

use num_traits::{One, Zero};
use std::cmp::Ordering;
use std::iter::Sum;
use std::ops::{Add, Div, Rem, SubAssign};

/// Transpose an m x n matrix.
///
/// # Examples
///
/// let string = "1 2 3 \n 4 5 6 \n 7 8 9";
///
/// let mut lines = string.split_whitespace();
/// let matrix = io::read_matrix::<i32>(&mut lines, 3, 3).unwrap();
/// let transposed = io::transpose(&matrix);
/// assert_eq!(transposed, vec![vec![1, 4, 7], vec![2, 5, 8], vec![3, 6, 9]]);
/// ```
pub fn transpose<T>(matrix: &[Vec<T>]) -> Vec<Vec<T>>
where
    T: Copy,
{
    if matrix.is_empty() {
        return vec![];
    }

    let m = matrix.len();
    let n = matrix.iter().map(|row| row.len()).max().unwrap();

    (0..n)
        .map(|i| (0..m).filter_map(|j| matrix[j].get(i)).copied().collect())
        .collect()
}

/// Computes the Euclidean distances between all pairs of points.
///
/// The input `points` should be a list of points, where each point is represented by a tuple of x and y coordinates.
///
/// # Examples
///
/// ```
/// use approx::assert_relative_eq;
/// use rpid::algorithms;
///
/// let points = vec![(0, 0), (3, 0), (3, 4)];
/// let distances = algorithms::compute_pairwise_euclidean_distances(&points);
/// let expected = [[0.0, 3.0, 5.0], [3.0, 0.0, 4.0], [5.0, 4.0, 0.0]];
/// assert_relative_eq!(distances[0][0], expected[0][0]);
/// assert_relative_eq!(distances[0][1], expected[0][1]);
/// assert_relative_eq!(distances[0][2], expected[0][2]);
/// assert_relative_eq!(distances[1][0], expected[1][0]);
/// assert_relative_eq!(distances[1][1], expected[1][1]);
/// assert_relative_eq!(distances[1][2], expected[1][2]);
/// assert_relative_eq!(distances[2][0], expected[2][0]);
/// assert_relative_eq!(distances[2][1], expected[2][1]);
/// assert_relative_eq!(distances[2][2], expected[2][2]);
/// ```
pub fn compute_pairwise_euclidean_distances<T>(points: &[(T, T)]) -> Vec<Vec<f64>>
where
    T: Copy,
    f64: From<T>,
{
    points
        .iter()
        .map(&|&(x1, y1)| {
            let x1 = f64::from(x1);
            let y1 = f64::from(y1);

            points
                .iter()
                .map(|&(x2, y2)| {
                    let x2 = f64::from(x2);
                    let y2 = f64::from(y2);

                    (x1 - x2).hypot(y1 - y2)
                })
                .collect()
        })
        .collect()
}

/// Computes the shortest path cost between all pairs of nodes in a graph.
///
/// `weights` should be a square matrix with the weights of the edges,
/// assuming that the graph is complete.
///
/// # Examples
///
/// ```
/// use rpid::algorithms;
///
/// let weights = vec![
///     vec![0, 1, 2],
///     vec![1, 0, 4],
///     vec![2, 4, 0],
/// ];
/// let shortest_paths = algorithms::compute_pairwise_shortest_path_costs(&weights);
/// let expected = vec![
///     vec![0, 1, 2],
///     vec![1, 0, 3],
///     vec![2, 3, 0],
/// ];
/// assert_eq!(shortest_paths, expected);
/// ```
pub fn compute_pairwise_shortest_path_costs<T>(weights: &[Vec<T>]) -> Vec<Vec<T>>
where
    T: Copy + PartialOrd + Add<Output = T>,
{
    let mut distance = Vec::from(weights);
    let n = distance.len();

    for k in 0..n {
        for i in 0..n {
            if i == k {
                continue;
            }

            for j in 0..n {
                if j == i || j == k {
                    continue;
                }

                if distance[i][k] + distance[k][j] < distance[i][j] {
                    distance[i][j] = distance[i][k] + distance[k][j];
                }
            }
        }
    }

    distance
}

/// Computes the shortest path cost between all pairs of nodes in a graph.
///
/// `weights` should be a square matrix with the weights of the edges.
/// If there is no edge between two nodes, the corresponding entry should be `None`.
///
/// # Examples
///
/// ```
/// use rpid::algorithms;
///
/// let weights = vec![
///     vec![None, Some(1), None],
///     vec![Some(1), None, Some(4)],
///     vec![Some(2), Some(4), None],
/// ];
/// let shortest_paths =
///     algorithms::compute_pairwise_shortest_path_costs_with_option(&weights);
/// let expected = vec![
///     vec![None, Some(1), Some(5)],
///     vec![Some(1), None, Some(4)],
///     vec![Some(2), Some(3), None],
/// ];
/// assert_eq!(shortest_paths, expected);
/// ```
pub fn compute_pairwise_shortest_path_costs_with_option<T>(
    weights: &[Vec<Option<T>>],
) -> Vec<Vec<Option<T>>>
where
    T: Copy + PartialOrd + Add<Output = T>,
{
    let mut distance = Vec::from(weights);
    let n = distance.len();

    for k in 0..n {
        for i in 0..n {
            if i == k {
                continue;
            }

            if let Some(d_ik) = distance[i][k] {
                for j in 0..n {
                    if j == i || j == k {
                        continue;
                    }

                    if let Some(d_kj) = distance[k][j] {
                        if let Some(d_ij) = distance[i][j] {
                            if d_ik + d_kj < d_ij {
                                distance[i][j] = Some(d_ik + d_kj);
                            }
                        } else {
                            distance[i][j] = Some(d_ik + d_kj);
                        }
                    }
                }
            }
        }
    }

    distance
}

/// Returns an iterator taking the minimum of each row in a matrix.
///
/// The iterator returns `None` if the row is empty.
///
/// # Examples
///
/// ```
/// use rpid::algorithms;
///
/// let matrix = vec![
///     vec![2, 9, 7],
///     vec![3, 6, 1],
///     vec![5, 4, 7],
/// ];
/// let result = algorithms::take_row_wise_min(&matrix).collect::<Vec<_>>();
/// let expected = vec![Some(2), Some(1), Some(4)];
/// assert_eq!(result, expected);
/// ```
pub fn take_row_wise_min<T>(matrix: &[Vec<T>]) -> impl Iterator<Item = Option<T>> + '_
where
    T: Copy + PartialOrd,
{
    matrix.iter().map(|row| {
        row.iter()
            .copied()
            .reduce(|a, b| if a <= b { a } else { b })
    })
}

/// Returns an iterator taking the minimum of each row in a matrix, ignoring the diagonal.
///
/// The iterator returns `None` if the row is empty.
///
/// # Examples
///
/// ```
/// use rpid::algorithms;
///
/// let matrix = vec![
///     vec![0, 9, 7],
///     vec![3, 0, 1],
///     vec![5, 4, 0],
/// ];
/// let result = algorithms::take_row_wise_min_without_diagonal(&matrix).collect::<Vec<_>>();
/// let expected = vec![Some(7), Some(1), Some(4)];
/// assert_eq!(result, expected);
/// ```
pub fn take_row_wise_min_without_diagonal<T>(
    matrix: &[Vec<T>],
) -> impl Iterator<Item = Option<T>> + '_
where
    T: Copy + PartialOrd,
{
    matrix.iter().enumerate().map(|(i, row)| {
        row.iter()
            .enumerate()
            .filter(|(j, _)| i != *j)
            .map(|(_, &w)| w)
            .reduce(|a, b| if a <= b { a } else { b })
    })
}

/// Returns an iterator taking the minimum of each row in a matrix.
///
/// The iterator returns `None` if the row is empty or all values in the row are `None`.
///
/// # Examples
///
/// ```
/// use rpid::algorithms;
///
/// let matrix = vec![
///     vec![None, Some(9), Some(7)],
///     vec![Some(3), None, Some(1)],
///     vec![None, None, None],
/// ];
/// let result = algorithms::take_row_wise_min_with_option(&matrix).collect::<Vec<_>>();
/// let expected = vec![Some(7), Some(1), None];
/// assert_eq!(result, expected);
/// ```
pub fn take_row_wise_min_with_option<T>(
    matrix: &[Vec<Option<T>>],
) -> impl Iterator<Item = Option<T>> + '_
where
    T: Copy + PartialOrd,
{
    matrix.iter().map(|row| {
        row.iter()
            .filter_map(|v| *v)
            .reduce(|a, b| if a <= b { a } else { b })
    })
}

/// Returns an iterator taking the minimum of each column in a matrix.
///
/// The iterator returns `None` if the column is empty.
///
/// # Examples
///
/// ```
/// use rpid::algorithms;
///
/// let matrix = vec![
///     vec![2, 9, 7],
///     vec![3, 6, 1],
///     vec![5, 4, 7],
/// ];
/// let result = algorithms::take_column_wise_min(&matrix).collect::<Vec<_>>();
/// let expected = vec![Some(2), Some(4), Some(1)];
/// assert_eq!(result, expected);
/// ```
pub fn take_column_wise_min<T>(matrix: &[Vec<T>]) -> impl Iterator<Item = Option<T>> + '_
where
    T: Copy + PartialOrd,
{
    (0..matrix.len()).map(|j| {
        matrix
            .iter()
            .filter_map(|row| row.get(j).copied())
            .reduce(|a, b| if a <= b { a } else { b })
    })
}

/// Returns an iterator taking the minimum of each column in a matrix, ignoring the diagonal.
///
/// The iterator returns `None` if the column is empty.
///
/// # Examples
///
/// ```
/// use rpid::algorithms;
///
/// let matrix = vec![
///     vec![0, 9, 7],
///     vec![3, 0, 1],
///     vec![5, 4, 0],
/// ];
/// let result = algorithms::take_column_wise_min_without_diagonal(&matrix).collect::<Vec<_>>();
/// let expected = vec![Some(3), Some(4), Some(1)];
/// assert_eq!(result, expected);
/// ```
pub fn take_column_wise_min_without_diagonal<T>(
    matrix: &[Vec<T>],
) -> impl Iterator<Item = Option<T>> + '_
where
    T: Copy + PartialOrd,
{
    (0..matrix.len()).map(|j| {
        matrix
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != j)
            .filter_map(|(_, row)| row.get(j).copied())
            .reduce(|a, b| if a <= b { a } else { b })
    })
}

/// Returns an iterator taking the minimum of each column in a matrix.
///
/// The iterator returns `None` if the row is empty or all values in the row are `None`.
///
/// # Examples
///
/// ```
/// use rpid::algorithms;
///
/// let matrix = vec![
///     vec![None, Some(9), Some(7)],
///     vec![Some(3), None, Some(1)],
///     vec![None, None, None],
/// ];
/// let result = algorithms::take_column_wise_min_with_option(&matrix).collect::<Vec<_>>();
/// let expected = vec![Some(3), Some(9), Some(1)];
/// assert_eq!(result, expected);
/// ```
pub fn take_column_wise_min_with_option<T>(
    matrix: &[Vec<Option<T>>],
) -> impl Iterator<Item = Option<T>> + '_
where
    T: Copy + PartialOrd,
{
    (0..matrix.len()).map(move |j| {
        matrix
            .iter()
            .filter_map(|row| row.get(j).and_then(|v| *v))
            .reduce(|a, b| if a <= b { a } else { b })
    })
}

/// Returns an iterator taking the maximum of each row in a matrix.
///
/// The iterator returns `None` if the row is empty.
///
/// # Examples
///
/// ```
/// use rpid::algorithms;
///
/// let matrix = vec![
///     vec![2, 9, 7],
///     vec![3, 6, 1],
///     vec![5, 4, 7],
/// ];
/// let result = algorithms::take_row_wise_max(&matrix).collect::<Vec<_>>();
/// let expected = vec![Some(9), Some(6), Some(7)];
/// assert_eq!(result, expected);
/// ```
pub fn take_row_wise_max<T>(matrix: &[Vec<T>]) -> impl Iterator<Item = Option<T>> + '_
where
    T: Copy + PartialOrd,
{
    matrix.iter().map(|row| {
        row.iter()
            .copied()
            .reduce(|a, b| if a >= b { a } else { b })
    })
}

/// Returns an iterator taking the maximum of each row in a matrix, ignoring the diagonal.
///
/// The iterator returns `None` if the row is empty.
///
/// # Examples
///
/// ```
/// use rpid::algorithms;
///
/// let matrix = vec![
///     vec![0, 9, 7],
///     vec![3, 0, 1],
///     vec![5, 4, 0],
/// ];
/// let result = algorithms::take_row_wise_max_without_diagonal(&matrix).collect::<Vec<_>>();
/// let expected = vec![Some(9), Some(3), Some(5)];
/// assert_eq!(result, expected);
/// ```
pub fn take_row_wise_max_without_diagonal<T>(
    matrix: &[Vec<T>],
) -> impl Iterator<Item = Option<T>> + '_
where
    T: Copy + PartialOrd,
{
    matrix.iter().enumerate().map(|(i, row)| {
        row.iter()
            .enumerate()
            .filter(|(j, _)| i != *j)
            .map(|(_, &w)| w)
            .reduce(|a, b| if a >= b { a } else { b })
    })
}

/// Returns an iterator taking the maximum of each row in a matrix.
///
/// The iterator returns `None` if the row is empty or all values in the row are `None`.
///
/// # Examples
///
/// ```
/// use rpid::algorithms;
///
/// let matrix = vec![
///     vec![None, Some(9), Some(7)],
///     vec![Some(3), None, Some(1)],
///     vec![None, None, None],
/// ];
/// let result = algorithms::take_row_wise_max_with_option(&matrix).collect::<Vec<_>>();
/// let expected = vec![Some(9), Some(3), None];
/// assert_eq!(result, expected);
/// ```
pub fn take_row_wise_max_with_option<T>(
    matrix: &[Vec<Option<T>>],
) -> impl Iterator<Item = Option<T>> + '_
where
    T: Copy + PartialOrd,
{
    matrix.iter().map(|row| {
        row.iter()
            .filter_map(|v| *v)
            .reduce(|a, b| if a >= b { a } else { b })
    })
}

/// Returns an iterator taking the maximum of each column in a matrix.
///
/// The iterator returns `None` if the column is empty.
///
/// # Examples
///
/// ```
/// use rpid::algorithms;
///
/// let matrix = vec![
///     vec![2, 9, 7],
///     vec![3, 6, 1],
///     vec![5, 4, 7],
/// ];
/// let result = algorithms::take_column_wise_max(&matrix).collect::<Vec<_>>();
/// let expected = vec![Some(5), Some(9), Some(7)];
/// assert_eq!(result, expected);
/// ```
pub fn take_column_wise_max<T>(matrix: &[Vec<T>]) -> impl Iterator<Item = Option<T>> + '_
where
    T: Copy + PartialOrd,
{
    (0..matrix.len()).map(|j| {
        matrix
            .iter()
            .filter_map(|row| row.get(j).copied())
            .reduce(|a, b| if a >= b { a } else { b })
    })
}

/// Returns an iterator taking the maximum of each column in a matrix, ignoring the diagonal.
///
/// The iterator returns `None` if the column is empty.
///
/// # Examples
///
/// ```
/// use rpid::algorithms;
///
/// let matrix = vec![
///     vec![0, 9, 7],
///     vec![3, 0, 1],
///     vec![5, 4, 0],
/// ];
/// let result = algorithms::take_column_wise_max_without_diagonal(&matrix).collect::<Vec<_>>();
/// let expected = vec![Some(5), Some(9), Some(7)];
/// assert_eq!(result, expected);
/// ```
pub fn take_column_wise_max_without_diagonal<T>(
    matrix: &[Vec<T>],
) -> impl Iterator<Item = Option<T>> + '_
where
    T: Copy + PartialOrd,
{
    (0..matrix.len()).map(|j| {
        matrix
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != j)
            .filter_map(|(_, row)| row.get(j).copied())
            .reduce(|a, b| if a >= b { a } else { b })
    })
}

/// Returns an iterator taking the maximum of each column in a matrix.
///
/// The iterator returns `None` if the row is empty or all values in the row are `None`.
///
/// # Examples
///
/// ```
/// use rpid::algorithms;
///
/// let matrix = vec![
///     vec![None, Some(9), Some(7)],
///     vec![Some(3), None, Some(1)],
///     vec![None, None, None],
/// ];
/// let result = algorithms::take_column_wise_max_with_option(&matrix).collect::<Vec<_>>();
/// let expected = vec![Some(3), Some(9), Some(7)];
/// assert_eq!(result, expected);
/// ```
pub fn take_column_wise_max_with_option<T>(
    matrix: &[Vec<Option<T>>],
) -> impl Iterator<Item = Option<T>> + '_
where
    T: Copy + PartialOrd,
{
    (0..matrix.len()).map(move |j| {
        matrix
            .iter()
            .filter_map(|row| row.get(j).and_then(|v| *v))
            .reduce(|a, b| if a >= b { a } else { b })
    })
}

fn total_cmp<T: PartialOrd>(a: &T, b: &T) -> Ordering {
    if a < b {
        Ordering::Less
    } else if a > b {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

/// Sorts an weight matrix in a ascending order.
///
/// The diagonal of the matrix is ignored.
///
/// # Examples
///
/// ```
/// use rpid::algorithms;
///
/// let matrix = vec![
///     vec![10, 9, 7],
///     vec![3, 11, 8],
///     vec![5, 1, 12],
/// ];
/// let expected = vec![
///     (2, 1, 1),
///     (1, 0, 3),
///     (2, 0, 5),
///     (0, 2, 7),
///     (1, 2, 8),
///     (0, 1, 9),
///     (0, 0, 10),
///     (1, 1, 11),
///     (2, 2, 12),
/// ];
/// assert_eq!(algorithms::sort_weight_matrix(&matrix), expected);
/// ```
pub fn sort_weight_matrix<T>(matrix: &[Vec<T>]) -> Vec<(usize, usize, T)>
where
    T: Copy + PartialOrd,
{
    let n = matrix.len();
    let mut edges = Vec::with_capacity(n * n);

    for (i, row) in matrix.iter().enumerate() {
        for (j, &w) in row.iter().enumerate() {
            edges.push((i, j, w));
        }
    }

    edges.sort_by(|a, b| total_cmp(&a.2, &b.2));

    edges
}

/// Sorts an weight matrix in a ascending order.
///
/// The diagonal of the matrix is ignored.
///
/// # Examples
///
/// ```
/// use rpid::algorithms;
///
/// let matrix = vec![
///     vec![0, 9, 7],
///     vec![3, 0, 8],
///     vec![5, 1, 0],
/// ];
/// let expected = vec![
///     (2, 1, 1),
///     (1, 0, 3),
///     (2, 0, 5),
///     (0, 2, 7),
///     (1, 2, 8),
///     (0, 1, 9),
/// ];
/// assert_eq!(algorithms::sort_weight_matrix_without_diagonal(&matrix), expected);
/// ```
pub fn sort_weight_matrix_without_diagonal<T>(matrix: &[Vec<T>]) -> Vec<(usize, usize, T)>
where
    T: Copy + PartialOrd,
{
    let n = matrix.len();
    let mut edges = Vec::with_capacity(n * n);

    for (i, row) in matrix.iter().enumerate() {
        for (j, &w) in row.iter().enumerate() {
            if i != j {
                edges.push((i, j, w));
            }
        }
    }

    edges.sort_by(|a, b| total_cmp(&a.2, &b.2));

    edges
}

/// Sorts an weight matrix in a ascending order.
///
/// The input `matrix` should be a square matrix with the weights of the edges.
/// If there is no edge between two nodes, the corresponding entry should be `None`.
///
/// # Examples
///
/// ```
/// use rpid::algorithms;
///
/// let matrix = vec![
///     vec![None, Some(9), Some(7)],
///     vec![Some(3), None, None],
///     vec![None, Some(1), None],
/// ];
/// let expected = vec![
///     (2, 1, 1),
///     (1, 0, 3),
///     (0, 2, 7),
///     (0, 1, 9),
/// ];
/// assert_eq!(algorithms::sort_weight_matrix_with_option(&matrix), expected);
/// ```
pub fn sort_weight_matrix_with_option<T>(matrix: &[Vec<Option<T>>]) -> Vec<(usize, usize, T)>
where
    T: Copy + PartialOrd,
{
    let n = matrix.len();
    let mut weights = Vec::with_capacity(n * n);

    for (i, row) in matrix.iter().enumerate() {
        for (j, &w) in row.iter().enumerate() {
            if let Some(w) = w {
                weights.push((i, j, w));
            }
        }
    }

    weights.sort_by(|a, b| total_cmp(&a.2, &b.2));

    weights
}

/// Union find tree data structure.
pub struct UnionFindTree {
    parent: Vec<usize>,
}

impl UnionFindTree {
    /// Creates a new union find tree with `n` nodes.
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    /// Finds the root of the tree containing the node `x`.
    pub fn find(&self, mut x: usize) -> usize {
        while self.parent[x] != x {
            x = self.parent[x];
        }

        x
    }

    /// Unites the trees containing the nodes `x` and `y`.
    pub fn union(&mut self, x: usize, y: usize) {
        let x_root = self.find(x);
        let y_root = self.find(y);

        self.parent[x_root] = y_root;
    }

    /// Checks if the nodes `x` and `y` are in the same tree.
    pub fn is_same(&self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }
}

/// Computes the weight of the minimum spanning tree of a graph.
///
/// `maximum_index` is the maximum index of the nodes in the graph.
///
/// `n_nodes` is the number of nodes in the graph.
///
/// The input `sorted_edges` should be an iterator over the edges of the graph,
/// sorted by their weights (the third element) in a ascending order.
///
/// # Examples
///
/// ```
/// use rpid::algorithms;
///
/// let matrix = vec![
///     vec![0, 1, 2],
///     vec![1, 0, 3],
///     vec![2, 3, 0]
/// ];
/// let sorted_edges = algorithms::sort_weight_matrix(&matrix);
/// let mst_weight
///     = algorithms::compute_minimum_spanning_tree_weight(2, 3, sorted_edges.iter().copied());
/// assert_eq!(mst_weight, 3);
/// ```
pub fn compute_minimum_spanning_tree_weight<T>(
    maximum_index: usize,
    n_nodes: usize,
    sorted_edges: impl Iterator<Item = (usize, usize, T)>,
) -> T
where
    T: PartialOrd + Add<Output = T> + Zero,
{
    if n_nodes <= 1 {
        return T::zero();
    }

    let mut tree = UnionFindTree::new(maximum_index + 1);
    let mut weight = T::zero();
    let mut n_added = 0;

    for (u, v, w) in sorted_edges {
        if !tree.is_same(u, v) {
            tree.union(u, v);
            weight = weight + w;
            n_added += 1;

            if n_added == n_nodes - 1 {
                break;
            }
        }
    }

    weight
}

/// Sort the items of the knapsack problem by their efficiency in a descending order.
///
/// The efficiency of an item is the value of the item divided by the weight of the item.
///
/// # Examples
///
/// ```
/// use rpid::algorithms;
///
/// let weights = [4, 1, 2];
/// let values = [5, 2, 3];
///
/// let sorted_weight_value_pairs = algorithms::sort_knapsack_items_by_efficiency(&weights, &values);
/// let expected = vec![(1, 1, 2), (2, 2, 3), (0, 4, 5)];
/// assert_eq!(sorted_weight_value_pairs, expected);
/// ```
pub fn sort_knapsack_items_by_efficiency<T, U>(weights: &[T], values: &[U]) -> Vec<(usize, T, U)>
where
    T: Copy,
    U: Copy,
    f64: From<T> + From<U>,
{
    let mut sorted_weight_value_pairs = weights
        .iter()
        .copied()
        .zip(values.iter().copied())
        .enumerate()
        .map(|(i, (w, v))| (i, w, v))
        .collect::<Vec<_>>();
    sorted_weight_value_pairs.sort_by(|&a, &b| {
        (f64::from(a.1) * f64::from(b.2)).total_cmp(&(f64::from(a.2) * f64::from(b.1)))
    });

    sorted_weight_value_pairs
}

/// Computes the profit of the knapsack problem, allowing fractions of items to be taken.
///
/// This bound is called Danzig's upper bound.
///
/// `capacity` is the maximum weight that the knapsack can carry.
///
/// `sorted_weight_value_pairs` is an iterator over the pairs of weight and value of the items,
/// sorted by the value per weight in a descending order.
/// The first element of the pair is the weight of the item and the second element is the value.
///
/// # Examples
///
/// ```
/// use rpid::algorithms;
///
/// let weights = [1, 2, 4];
/// let values = [2, 3, 5];
///
/// let sorted_items = algorithms::sort_knapsack_items_by_efficiency(&weights, &values);
/// let sorted_weight_value_pairs = sorted_items.iter().map(|&(_, w, v)| (w, v));
/// let profit = algorithms::compute_fractional_knapsack_profit(5, sorted_weight_value_pairs, 1e-6);
/// assert_eq!(profit, 7.0);
/// ```
pub fn compute_fractional_knapsack_profit<T>(
    capacity: T,
    sorted_weight_value_pairs: impl Iterator<Item = (T, T)>,
    epsilon: f64,
) -> f64
where
    T: PartialOrd + SubAssign + Copy,
    f64: From<T>,
{
    let mut profit = 0.0;
    let mut remaining_capacity = capacity;

    for (weight, value) in sorted_weight_value_pairs {
        if remaining_capacity >= weight {
            profit += f64::from(value);
            remaining_capacity -= weight;
        } else {
            profit += f64::from(remaining_capacity) * (f64::from(value) / f64::from(weight));
            break;
        }
    }

    f64::floor(profit + epsilon)
}

/// Computes the number of bins to pack weighted items, allowing fractions of items to be taken.
///
/// This bound is sometimes called LB1.
///
/// `epsilon` is the upper bound on the difference between two values to be considered equal.
/// If the cost of the fractional bin packing is `c`, the number of bins is `ceil(c - epsilon)`.
///
/// # Examples
///
/// ```
/// use rpid::algorithms;
///
/// let weights = [2, 2, 3, 4, 5];
///
/// let n_bins = algorithms::compute_fractional_bin_packing_cost(5, weights.iter().sum(), 0);
/// assert_eq!(n_bins, 4.0);
/// ```
pub fn compute_fractional_bin_packing_cost<T>(capacity: T, weight_sum: T, epsilon: T) -> f64
where
    T: Sum<T> + Rem<Output = T> + Div<Output = T> + PartialOrd + Copy + One,
    f64: From<T>,
{
    if capacity <= epsilon {
        return f64::INFINITY;
    }

    if weight_sum % capacity <= epsilon {
        f64::from(weight_sum / capacity).trunc()
    } else {
        f64::from(weight_sum / capacity).trunc() + 1.0
    }
}

/// Computes the number of bins to pack items, whose weights are at least half of the capacity.
///
/// This bound is sometimes called LB2.
///
/// `epsilon` is the upper bound on the difference between two values to be considered equal.
///
/// # Examples
///
/// ```
/// use rpid::algorithms;
///
/// let weights = [4, 2, 3, 5, 4, 3, 3];
///
/// let n_bins = algorithms::compute_bin_packing_lb2(6, weights.iter().copied(), 0);
/// assert_eq!(n_bins, 5.0);
/// ```
pub fn compute_bin_packing_lb2<T>(capacity: T, weights: impl Iterator<Item = T>, epsilon: T) -> f64
where
    T: PartialOrd + Add<Output = T> + Copy,
{
    if capacity <= epsilon {
        return f64::INFINITY;
    }

    let mut n_more_than_half = 0;
    let mut n_half = 0;

    weights.for_each(|w| {
        let twice = w + w;

        if twice > capacity + epsilon {
            n_more_than_half += 1;
        } else if twice + epsilon >= capacity {
            n_half += 1;
        }
    });

    if n_half % 2 == 0 {
        (n_more_than_half + n_half / 2) as f64
    } else {
        (n_more_than_half + n_half / 2 + 1) as f64
    }
}

/// Computes the number of bins to pack items, whose weights are at least one third of the capacity.
///
/// This bound is sometimes called LB3.
///
/// `epsilon` is the upper bound on the difference between two values to be considered equal.
///
/// # Examples
///
/// ```
/// use rpid::algorithms;
///
/// let weights = [4, 2, 3, 5, 4, 3, 3];
///
/// let n_bins = algorithms::compute_bin_packing_lb3(6, weights.iter().copied(), 0);
/// assert_eq!(n_bins, 5.0);
/// ```
pub fn compute_bin_packing_lb3<T>(capacity: T, weights: impl Iterator<Item = T>, epsilon: T) -> f64
where
    T: PartialOrd + Add<Output = T> + Copy,
{
    if capacity <= epsilon {
        return f64::INFINITY;
    }

    let twice_capacity = capacity + capacity;

    let weight_sum = weights
        .map(|w| {
            let thrice = w + w + w;

            if thrice > twice_capacity + epsilon {
                6
            } else if thrice + epsilon >= twice_capacity {
                4
            } else if thrice > capacity + epsilon {
                3
            } else if thrice + epsilon >= capacity {
                2
            } else {
                0
            }
        })
        .sum::<i32>();

    if weight_sum % 6 == 0 {
        (weight_sum / 6) as f64
    } else {
        (weight_sum / 6 + 1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_transpose() {
        let matrix = vec![vec![1, 2, 3, 4], vec![5, 6], vec![7]];
        let transposed = transpose(&matrix);
        assert_eq!(
            transposed,
            vec![vec![1, 5, 7], vec![2, 6], vec![3], vec![4]]
        )
    }

    #[test]
    fn test_compute_pairwise_euclidean_distances() {
        let points = [(0, 0), (3, 0), (3, 4)];
        let expected = [[0.0, 3.0, 5.0], [3.0, 0.0, 4.0], [5.0, 4.0, 0.0]];
        let distances = compute_pairwise_euclidean_distances(&points);
        assert_relative_eq!(distances[0][0], expected[0][0]);
        assert_relative_eq!(distances[0][1], expected[0][1]);
        assert_relative_eq!(distances[0][2], expected[0][2]);
        assert_relative_eq!(distances[1][0], expected[1][0]);
        assert_relative_eq!(distances[1][1], expected[1][1]);
        assert_relative_eq!(distances[1][2], expected[1][2]);
        assert_relative_eq!(distances[2][0], expected[2][0]);
        assert_relative_eq!(distances[2][1], expected[2][1]);
        assert_relative_eq!(distances[2][2], expected[2][2]);
    }

    #[test]
    fn test_compute_pairwise_shortest_path_costs() {
        let weights = vec![
            vec![0, 1, 2, 3],
            vec![1, 0, 4, 5],
            vec![2, 4, 0, 6],
            vec![3, 5, 6, 0],
        ];
        let expected = vec![
            vec![0, 1, 2, 3],
            vec![1, 0, 3, 4],
            vec![2, 3, 0, 5],
            vec![3, 4, 5, 0],
        ];
        assert_eq!(compute_pairwise_shortest_path_costs(&weights), expected);
    }

    #[test]
    fn test_compute_pairwise_shortest_path_costs_with_option() {
        let weights = vec![
            vec![None, Some(1), None, Some(3)],
            vec![Some(1), None, Some(4), Some(5)],
            vec![Some(2), Some(4), None, Some(6)],
            vec![None, None, None, None],
        ];
        let expected = vec![
            vec![None, Some(1), Some(5), Some(3)],
            vec![Some(1), None, Some(4), Some(4)],
            vec![Some(2), Some(3), None, Some(5)],
            vec![None, None, None, None],
        ];
        assert_eq!(
            compute_pairwise_shortest_path_costs_with_option(&weights),
            expected
        );
    }

    #[test]
    fn test_take_row_wise_min() {
        let matrix = vec![vec![2, 9, 7], vec![3, 6, 1], vec![5, 4, 7]];
        let result = take_row_wise_min(&matrix).collect::<Vec<_>>();
        let expected = vec![Some(2), Some(1), Some(4)];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_take_row_wise_min_without_diagonal() {
        let matrix = vec![vec![0, 9, 7], vec![3, 0, 1], vec![5, 4, 0]];
        let result = take_row_wise_min_without_diagonal(&matrix).collect::<Vec<_>>();
        let expected = vec![Some(7), Some(1), Some(4)];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_take_row_wise_min_with_option() {
        let matrix = vec![
            vec![None, Some(9), Some(7)],
            vec![Some(3), None, Some(1)],
            vec![None, None, None],
        ];
        let result = take_row_wise_min_with_option(&matrix).collect::<Vec<_>>();
        let expected = vec![Some(7), Some(1), None];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_take_column_wise_min() {
        let matrix = vec![vec![2, 9, 7], vec![3, 6, 1], vec![5, 4, 7]];
        let result = take_column_wise_min(&matrix).collect::<Vec<_>>();
        let expected = vec![Some(2), Some(4), Some(1)];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_take_column_wise_min_without_diagonal() {
        let matrix = vec![vec![0, 9, 7], vec![3, 0, 1], vec![5, 4, 0]];
        let result = take_column_wise_min_without_diagonal(&matrix).collect::<Vec<_>>();
        let expected = vec![Some(3), Some(4), Some(1)];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_take_column_wise_min_with_option() {
        let matrix = vec![
            vec![None, Some(9), Some(7)],
            vec![Some(3), None, Some(1)],
            vec![None, None, None],
        ];
        let result = take_column_wise_min_with_option(&matrix).collect::<Vec<_>>();
        let expected = vec![Some(3), Some(9), Some(1)];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_take_row_wise_max() {
        let matrix = vec![vec![2, 9, 7], vec![3, 6, 1], vec![5, 4, 7]];
        let result = take_row_wise_max(&matrix).collect::<Vec<_>>();
        let expected = vec![Some(9), Some(6), Some(7)];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_take_row_wise_max_without_diagonal() {
        let matrix = vec![vec![0, 9, 7], vec![3, 0, 1], vec![5, 4, 0]];
        let result = take_row_wise_max_without_diagonal(&matrix).collect::<Vec<_>>();
        let expected = vec![Some(9), Some(3), Some(5)];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_take_row_wise_max_with_option() {
        let matrix = vec![
            vec![None, Some(9), Some(7)],
            vec![Some(3), None, Some(1)],
            vec![None, None, None],
        ];
        let result = take_row_wise_max_with_option(&matrix).collect::<Vec<_>>();
        let expected = vec![Some(9), Some(3), None];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_take_column_wise_max() {
        let matrix = vec![vec![2, 9, 7], vec![3, 6, 1], vec![5, 4, 7]];
        let result = take_column_wise_max(&matrix).collect::<Vec<_>>();
        let expected = vec![Some(5), Some(9), Some(7)];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_take_column_wise_max_without_diagonal() {
        let matrix = vec![vec![0, 9, 7], vec![3, 0, 1], vec![5, 4, 0]];
        let result = take_column_wise_max_without_diagonal(&matrix).collect::<Vec<_>>();
        let expected = vec![Some(5), Some(9), Some(7)];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_take_column_wise_max_with_option() {
        let matrix = vec![
            vec![None, Some(9), Some(7)],
            vec![Some(3), None, Some(1)],
            vec![None, None, None],
        ];
        let result = take_column_wise_max_with_option(&matrix).collect::<Vec<_>>();
        let expected = vec![Some(3), Some(9), Some(7)];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_sort_weighted_matrix() {
        let matrix = vec![
            vec![13, 9, 11, 7],
            vec![3, 14, 8, 10],
            vec![12, 5, 15, 1],
            vec![6, 2, 4, 16],
        ];
        let expected = vec![
            (2, 3, 1),
            (3, 1, 2),
            (1, 0, 3),
            (3, 2, 4),
            (2, 1, 5),
            (3, 0, 6),
            (0, 3, 7),
            (1, 2, 8),
            (0, 1, 9),
            (1, 3, 10),
            (0, 2, 11),
            (2, 0, 12),
            (0, 0, 13),
            (1, 1, 14),
            (2, 2, 15),
            (3, 3, 16),
        ];
        assert_eq!(sort_weight_matrix(&matrix), expected);
    }

    #[test]
    fn test_sort_weighted_matrix_without_diagonal() {
        let matrix = vec![
            vec![0, 9, 11, 7],
            vec![3, 0, 8, 10],
            vec![12, 5, 0, 1],
            vec![6, 2, 4, 0],
        ];
        let expected = vec![
            (2, 3, 1),
            (3, 1, 2),
            (1, 0, 3),
            (3, 2, 4),
            (2, 1, 5),
            (3, 0, 6),
            (0, 3, 7),
            (1, 2, 8),
            (0, 1, 9),
            (1, 3, 10),
            (0, 2, 11),
            (2, 0, 12),
        ];
        assert_eq!(sort_weight_matrix_without_diagonal(&matrix), expected);
    }

    #[test]
    fn test_sort_weighted_matrix_with_option() {
        let matrix = vec![
            vec![None, Some(9), None, Some(7)],
            vec![None, None, Some(8), Some(10)],
            vec![Some(12), Some(5), None, Some(1)],
            vec![Some(6), None, Some(4), None],
        ];
        let expected = vec![
            (2, 3, 1),
            (3, 2, 4),
            (2, 1, 5),
            (3, 0, 6),
            (0, 3, 7),
            (1, 2, 8),
            (0, 1, 9),
            (1, 3, 10),
            (2, 0, 12),
        ];
        assert_eq!(sort_weight_matrix_with_option(&matrix), expected);
    }

    #[test]
    fn test_union_find_tree() {
        let mut tree = UnionFindTree::new(4);
        assert_eq!(tree.find(0), 0);
        assert_eq!(tree.find(1), 1);
        assert_eq!(tree.find(2), 2);
        assert_eq!(tree.find(3), 3);
        assert!(!tree.is_same(0, 1));
        assert!(!tree.is_same(0, 2));
        assert!(!tree.is_same(0, 3));
        assert!(!tree.is_same(1, 2));
        assert!(!tree.is_same(1, 3));
        assert!(!tree.is_same(2, 3));

        tree.union(1, 3);
        assert_eq!(tree.find(0), 0);
        assert_eq!(tree.find(1), tree.find(3));
        assert_eq!(tree.find(2), 2);
        assert!(!tree.is_same(0, 1));
        assert!(!tree.is_same(0, 2));
        assert!(!tree.is_same(0, 3));
        assert!(!tree.is_same(1, 2));
        assert!(tree.is_same(1, 3));
        assert!(!tree.is_same(2, 3));

        tree.union(1, 2);
        assert_eq!(tree.find(0), 0);
        assert_eq!(tree.find(1), tree.find(3));
        assert_eq!(tree.find(1), tree.find(2));
        assert!(!tree.is_same(0, 1));
        assert!(!tree.is_same(0, 2));
        assert!(!tree.is_same(0, 3));
        assert!(tree.is_same(1, 2));
        assert!(tree.is_same(1, 3));
        assert!(tree.is_same(2, 3));

        tree.union(3, 0);
        assert_eq!(tree.find(0), tree.find(1));
        assert_eq!(tree.find(1), tree.find(2));
        assert_eq!(tree.find(1), tree.find(3));
        assert!(tree.is_same(0, 1));
        assert!(tree.is_same(0, 2));
        assert!(tree.is_same(0, 3));
        assert!(tree.is_same(1, 2));
        assert!(tree.is_same(1, 3));
        assert!(tree.is_same(2, 3));
    }

    #[test]
    fn test_compute_minimum_spanning_tree_weight() {
        let sorted_edges = [
            (0, 1, 1),
            (1, 2, 2),
            (2, 5, 3),
            (0, 2, 4),
            (1, 5, 5),
            (0, 5, 6),
        ];
        assert_eq!(
            compute_minimum_spanning_tree_weight(5, 4, sorted_edges.into_iter()),
            6
        );
    }

    #[test]
    fn test_compute_minimum_spanning_tree_weight_empty() {
        let sorted_edges: [(_, _, i32); 0] = [];
        assert_eq!(
            compute_minimum_spanning_tree_weight(0, 0, sorted_edges.into_iter()),
            0
        );
    }

    #[test]
    fn test_sort_knapsack_items_by_efficiency() {
        let weights = [4, 1, 2];
        let values = [5, 2, 3];
        let sorted_items = sort_knapsack_items_by_efficiency(&weights, &values);
        let expected = vec![(1, 1, 2), (2, 2, 3), (0, 4, 5)];
        assert_eq!(sorted_items, expected);
    }

    #[test]
    fn test_compute_fractional_knapsack_profit() {
        let sorted_weight_value_pairs = [(1, 2), (2, 3), (4, 5), (1, 1)];
        assert_relative_eq!(
            compute_fractional_knapsack_profit(5, sorted_weight_value_pairs.into_iter(), 1e-6),
            7.0
        );
    }

    #[test]
    fn test_compute_fractional_knapsack_profit_empty() {
        let sorted_weight_value_pairs: [(i32, i32); 0] = [];
        assert_relative_eq!(
            compute_fractional_knapsack_profit(0, sorted_weight_value_pairs.into_iter(), 1e-6),
            0.0
        );
    }

    #[test]
    fn test_compute_fractional_bin_packing_cost() {
        let weights = [2, 2, 3, 4, 5];
        assert_relative_eq!(
            compute_fractional_bin_packing_cost(5, weights.iter().sum(), 0),
            4.0
        );
    }

    #[test]
    fn test_compute_fractional_bin_packing_cost_empty() {
        assert_relative_eq!(compute_fractional_bin_packing_cost(5, 0, 0), 0.0);
    }

    #[test]
    fn test_compute_fractional_bin_packing_cost_infinity() {
        let weights = [2, 2, 3, 4, 5];
        assert!(compute_fractional_bin_packing_cost(0, weights.iter().sum(), 0).is_infinite());
    }

    #[test]
    fn test_compute_bin_packing_lb2() {
        let weights = [4, 2, 3, 5, 4, 3, 3];
        assert_relative_eq!(compute_bin_packing_lb2(6, weights.into_iter(), 0), 5.0);
    }

    #[test]
    fn test_compute_bin_packing_lb2_empty() {
        let weights: [i32; 0] = [];
        assert_relative_eq!(compute_bin_packing_lb2(6, weights.into_iter(), 0), 0.0);
    }

    #[test]
    fn test_compute_bin_packing_lb2_infinity() {
        let weights = [4, 2, 3, 5, 4, 3, 3];
        assert!(compute_bin_packing_lb2(0, weights.into_iter(), 0).is_infinite());
    }

    #[test]
    fn test_compute_bin_packing_lb3() {
        let weights = [4, 2, 3, 5, 4, 3, 3];
        assert_relative_eq!(compute_bin_packing_lb3(6, weights.into_iter(), 0), 5.0);
    }

    #[test]
    fn test_compute_bin_packing_lb3_empty() {
        let weights: [i32; 0] = [];
        assert_relative_eq!(compute_bin_packing_lb3(6, weights.into_iter(), 0), 0.0);
    }

    #[test]
    fn test_compute_bin_packing_lb3_infinity() {
        let weights = [4, 2, 3, 5, 4, 3, 3];
        assert!(compute_bin_packing_lb3(0, weights.into_iter(), 0).is_infinite());
    }
}
