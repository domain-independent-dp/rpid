use std::sync::Arc;
use crate::solvers::search_algorithms::Sequence;

/// Tree data structure to store a sequence of ids.
#[derive(Clone, Debug, Default)]
pub struct ArcIdTree<T> {
    id: Option<T>,
    parent: Option<Arc<Self>>,
}

impl<L> Sequence<L, Arc<ArcIdTree<L>>> for ArcIdTree<L>
where
    L: Copy,
{
    /// Creates a child node.
    fn create_child(node: Arc<Self>, id: L) -> Self {
        Self {
            id: Some(id),
            parent: Some(node.clone()),
        }
    }

    /// Returns the path from the root to the current node.
    fn get_path(&self) -> Vec<L> {
        let mut path = Vec::new();
        let mut current = self;

        while let Some(id) = current.id {
            path.push(id);
            current = current.parent.as_ref().unwrap();
        }

        path.reverse();

        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_id_tree() {
        let node = Arc::new(ArcIdTree::default());
        let node = Arc::new(ArcIdTree::create_child(node.clone(), 1));
        let node = Arc::new(ArcIdTree::create_child(node.clone(), 2));
        let node = Arc::new(ArcIdTree::create_child(node.clone(), 3));

        assert_eq!(node.get_path(), vec![1, 2, 3]);
    }
}
