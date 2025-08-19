use std::rc::Rc;

/// Tree data structure to store a sequence of ids.
#[derive(Clone, Debug, Default)]
pub struct IdTree<T> {
    id: Option<T>,
    parent: Option<Rc<Self>>,
}

impl<T> IdTree<T>
where
    T: Copy,
{
    /// Creates a child node.
    pub fn create_child(node: Rc<Self>, id: T) -> Self {
        Self {
            id: Some(id),
            parent: Some(node.clone()),
        }
    }

    /// Returns the path from the root to the current node.
    pub fn get_path(&self) -> Vec<T> {
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
        let node = Rc::new(IdTree::default());
        let node = Rc::new(IdTree::create_child(node.clone(), 1));
        let node = Rc::new(IdTree::create_child(node.clone(), 2));
        let node = Rc::new(IdTree::create_child(node.clone(), 3));

        assert_eq!(node.get_path(), vec![1, 2, 3]);
    }
}
