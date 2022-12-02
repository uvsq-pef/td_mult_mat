use std::ops::Index;
use std::ops::IndexMut;

type Element = f64;

#[derive(Debug, PartialEq)]
pub struct Matrix {
    n: usize,
    values: Vec<Element>,
}

impl Matrix {
    pub fn new(n: usize, values: Vec<Element>) -> Self {
        assert_eq!(n * n, values.len());

        Matrix {
            n: n,
            values: values,
        }
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = Element;
    fn index(&self, index: (usize, usize)) -> &Element {
        &self.values[self.n * index.0 + index.1]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Element {
        &mut self.values[self.n * index.0 + index.1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_should_create_a_matrix() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let m = Matrix::new(2, values.clone());
        assert_eq!(values, m.values);
    }

    #[test]
    fn should_have_access_by_indices() {
        let mut m = Matrix::new(2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 0)], 3.0);

        m[(1,0)] = 5.0;
        assert_eq!(m[(1, 0)], 5.0);
    }
}
