use std::ops::Index;
use std::ops::IndexMut;
use rand::Rng;

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

    pub fn zero(n: usize) -> Self {
        Self::new(n, vec![Element::default(); n * n])
    }

    pub fn id(n: usize) -> Self {
        let mut m = Self::zero(n);
        for i in 0..n {
            m[(i, i)] = 1.0;
        }
        m
    }

    pub fn random(n: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self::new(n, (0..n * n).map(|_| rng.gen_range(-1.0..1.0)).collect())
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
    fn zero_should_create_a_matrix_filled_with_zero() {
        let values = vec![0.0, 0.0, 0.0, 0.0];
        let zero = Matrix::zero(2);
        assert_eq!(values, zero.values);
    }

    #[test]
    fn id_should_create_an_identity_matrix() {
        let values = vec![1.0, 0.0, 0.0, 1.0];
        let identity = Matrix::id(2);
        assert_eq!(values, identity.values);
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
