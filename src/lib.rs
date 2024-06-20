use rand::Rng;
use rayon::prelude::*;
use std::ops::{Index, IndexMut};
type Element = f64;

#[derive(Debug, PartialEq)]
pub struct Matrix {
    n: usize,
    values: Vec<Element>,
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

impl Matrix {
    const BLOCK: usize = 64;

    pub fn new(n: usize, values: Vec<Element>) -> Self {
        assert_eq!(n * n, values.len());
        Matrix {
            n: n,
            values: values,
        }
    }

    pub fn number_of_blocks(&self) -> usize {
        assert_eq!(
            self.n % Matrix::BLOCK,
            0,
            "matrix size must be a multiple of the block size"
        );

        self.n / Matrix::BLOCK
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

pub fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
    assert!(a.n == b.n);
    let n = a.n;
    let mut c = Matrix::zero(n);

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                c[(i, j)] += a[(i, k)] * b[(k, j)];
            }
        }
    }
    c
}

pub fn multiply_vectorized(a: &Matrix, b: &Matrix) -> Matrix {
    assert!(a.n == b.n);
    let n = a.n;
    let mut c = Matrix::zero(n);

    for i in 0..n {
        for k in 0..n {
            for j in 0..n {
                c[(i, j)] += a[(i, k)] * b[(k, j)];
            }
        }
    }
    c
}

pub fn multiply_blocked(a: &Matrix, b: &Matrix) -> Matrix {
    assert!(a.n == b.n);
    let n = a.n;
    let s = Matrix::BLOCK;
    let mut c = Matrix::zero(n);

    let nn = a.number_of_blocks();
    for ii in 0..nn {
        for kk in 0..nn {
            for jj in 0..nn {
                for i in (ii * s)..(ii + 1) * s {
                    for k in (kk * s)..(kk + 1) * s {
                        for j in (jj * s)..(jj + 1) * s {
                            c[(i, j)] += a[(i, k)] * b[(k, j)];
                        }
                    }
                }
            }
        }
    }
    c
}

fn multiply_block(n: usize, nn: usize, a: &[Element], b: &[Element], c: &mut [Element]) {
    let s = Matrix::BLOCK;
    for kk in 0..nn {
        for jj in 0..nn {
            for i in 0..s {
                for k in (kk * s)..(kk + 1) * s {
                    for j in (jj * s)..(jj + 1) * s {
                        c[i * n + j] += a[i * n + k] * b[k * n + j];
                    }
                }
            }
        }
    }
}

pub fn multiply_rayon(a: &Matrix, b: &Matrix) -> Matrix {
    assert!(a.n == b.n);
    let n = a.n;
    let mut c = Matrix::zero(n);

    let cc = c.values.par_chunks_mut(n * Matrix::BLOCK);
    let aa = a.values.par_chunks(n * Matrix::BLOCK);

    cc.zip(aa)
        .for_each(|(cv, av)| multiply_block(n, a.number_of_blocks(), av, &b.values, cv));
    c
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
    fn indexes() {
        let mut m = Matrix::new(2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 0)], 3.0);

        m[(1, 0)] = 5.0;
        assert_eq!(m[(1, 0)], 5.0);
    }

    #[test]
    fn should_naively_multiply_two_matrices() {
        const SIZE: usize = 2;

        let m1 = Matrix::new(SIZE, vec![1.0, 2.0, 3.0, 4.0]);
        let m2 = Matrix::new(SIZE, vec![4.0, 3.0, 2.0, 1.0]);
        let m_result = Matrix::new(SIZE, vec![8.0, 5.0, 20.0, 13.0]);
        assert_eq!(multiply(&m1, &m2), m_result);
    }

    #[test]
    fn naive_identity() {
        let n = Matrix::BLOCK * 3;
        let a = Matrix::id(n);
        let b = Matrix::random(n);
        assert_eq!(multiply(&a, &b), b);
    }

    #[test]
    fn blocked_identity() {
        let n = Matrix::BLOCK * 3;
        let a = Matrix::id(n);
        let b = Matrix::random(n);
        assert_eq!(multiply_blocked(&a, &b), b);
    }

    #[test]
    fn vectorized_identity() {
        let n = Matrix::BLOCK * 3;
        let a = Matrix::id(n);
        let b = Matrix::random(n);
        assert_eq!(multiply_vectorized(&a, &b), b);
    }

    #[test]
    fn rayon_identity() {
        let n = Matrix::BLOCK * 3;
        let a = Matrix::id(n);
        let b = Matrix::random(n);
        assert_eq!(multiply_rayon(&a, &b), b);
    }

    #[test]
    fn consistent() {
        let n = Matrix::BLOCK * 3;
        let a = Matrix::random(n);
        let b = Matrix::random(n);
        let r = multiply(&a, &b);
        assert_eq!(r, multiply_vectorized(&a, &b));
        assert_eq!(r, multiply_blocked(&a, &b));
        assert_eq!(r, multiply_rayon(&a, &b));
    }
}
