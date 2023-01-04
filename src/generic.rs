use num_traits::{One, Signed, Zero};
use rand::distributions::uniform::SampleUniform;
use rand::Rng;
use rayon::prelude::*;
use std::ops::{Index, IndexMut};

#[derive(Debug, PartialEq)]
pub struct Matrix<T> {
    n: usize,
    values: Vec<T>,
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &T {
        &self.values[self.n * index.0 + index.1]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut T {
        &mut self.values[self.n * index.0 + index.1]
    }
}

impl<T> Matrix<T> {
    const BLOCK: usize = 64;

    pub fn new(n: usize, values: Vec<T>) -> Self {
        assert_eq!(n * n, values.len());
        Matrix {
            n: n,
            values: values,
        }
    }

    pub fn number_of_blocks(&self) -> usize {
        assert_eq!(
            self.n % Self::BLOCK,
            0,
            "matrix size must be a multiple of the block size"
        );

        self.n / Self::BLOCK
    }
}

impl<T: Zero + Clone> Matrix<T> {
    pub fn zero(n: usize) -> Self {
        Self::new(n, vec![T::zero(); n * n])
    }
}

impl<T: One + Zero + Clone> Matrix<T> {
    pub fn id(n: usize) -> Self {
        let mut m = Self::zero(n);
        for i in 0..n {
            m[(i, i)] = T::one();
        }
        m
    }
}

impl<T> Matrix<T>
where
    T: Zero + One + Signed + SampleUniform + PartialOrd,
{
    pub fn random(n: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self::new(
            n,
            (0..n * n)
                .map(|_| rng.gen_range(-T::one()..T::one()))
                .collect(),
        )
    }
}

pub fn multiply<T>(a: &Matrix<T>, b: &Matrix<T>) -> Matrix<T>
where
    T: Zero + Signed + Copy,
{
    assert!(a.n == b.n);
    let n = a.n;
    let mut c = Matrix::<T>::zero(n);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                c[(i, j)] = c[(i, j)] + a[(i, k)] * b[(k, j)];
            }
        }
    }
    c
}

pub fn multiply_blocked<T>(a: &Matrix<T>, b: &Matrix<T>) -> Matrix<T>
where
    T: Zero + Signed + Copy,
{
    assert!(a.n == b.n);
    let n = a.n;
    let s = Matrix::<T>::BLOCK;
    let nn = a.number_of_blocks();
    let mut c = Matrix::<T>::zero(n);

    for ii in 0..nn {
        for jj in 0..nn {
            for kk in 0..nn {
                for i in ii * s..(ii + 1) * s {
                    for j in jj * s..(jj + 1) * s {
                        for k in kk * s..(kk + 1) * s {
                            c[(i, j)] = c[(i, j)] + a[(i, k)] * b[(k, j)];
                        }
                    }
                }
            }
        }
    }
    c
}

pub fn multiply_iter<T>(a: &Matrix<T>, b: &Matrix<T>) -> Matrix<T>
where
    T: Zero + Signed + Copy,
{
    assert!(a.n == b.n);
    let n = a.n;
    let s = Matrix::<T>::BLOCK;
    let nn = a.number_of_blocks();
    let mut c = Matrix::<T>::zero(n);

    let cc = c.values.chunks_mut(n);
    let aa = a.values.chunks(n);

    cc.zip(aa).for_each(|(cv, av)| {
        for jj in 0..nn {
            for (i, c) in cv.iter_mut().enumerate() {
                for j in (jj * s)..(jj + 1) * s {
                    *c = *c + av[j] * b.values[j * n + i];
                }
            }
        }
    });
    c
}

pub fn multiply_rayon<T>(a: &Matrix<T>, b: &Matrix<T>) -> Matrix<T>
where
    T: Zero + Signed + Copy + Send + Sync,
{
    assert!(a.n == b.n);
    let n = a.n;
    let s = Matrix::<T>::BLOCK;
    let nn = a.number_of_blocks();
    let mut c = Matrix::<T>::zero(n);

    let cc = c.values.par_chunks_mut(n);
    let aa = a.values.par_chunks(n);

    cc.zip(aa).for_each(|(cv, av)| {
        for jj in 0..nn {
            for (i, c) in cv.iter_mut().enumerate() {
                for j in (jj * s)..(jj + 1) * s {
                    *c = *c + av[j] * b.values[j * n + i];
                }
            }
        }
    });
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn indexes() {
        let mut m = Matrix::new(2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 0)], 3.0);

        m[(1, 0)] = 5.0;
        assert_eq!(m[(1, 0)], 5.0);
    }

    #[test]
    fn naive_identity() {
        let n = Matrix::<f64>::BLOCK * 3;
        let a = Matrix::<f64>::id(n);
        let b = Matrix::<f64>::random(n);
        assert_eq!(multiply(&a, &b), b);
    }

    #[test]
    fn blocked_identity() {
        let n = Matrix::<f64>::BLOCK * 3;
        let a = Matrix::<f64>::id(n);
        let b = Matrix::<f64>::random(n);
        assert_eq!(multiply_blocked(&a, &b), b);
    }

    #[test]
    fn iter_identity() {
        let n = Matrix::<f64>::BLOCK * 3;
        let a = Matrix::<f64>::id(n);
        let b = Matrix::<f64>::random(n);
        assert_eq!(multiply_iter(&a, &b), b);
    }

    #[test]
    fn rayon_identity() {
        let n = Matrix::<f64>::BLOCK * 3;
        let a = Matrix::<f64>::id(n);
        let b = Matrix::<f64>::random(n);
        assert_eq!(multiply_rayon(&a, &b), b);
    }

    #[test]
    fn consistent() {
        let n = Matrix::<f64>::BLOCK * 3;
        let a = Matrix::<f64>::random(n);
        let b = Matrix::<f64>::random(n);
        let r = multiply(&a, &b);
        assert_eq!(r, multiply_blocked(&a, &b));
        assert_eq!(r, multiply_iter(&a, &b));
        assert_eq!(r, multiply_rayon(&a, &b));
    }
}
