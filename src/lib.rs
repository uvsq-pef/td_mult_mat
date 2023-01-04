use std::fmt;
use std::ops::Index;
use std::ops::IndexMut;
use rand::Rng;
use rayon::prelude::*;

type Element = f64;

#[derive(Debug, PartialEq)]
pub struct Matrix {
    n: usize,
    values: Vec<Element>,
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

    pub fn number_of_blocks(&self) -> usize {
        assert_eq!(
            self.n % Matrix::BLOCK,
            0,
            "matrix size must be a multiple of the block size"
        );

        self.n / Matrix::BLOCK
    }
}

pub fn multiply(m1: &Matrix, m2: &Matrix) -> Matrix {
    assert!(m1.n == m2.n);

    let size = m1.n;
    let mut m_result = Matrix::zero(size);

    for i in 0..size {
        for j in 0..size {
            for k in 0..size {
                m_result[(i, j)] += m1[(i, k)] * m2[(k, j)];
            }
        }
    }
    m_result
}

pub fn multiply_blocked(a: &Matrix, b: &Matrix) -> Matrix {
    assert!(a.n == b.n);
    let n = a.n;
    let s = Matrix::BLOCK;
    let nn = a.number_of_blocks();
    let mut c = Matrix::zero(n);

    for ii in 0..nn {
        for jj in 0..nn {
            for kk in 0..nn {
                for i in ii * s..(ii + 1) * s {
                    for j in jj * s..(jj + 1) * s {
                        for k in kk * s..(kk + 1) * s {
                            c[(i, j)] += a[(i, k)] * b[(k, j)];
                        }
                    }
                }
            }
        }
    }
    c
}

pub fn multiply_iter(a: &Matrix, b: &Matrix) -> Matrix {
    assert!(a.n == b.n);
    let n = a.n;
    let s = Matrix::BLOCK;
    let nn = a.number_of_blocks();
    let mut c = Matrix::zero(n);

    let cc = c.values.chunks_mut(n);
    let aa = a.values.chunks(n);

    cc.zip(aa).for_each(|(cv, av)| {
        for jj in 0..nn {
            for (i, c) in cv.iter_mut().enumerate() {
                for j in (jj * s)..(jj + 1) * s {
                    *c += av[j] * b.values[j * n + i];
                }
            }
        }
    });
    c
}

pub fn multiply_rayon(a: &Matrix, b: &Matrix) -> Matrix {
    assert!(a.n == b.n);
    let n = a.n;
    let s = Matrix::BLOCK;
    let nn = a.number_of_blocks();
    let mut c = Matrix::zero(n);

    let cc = c.values.par_chunks_mut(n);
    let aa = a.values.par_chunks(n);

    cc.zip(aa).for_each(|(cv, av)| {
        for jj in 0..nn {
            for (i, c) in cv.iter_mut().enumerate() {
                for j in (jj * s)..(jj + 1) * s {
                    *c += av[j] * b.values[j * n + i];
                }
            }
        }
    });
    c
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        const ELEMENT_SIZE: usize = 5;
        let padding = self.n * (ELEMENT_SIZE + 1);
        write!(f, "┌ {:1$} ┐", " ", padding).ok();
        let mut line = String::new();
        for (i, value) in self.values.iter().enumerate() {
            if i % self.n == 0 {    // new line
                if i > 0 {
                    write!(f, "{} │", line).ok();
                }
                line = String::from("\n│ ");
            }
            let element = format!("{:1$.2}", value, ELEMENT_SIZE);
            line.push_str(&element);
            line.push(' ');
            
        }
        write!(f, "{} │", line).ok();
        write!(f, "\n└ {:1$} ┘\n", " ", padding)
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

    #[test]
    fn should_naively_multiply_random_by_identity() {
        const SIZE : usize = 16;

        let m1 = Matrix::id(SIZE);
        let m2 = Matrix::random(SIZE);
        assert_eq!(multiply(&m1, &m2), m2);
    }

    #[test]
    fn should_naively_multiply_two_matrices() {
        const SIZE : usize = 2;

        let m1 = Matrix::new(SIZE, vec![1.0, 2.0, 3.0, 4.0]);
        let m2 = Matrix::new(SIZE, vec![4.0, 3.0, 2.0, 1.0]);
        let m_result = Matrix::new(SIZE, vec![8.0, 5.0, 20.0, 13.0]);
        assert_eq!(multiply(&m1, &m2), m_result);
    }

    #[test]
    fn should_multiply_by_bloc_random_by_identity() {
        const SIZE : usize = 256;

        let m1 = Matrix::id(SIZE);
        let m2 = Matrix::random(SIZE);
        assert_eq!(multiply_blocked(&m1, &m2), m2);
    }

    #[test]
    fn should_multiply_by_iter_random_by_identity() {
        const SIZE : usize = 256;

        let m1 = Matrix::id(SIZE);
        let m2 = Matrix::random(SIZE);
        assert_eq!(multiply_iter(&m1, &m2), m2);
    }

    #[test]
    fn should_multiply_by_rayon_random_by_identity() {
        const SIZE : usize = 256;

        let m1 = Matrix::id(SIZE);
        let m2 = Matrix::random(SIZE);
        assert_eq!(multiply_rayon(&m1, &m2), m2);
    }
}
