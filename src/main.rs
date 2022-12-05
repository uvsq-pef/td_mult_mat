use mult_mat::Matrix;

fn main() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 12.345, 7.0, 8.0, 9.0];
    let m = Matrix::new(3, values.clone());
    println!("{}", m);
}
