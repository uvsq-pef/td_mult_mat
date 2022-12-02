use mult_mat::Matrix;

fn main() {
    let values = vec![1.0, 2.0, 3.0, 4.0];
    let m = Matrix::new(2, values.clone());
    println!("{:?}", m);
}
