use mult_mat::*;

fn main() {
    let arg1 = std::env::args()
        .nth(1)
        .expect("usage: matrixmult <algo> <n>");
    let arg2 = std::env::args()
        .nth(2)
        .expect("usage: matrixmult <algo> <n>");
    let n: usize = arg2.parse().expect("<n> should be a positive integer");

    let a = Matrix::random(n);
    let b = Matrix::random(n);

    match arg1.as_str() {
        "naive" => multiply(&a, &b),
        "vectorized" => multiply_vectorized(&a, &b),
        "blocked" => multiply_blocked(&a, &b),
        "rayon" => multiply_rayon(&a, &b),
        _ => panic!("<algo> should be either \"naive\", \"blocked\" or \"iter\""),
    };
}
