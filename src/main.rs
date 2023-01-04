use mult_mat::Matrix;
use mult_mat::multiply;

fn main() {
    let arg1 = std::env::args()
        .nth(1)
        .expect("usage: matrixmult <algo> <n>");
    let arg2 = std::env::args()
        .nth(2)
        .expect("usage: matrixmult <algo> <n>");

    let size: usize = arg2.parse().expect("<n> should be a positive integer");

    let m1 = Matrix::random(size);
    let m2 = Matrix::random(size);

    match arg1.as_str() {
        "naive" => {
            multiply(&m1, &m2);
        }
        "display" => {
            println!("{}multiplyied by\n{}gives\n{}", m1, m2, multiply(&m1, &m2));
        }
        _ => panic!("<algo> should be either \"naive\" or \"display\""),
    };
}
