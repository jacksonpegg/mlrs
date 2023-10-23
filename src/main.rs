use mlrs::{matrix::Matrix, nn::NN};

fn main() {
    #[rustfmt::skip]
    let xor_model: Matrix<f32> = Matrix::from_vec(4, 3, vec![
        1.0, 1.0, 1.0, 
        1.0, 1.0, 0.0, 
        1.0, 0.0, 1.0, 
        0.0, 0.0, 0.0,
    ]).unwrap();

    let mut nn = NN::new(vec![2, 2, 1]).unwrap();
    nn.rand();
    nn.test_model(&xor_model).unwrap();
    nn.train_model(&xor_model, 10_000).unwrap();
}
