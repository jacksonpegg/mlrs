use crate::matrix::{Matrix, MatrixError};
use rand;

impl Matrix<f32> {
    pub fn sigmoid(&mut self) {
        self.iter_mut().for_each(|x| {
            *x = sigmoid(*x);
        });
    }
    pub fn rand(&mut self) {
        self.iter_mut().for_each(|x| {
            *x = rand::random();
        });
    }
}

fn sigmoid(i: f32) -> f32 {
    1.0 / (1.0 + std::f32::consts::E.powf(-i))
}

#[derive(Debug)]
struct Layer {
    weights: Matrix<f32>,
    biases: Matrix<f32>,
}

impl Layer {
    pub fn new(prev: usize, current: usize) -> Result<Self, MatrixError> {
        let weights = Matrix::new(current, prev)?;
        let biases = Matrix::new(current, 1)?;
        return Ok(Layer { weights, biases });
    }
    fn activate(&self, activation: Matrix<f32>) -> Matrix<f32> {
        let mut result = (self.weights.clone() * activation.clone()) + self.biases.clone();
        result.sigmoid();
        return result;
    }
}

impl std::fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Weights:\n {}\n", self.weights)?;
        write!(f, "Biases: \n {}", self.biases)?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct NN {
    data: Vec<Layer>,
}

impl NN {
    pub fn new(sizes: Vec<usize>) -> Result<NN, MatrixError> {
        let mut data = Vec::with_capacity(sizes.len());
        let mut iter = sizes.iter().peekable();
        while let Some(n) = iter.next() {
            match iter.peek() {
                Some(m) => {
                    data.push(Layer::new(*n, **m)?);
                }
                None => {}
            }
        }
        Ok(NN { data })
    }
    fn activate(&self, mut input: Matrix<f32>) -> Matrix<f32> {
        for i in self.data.iter() {
            input = i.activate(input);
        }
        return input;
    }
    pub fn rand(&mut self) {
        for i in self.data.iter_mut() {
            for j in i.weights.iter_mut() {
                *j = rand::random();
            }
            for j in i.biases.iter_mut() {
                *j = rand::random();
            }
        }
    }
    pub fn train_model(&mut self, model: &Matrix<f32>, epochs: usize) -> Result<(), MatrixError> {
        return Ok(());
    }

    pub fn test_model(&self, model: &Matrix<f32>) -> Result<(), MatrixError> {
        let input_layer = &self.data[0];
        let rows = input_layer.weights.rows() + 1;
        assert!(model.size() % rows == 0);
        for (i, mut j) in model.row_chunks().enumerate() {
            j = &j[1..];
            println!(
                "Case {}:\n\tResult: {}\n\tExpected: {}",
                i,
                self.activate(Matrix::from_vec(rows - 1, 1, (*j).to_vec()).unwrap()),
                j[0],
            );
        }
        return Ok(());
    }
}

impl std::fmt::Display for NN {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.data.iter().for_each(|x| {
            writeln!(f, "\n{}", x).unwrap();
        });
        Ok(())
    }
}
