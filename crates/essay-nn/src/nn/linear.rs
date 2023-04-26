use ndarray::Array2;

use super::{Tensor, Model};

#[derive(Clone, Debug)]
pub struct Linear {
    weights: Array2<f32>,
}

impl Linear {
    pub fn new(n_in: usize, n_out: usize) -> Self {
        Self {
            weights: Array2::<f32>::zeros((n_out, n_in)),
        }
    }

    pub fn done(self) -> Box<dyn Model> {
        Box::new(self)
    }

    fn set(&mut self, r: usize, c: usize, value: f32) -> &mut Self {
        *self.weights.get_mut((r, c)).unwrap() = value;
        self
    }
}

impl Model for Linear {
    fn n_input(&self) -> usize {
        self.weights.shape()[0]
    }

    fn n_output(&self) -> usize {
        self.weights.shape()[1]
    }

    fn forward(&mut self, data: &Array2<f32>) -> Array2<f32> {
        let result = &self.weights.dot(data);
        println!();
        println!("result {:?}", result);
        println!("weights {:?}", self.weights);
        println!("dot {:?}", self.weights);
        result.clone()
    }

    fn box_clone(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod test {
    use ndarray::{Array2, ArrayView, Array1};

    use crate::nn::Tensor;

    use super::{Linear, Model};

    #[test]
    fn linear() {
        let mut linear = Linear::new(2, 1);

        linear.set(0, 0, 1.);
        //linear.forward(tensor);
        println!("line {:?}", linear);

        let mut input = Array2::<f32>::zeros((2, 0));
        input.push_column(ArrayView::from(&[1., 0.])).unwrap();
        // let i2: Array1<f32> = ArrayView::from(&[2., 0.]);
        let out = linear.forward(&input);

        println!();
        println!("input {:?}", input);
        println!("out {:?}", out);
        println!("o2 {:?}", out.get((0, 0)).unwrap());
    }
}