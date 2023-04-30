use crate::{Tensor};

use super::Var;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TensorId(pub usize);

pub struct Tape {
    tensors: Vec<Tensor>,
}

impl Tape {
    pub fn new() -> Tape {
        Self {
            tensors: Default::default(),
        }
    }

    pub fn var(&mut self, var: &Var) -> Tensor {
        println!("Var: {:?}", var);
        var.tensor()
    }

    pub fn gradient(
        &self, 
        loss: &Tensor, 
        var: &Var
    ) -> Tensor {
        let op = match loss.op() {
            Some(op) => op,
            None => panic!("gradient needs saved graph in loss tensor"),
        };
        
        println!("var {:?}", var);
        println!("loss {:?}", loss);
        op.gradient(self, 0)
    }
}

#[cfg(test)]
mod test {
    use crate::model::Var;
    use crate::model::tape::Tape;
    use crate::{Tensor, random::uniform};
    use crate::prelude::{*};

    #[test]
    fn test_mse() {
        let mut tape = Tape::new();
        let z = Var::new("z", tensor!(0.));
        let z_t = tape.var(&z);

        let y : Tensor = tensor!(1.0);
        let loss: Tensor = z_t.mean_square_error(&y);

        println!("z {:#?}", &z_t);
        println!("y {:#?}", &y);
        println!("loss {:#?}", &loss);

        let dz = tape.gradient(&loss, &z);
        println!("dz {:#?}", &dz);
    }

    #[test]
    fn test_matvec() {
        let w = Var::new("w", tensor!(0.5));
        let b = Var::new("b", tensor!(0.5));

        let mut tape = Tape::new();
        let w_t = tape.var(&w);
        let b_t = tape.var(&b);

        let x = tensor!(0.0);

        let z = x.clone() * w_t.clone() + b_t;

        let y : Tensor = tensor!(2.0) * x + 1.0.into();
        let loss: Tensor = z.mean_square_error(&y);

        println!("w_t {:#?}", &w_t);
        println!("{:#?} loss {:#?}", &z, &loss);

        let dw = tape.gradient(&loss, &w);

        println!("w_t {:#?}", &w_t);
        println!("{:#?} loss {:#?}", &z, &loss);
        println!("dw {:#?}", &dw);
    }
}