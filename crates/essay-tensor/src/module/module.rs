use std::collections::HashMap;

use crate::Tensor;

use super::Var;

pub trait Bundle {}

pub struct Train<'a, In:Bundle, Out:Bundle> {
    out: Out,
    module: &'a Module<In, Out>,
}

pub struct Module<In:Bundle, Out:Bundle> {
    vars: HashMap<String, Var>,
    fun: Box<dyn Fn(In) -> Out>,
}

impl<In:Bundle, Out:Bundle> Module<In, Out> {
    pub fn build<F>(init: In, fun: F) -> Module<In, Out>
    where
        F: FnOnce(In) -> Out
    {
        todo!()
    }

    pub fn eval(&self, input: In) -> Out {
        todo!()
    }

    pub fn train(&self, input: In) -> Train<In, Out> {
        todo!()
    }
}

impl Bundle for Tensor {
}

impl Bundle for () {
}

impl Bundle for (Tensor, Tensor) {
}


#[cfg(test)]
mod test {
    use crate::{module::{Var, Tape, module::Module}, tensor, Tensor};

    #[test]
    fn backprop_1_1_prev() {
        let a = Var::new("a", tensor!([[1.]]));
        let x = Var::new("x", tensor!([1.]));

        let module = Module::build(tensor!(1.), |x| {
            (x.clone(), &a * &x)
        });

        let value = module.eval(tensor!(3.));
        let train = module.train(tensor!(3.));
    
        let mut tape = Tape::with(|| {
            let out: Tensor = a.matvec(&x);

            let loss = out.l2_loss();
            assert_eq!(loss, tensor!(0.5));
    
            Ok(loss)
        }).unwrap();
    
        let da = tape.gradient(&a);
        assert_eq!(da, tensor!([[1.0]]));
    
        let dx = tape.gradient(&x);
        assert_eq!(dx, tensor!([1.0]));
    }
}
