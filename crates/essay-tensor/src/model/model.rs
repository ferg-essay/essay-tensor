#[cfg(test)]
mod test {
    use crate::model::{Var, Tape};
    use crate::{Tensor, random::uniform};
    use crate::prelude::{*};

    #[test]
    fn test() {
        let model = TestModel::new();
        let mut tape = Tape::new();
        
        let tail = model.call(&tape, 0.0.into());
        //tape.complete(&tail);

        println!("{:?}", &tail);
    }

    struct TestModel {
        w: Var,
        b: Var,
    }

    impl TestModel {
        fn new() -> Self {
            let v = uniform([2], 0., 1., Some(100));
            Self {
                w: Var::new("w", tensor!(v.get(0).unwrap())),
                b: Var::new("b", tensor!(v.get(1).unwrap())),
            }
        }

        fn call(&self, tape: &Tape, x: Tensor) -> Tensor {
            self.w.tensor() * x + self.b.tensor()
        }
    }
}