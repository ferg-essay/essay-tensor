#[cfg(test)]
mod test {
    use crate::expr::Var;
    use crate::{Tensor, random::uniform};
    use crate::prelude::{*};

    #[test]
    fn test() {
        let model = Model::new();
        println!("{:?}", model.call(0.0.into()));
        println!("{:?}", model.call(0.0.into()));
        println!("{:?}", model.call(0.5.into()));
        println!("{:?}", model.call(1.0.into()));
    }

    struct Model {
        w: Var<0>,
        b: Var<0>,
    }

    impl Model {
        fn new() -> Self {
            let v = uniform([2], 0., 1., Some(100));
            Self {
                w: Var::new("w", tensor!(v.get(0).unwrap())),
                b: Var::new("b", tensor!(v.get(1).unwrap())),
            }
        }

        fn call(&self, x: Tensor<0>) -> Tensor<0> {
            self.w.tensor() * x + self.b.tensor()
        }
    }
}