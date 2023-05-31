use std::ops::Deref;

use essay_opt::derive_opt;

use crate::{
    prelude::Dataset, 
    dataset::{DatasetIter, rebatch}, Tensor, loss::l2_loss,
    function::Trainer,
};

use super::model::Model;

pub struct Fit<'a> {
    // model: model
    // optimizer
    // loss
    epochs: usize,

    // batch_size: usize,
    steps_per_epoch: usize,

    train_x: DatasetIter<'a>,
    train_y: DatasetIter<'a>,

    trainer: Trainer<(Tensor, Tensor), (Tensor, Tensor)>,

    // validation_x: Option<Dataset>,
    // validation_y: Option<Dataset>,
}

impl Fit<'_> {
    pub fn step(&mut self) -> bool {
        let x = self.train_x.next();
        let y = self.train_y.next();

        if x.is_none() {
            return false;
        }

        let x = x.unwrap();
        let y = y.unwrap();

        // let x_ptr = x.clone();
        // let y_ptr = y.clone();

        let train = self.trainer.train((x, y));


        let rate = 0.1f32;

        for (id, grad) in train.gradients() {
            let var = self.trainer.get_var(id);

            var.assign_sub(rate * grad);
        }

        true
    }
}

pub struct FitBuilder {
    model: Option<Box<dyn FnOnce(Tensor) -> Tensor>>,
    train_x: Dataset,
    train_y: Dataset,

    epochs: usize,
    steps_per_epoch: usize,
}

//pub trait FitOpt {
//    fn epochs(self, epochs: usize) -> FitArgs;
//
//    fn into(self) -> FitArgs;
//}

#[derive(Default)]
#[derive_opt(FitOpt)]
pub struct FitArgs {
    epochs: usize,
    steps_per_epoch: usize,
}
/*
//derive_arg!(FitOpt, FitArg);
impl FitOpt for FitArgs {
    fn epochs(self, epochs: usize) -> FitArgs {
        Self { epochs, ..self }
    }

    fn into(self) -> FitArgs {
        self
    }
}

impl FitOpt for () {
    fn epochs(self, epochs: usize) -> FitArgs {
        FitArgs::default().epochs(epochs)
    }

    fn into(self) -> FitArgs {
        FitArgs::default()
    }
}
*/

fn test(opt: impl FitOpt) {

}

fn my_test() {
    test(().epochs(1));
}

impl FitBuilder {
    fn new_tensor(
        model: impl FnOnce(Tensor) -> Tensor + 'static,
        train_x: impl Into<Tensor>, 
        train_y: impl Into<Tensor>,
        options: impl FitOpt,
    ) -> FitBuilder {
        let train_x = train_x.into();
        let train_y = train_y.into();

        let options = options.into_arg();

        assert_eq!(train_x.dim(0), train_y.dim(0));

        let train_x = rebatch(train_x, 2);
        let train_y = rebatch(train_y, 2);

        FitBuilder {
            model: Some(Box::new(model)),
            train_x,
            train_y,
            epochs: options.epochs,
            steps_per_epoch: options.steps_per_epoch,
        }
    }

    /*
    fn new_dataset(
        train_x: impl Into<Dataset>, 
        train_y: impl Into<Dataset>
    ) -> FitBuilder {
        FitBuilder {
            train_x: train_x.into(),
            train_y: train_y.into(),
            epochs: 1,
            steps_per_epoch: usize::MAX,
        }
    }
    */

    fn build(&mut self) -> Fit {
        let x = self.train_x.get_single_element();
        let y = self.train_y.get_single_element();

        let trainer = Trainer::compile((x, y), |(x, y)| {
            let model = self.model.take().unwrap();
            let y_pred = model(x);
            let loss = l2_loss(&y - &y_pred);

            (y_pred, loss)
        });

        Fit {
            trainer,
            
            train_x: self.train_x.iter(),
            train_y: self.train_y.iter(),

            epochs: self.epochs,
            steps_per_epoch: self.steps_per_epoch,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{model::{fit::FitBuilder, model::Model}, function::Var};
    pub use crate::prelude::*;
    pub use crate::model::prelude::*;

    #[test]
    fn fit_builder() {
        let train_x = tf32!([[1.], [2.], [3.]]);
        let train_y = tf32!([[2.], [4.], [6.]]);

        let a = Var::new("a", tf32!([[1.]]));
        let b = Var::new("b", tf32!([1.]));

        let a_ptr = a.clone();
        let b_ptr = b.clone();

        let mut builder = FitBuilder::new_tensor(
            move |x| &a_ptr.matvec(&x) + &b_ptr,
            &train_x, 
            &train_y,
            ().epochs(2)
        );

        let mut fit = builder.build();

        fit.step();

        assert_eq!(a.tensor(), tf32!([[1.2]]));
        assert_eq!(b.tensor(), tf32!([1.1]));

        fit.step();

        assert_eq!(a.tensor(), tf32!([[1.5899999]]));
        assert_eq!(b.tensor(), tf32!([1.23]));
    }
}