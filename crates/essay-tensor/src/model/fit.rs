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

    trainer: Trainer<(Tensor, Tensor), Tensor>,

    // validation_x: Option<Dataset>,
    // validation_y: Option<Dataset>,
}

impl Fit<'_> {
    fn step(&mut self) -> bool {
        let x = self.train_x.next();
        let y = self.train_y.next();

        if x.is_none() {
            return false;
        }

        let x = x.unwrap();
        let y = y.unwrap();

        let x_ptr = x.clone();
        let y_ptr = y.clone();

        let train = self.trainer.train((x, y));

        println!("Step! {:?} {:?} loss={:?}", x_ptr, y_ptr, train.value());

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

impl FitBuilder {
    fn new_tensor(
        model: impl FnOnce(Tensor) -> Tensor + 'static,
        train_x: impl Into<Tensor>, 
        train_y: impl Into<Tensor>
    ) -> FitBuilder {
        let train_x = train_x.into();
        let train_y = train_y.into();

        assert_eq!(train_x.dim(0), train_y.dim(0));

        let train_x = rebatch(train_x, 2);
        let train_y = rebatch(train_y, 2);

        FitBuilder {
            model: Some(Box::new(model)),
            train_x,
            train_y,
            epochs: 1,
            steps_per_epoch: usize::MAX,
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

            l2_loss(&(&y - &y_pred))
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

    #[test]
    fn fit_builder() {
        let train_x = tf32!([[1.], [2.], [3.]]);
        let train_y = tf32!([[2.], [4.], [6.]]);

        let a = Var::new("a", tf32!([[1.]]));
        let b = Var::new("b", tf32!([1.]));

        // let model : Box<dyn Model> = Box::new(move 

        let mut builder = FitBuilder::new_tensor(
            move |x| &a.matvec(&x) + &b,
            &train_x, 
            &train_y
        );

        let mut fit = builder.build();

        fit.step();
    }
}