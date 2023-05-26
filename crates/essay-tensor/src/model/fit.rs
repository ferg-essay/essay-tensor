use crate::{prelude::Dataset, dataset::DatasetIter, Tensor};

pub struct Fit<'a> {
    // model: model
    // optimizer
    // loss
    epochs: usize,

    // batch_size: usize,
    steps_per_epoch: usize,

    train_x: DatasetIter<'a>,
    train_y: DatasetIter<'a>,

    // validation_x: Option<Dataset>,
    // validation_y: Option<Dataset>,
}

impl Fit<'_> {
    fn step(&mut self) {
        println!("Step!");
    }
}

pub struct FitBuilder {
    train_x: Dataset,
    train_y: Dataset,

    epochs: usize,
    steps_per_epoch: usize,
}

impl FitBuilder {
    fn new_tensor(
        train_x: impl Into<Tensor>, 
        train_y: impl Into<Tensor>
    ) -> FitBuilder {
        let train_x = train_x.into();
        let train_y = train_y.into();

        assert_eq!(train_x.dim(0), train_y.dim(0));

        //let data_x = train_x.from_tensor_slices(train_x).batch(64);
        //let data_y = train_y.from_tensor_slices(train_y).batch(64);

        //let data_x = from_tensor_slices(train_x);
        //let data_y = from_tensor_slices(train_y);
        /*
        FitBuilder {
            train_x: train_x.into(),
            train_y: train_y.into(),
            epochs: 1,
            steps_per_epoch: usize::MAX,
        }
        */
        todo!();
    }

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

    fn build(&mut self) -> Fit {
        Fit {
            train_x: self.train_x.iter(),
            train_y: self.train_y.iter(),

            epochs: self.epochs,
            steps_per_epoch: self.steps_per_epoch,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::model::fit::FitBuilder;
    pub use crate::prelude::*;

    #[test]
    fn fit_builder() {
        let train_x = tf32!([1., 2., 3.]);
        let train_y = tf32!([2., 4., 6.]);

        let mut builder = FitBuilder::new_dataset(&train_x, &train_y);

        let mut fit = builder.build();

        fit.step();
    }
}