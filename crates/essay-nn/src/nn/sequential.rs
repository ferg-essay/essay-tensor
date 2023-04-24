use super::Model;

pub struct Sequential {
    models: Vec<Box<dyn Model>>,
}

impl Sequential {
    pub fn new(models: &[Box<dyn Model>]) -> Sequential {
        let box_models: Vec<Box<dyn Model>> = models.iter().map(|m| m.box_clone()).collect();

        Self {
            models: box_models,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::nn::{linear::Linear, relu::ReLU};

    use super::Sequential;

    #[test]
    fn test() {
        let _model = Sequential::new(&[
            Linear::new(20, 8),
            ReLU::new(8),
            Linear::new(8, 20),
        ]);
    }
}