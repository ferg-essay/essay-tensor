
//
// DataSource.take
//

use std::marker::PhantomData;

use crate::{Tensor};

use super::Dataset;

#[derive(Clone)]
pub struct Take<T:Clone, D:Dataset<T>> {
    source: D,
    count: usize,
    marker: PhantomData<T>,
}

impl<T:Clone, D:Dataset<T>> Take<T, D> {
    pub(crate) fn new(source: D, count: usize) -> Self {
        Self {
            source,
            count,
            marker: Default::default()
        }
    }
}

impl<T:Clone, D:Dataset<T>> Dataset<T> for Take<T, D> {
    type IntoIter = TakeIter<T, D>;

    fn iter(&self) -> Self::IntoIter {
        TakeIter {
            source: self.source.iter(),
            count: self.count,
        }
    }

    fn get_single_element(&self) -> Tensor<T> {
        self.source.get_single_element()
    }
}

pub struct TakeIter<T:Clone, D:Dataset<T>> {
    source: D::IntoIter,
    count: usize,
}

impl<T:Clone, D:Dataset<T>> Iterator for TakeIter<T, D> {
    type Item = Tensor<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count > 0 {
            self.count -= 1;
            self.source.next()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, data::range};

    #[test]
    fn test_take() {
        let data = range(4.., None).take(2);
        assert_eq!(
            &data.iter().collect::<Vec<Tensor<f32>>>(),
            &vec![tensor!(4.), tensor!(5.)]
        );
    }
}