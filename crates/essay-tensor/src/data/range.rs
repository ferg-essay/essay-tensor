use std::{ops::{RangeBounds, Bound}};

use crate::{Tensor, tensor};

use super::dataset::Dataset;

#[derive(Clone)]
pub struct DatasetRange {
    start: usize,
    stop: usize,
    step: usize,
}

pub fn range<R>(range: R, step: Option<usize>) -> DatasetRange
where R: RangeBounds<usize>,
{
    let start = match range.start_bound() {
        Bound::Included(value) => *value,
        Bound::Excluded(value) => *value + 1,
        Bound::Unbounded => 0,
    };

    let stop = match range.end_bound() {
        Bound::Included(value) => *value + 1,
        Bound::Excluded(value) => *value,
        Bound::Unbounded => usize::MAX,
    };

    DatasetRange {
        start,
        stop,
        step: match step { Some(step) => step, None => 1 },
    }
}

impl Dataset<f32> for DatasetRange {
    type IntoIter=RangeIter;

    fn iter(&self) -> Self::IntoIter {
        RangeIter {
            index: self.start,
            stop: self.stop,
            step: self.step,
        }
    }

    fn get_single_element(&self) -> Tensor<f32> {
        Tensor::from(self.start as f32)
    }
}

pub struct RangeIter {
    index: usize,
    stop: usize,
    step: usize,
}

impl Iterator for RangeIter {
    type Item=Tensor<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.stop {
            let value = self.index;
            self.index += self.step;
            Some(tensor!(value as f32))
        } else {
            None
        }
    }
}


#[cfg(test)]
mod test {
    use crate::prelude::*;

    use super::range;

    #[test]
    fn test_range() {
        let data = range(0..4, None);
        assert_eq!(
            &data.iter().collect::<Vec<Tensor<f32>>>(),
            &vec![tensor!(0.), tensor!(1.), tensor!(2.), tensor!(3.)]
        );

        let data = range(1..4, None);
        assert_eq!(
            &data.iter().collect::<Vec<Tensor<f32>>>(),
            &vec![tensor!(1.), tensor!(2.), tensor!(3.)]
        );

        let data = range(1..=4, None);
        assert_eq!(
            &data.iter().collect::<Vec<Tensor<f32>>>(),
            &vec![tensor!(1.), tensor!(2.), tensor!(3.), tensor!(4.)]
        );

        let data = range(1..=4, Some(2));
        assert_eq!(
            &data.iter().collect::<Vec<Tensor<f32>>>(),
            &vec![tensor!(1.), tensor!(3.),]
        );
    }

    #[test]
    fn test_range_get_single_element() {
        let data = range(0..0, None);
        assert_eq!(data.get_single_element(), tensor!(0.));
    }
}