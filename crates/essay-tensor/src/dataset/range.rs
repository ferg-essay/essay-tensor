use std::{ops::{RangeBounds, Bound}};

use crate::{Tensor, tensor, flow::{Source, self, Out, SourceFactory}};

use super::{dataset::{Dataset, IntoFlow}, IntoFlowBuilder};

pub fn range<R>(range: R, step: Option<usize>) -> Dataset<Tensor<f32>>
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

    let step = match step { Some(step) => step, None => 1 };

    Dataset::from_flow(|builder| {
        builder.source(move || {
            Range {
                index: start,
                stop,
                step,
            }
        }, &())
    })
}

//
// Range - source
//

pub struct Range {
    index: usize,
    stop: usize,
    step: usize,
}

impl Source<(), Tensor<f32>> for Range {
    fn next(&mut self, _input: &mut ()) -> flow::Result<Out<Tensor<f32>>> {
        if self.index < self.stop {
            let value = self.index;

            self.index += self.step;

            Ok(Out::Some(tensor!(value as f32)))
        } else {
            Ok(Out::None)
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