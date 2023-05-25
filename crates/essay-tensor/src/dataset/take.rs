
//
// DataSource.take
//

use std::marker::PhantomData;

use crate::{flow::{self, Out, Source, In, FlowData, SourceId}};

use super::{IntoFlowBuilder};

pub struct Take<T: FlowData> {
    count: usize, 
    marker: PhantomData<T>,
}

impl<T: FlowData> Take<T> {
    pub(crate) fn build(
        builder: &mut IntoFlowBuilder,
        input: SourceId<T>,
        count: usize
    ) -> SourceId<T> {
        builder.source(move || {
            Take {
                count,
                marker: Default::default(),
            }
        }, &input)
    }
}

impl<T> Source<T, T> for Take<T> 
where
    T: FlowData,
{
    fn next(&mut self, input: &mut In<T>) -> flow::Result<Out<T>> {
        if self.count > 0 {
            self.count -= 1;

            Ok(Out::from(input.next()))
        } else {
            Ok(Out::None)
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, dataset::range};

    #[test]
    fn test_take() {
        let mut data = range(4.., None).take(2);
        assert_eq!(
            &data.iter().collect::<Vec<Tensor<f32>>>(),
            &vec![tensor!(4.), tensor!(5.)]
        );
    }
}