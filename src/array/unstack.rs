use crate::{Tensor, tensor::{Dtype, TensorUninit}};

use super::{AxisOpt, axis::axis_from_rank};

// TODO: unstack can't currenly be written as an operation.
pub fn unstack<D: Dtype>(
    tensor: impl Into<Tensor<D>>, 
    axis: impl Into<AxisOpt>
) -> Vec<Tensor<D>> {
    let tensor = tensor.into();

    tensor.unstack(axis)
}

impl<D: Dtype> Tensor<D> {
    #[inline]
    pub fn unstack(&self, axis: impl Into<AxisOpt>) -> Vec<Tensor<D>> {
        let section = sections.into_sections();
        let axis : AxisOpt = axis.into();
    
        let axis = axis_from_rank(&axis.get_axis(), tensor.rank());
    
        let len = tensor.shape().dim(axis);
        let mut cuts = Vec::<usize>::new();
    
    
        if axis == 0 {
            let mut slices = Vec::<Tensor<D>>::new();
    
            let mut prev = 0;
            for i in cuts {
                if i != prev {
                    slices.push(tensor.subslice(prev, i - prev));
                }
                prev = i;
            }
        
    
            slices
        } else {
            split_by_axis(tensor, axis, cuts)
        }
    }
}

#[cfg(test)]
mod test {
    use split::{vsplit, hsplit, dsplit};

    use crate::{prelude::*, array::{split}};
    
    #[test]
    fn test_split() {
        assert_eq!(
            split(&tf32!([[1., 2.], [3., 4.]]), 2, ()), 
            vec![tf32!([[1., 2.]]), tf32!([[3., 4.]])],
        );

        assert_eq!(
            split(&tf32!([1., 2., 3., 4.]), [1, 3], ()), 
            vec![tf32!([1.]), tf32!([2., 3.]), tf32!([4.])],
        );
    }
    
    #[test]
    fn test_split_axis() {
        assert_eq!(
            split(&tf32!([[1., 2.], [3., 4.]]), 2, 1), 
            vec![tf32!([[1.], [3.]]), tf32!([[2.], [4.]])],
        );
    }
    
    #[test]
    fn test_vsplit() {
        assert_eq!(
            vsplit(&tf32!([[1., 2.], [3., 4.]]), 2), 
            vec![tf32!([[1., 2.]]), tf32!([[3., 4.]])],
        );

        assert_eq!(
            vsplit(&tf32!([1., 2., 3., 4.]), [1, 3]), 
            vec![tf32!([1.]), tf32!([2., 3.]), tf32!([4.])],
        );
    }
    
    #[test]
    fn test_hsplit() {
        assert_eq!(
            hsplit(&tf32!([[1., 2.], [3., 4.]]), 2), 
            vec![tf32!([[1.], [3.]]), tf32!([[2.], [4.]])],
        );
    }
    
    #[test]
    fn test_dsplit() {
        assert_eq!(
            dsplit(&tf32!([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]), 2), 
            vec![
                tf32!([[[1.], [3.]], [[5.], [7.]]]),
                tf32!([[[2.], [4.]], [[6.], [8.]]])
            ],
        );
    }
}
