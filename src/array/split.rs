use crate::{Tensor, tensor::{Dtype, TensorUninit}};

use super::{AxisOpt, axis::axis_from_rank};

// TODO: split can't currenly be written as an operation.
pub fn split<D: Dtype>(
    tensor: impl Into<Tensor<D>>, 
    sections: impl IntoSections,
    axis: impl Into<AxisOpt>
) -> Vec<Tensor<D>> {
    let tensor = tensor.into();
    let section = sections.into_sections();
    let axis : AxisOpt = axis.into();

    let axis = axis_from_rank(&axis.get_axis(), tensor.rank());

    let len = tensor.shape().dim(axis);
    let mut cuts = Vec::<usize>::new();

    match section {
        Sections::SplitEqual(n) => {
            let step = len / n;
            let mut i = 0;
            while i < len {
                i += step;
                cuts.push(i);
            }
        }

        Sections::SplitCuts(split_cuts) => {
            cuts.append(&mut split_cuts.clone());
            cuts.push(len);
        }
    }

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

pub fn vsplit<D: Dtype>(
    tensor: impl Into<Tensor<D>>, 
    sections: impl IntoSections,
) -> Vec<Tensor<D>> {
    split(tensor, sections, 0)
}

pub fn hsplit<D: Dtype>(
    tensor: impl Into<Tensor<D>>, 
    sections: impl IntoSections,
) -> Vec<Tensor<D>> {
    split(tensor, sections, 1)
}

pub fn dsplit<D: Dtype>(
    tensor: impl Into<Tensor<D>>, 
    sections: impl IntoSections,
) -> Vec<Tensor<D>> {
    split(tensor, sections, 2)
}

fn split_by_axis<D: Dtype>(
    tensor: Tensor<D>, 
    axis: usize, 
    cuts: Vec<usize>
) -> Vec<Tensor<D>> {
    let n_outer = tensor.shape().sublen(0, axis);
    let n_inner = if axis < tensor.rank() {
        tensor.shape().sublen(axis + 1, tensor.rank())
    } else {
        1
    };
    let axis_len = tensor.shape().dim(axis);

    let mut prev = 0;
    let x = tensor.as_slice();

    let mut slices = Vec::<Tensor<D>>::new();

    for cut in cuts {
        if cut == prev {
            continue;
        }

        unsafe {
            let mut uninit = TensorUninit::<D>::new(n_outer * n_inner * (cut - prev));
            let o = uninit.as_mut_slice();

            for j in 0..n_outer {
                for k in prev..cut {
                    for i in 0..n_inner {
                        let v = x[j * axis_len * n_inner + k * n_inner + i].clone();

                        o[j * n_inner * (cut - prev) + (k - prev) * n_inner + i] = v;
                    }
                }
            }

            let mut shape = Vec::<usize>::new();
            for i in 0..axis {
                shape.push(tensor.shape().dim(i))
            }
            shape.push(cut - prev);
            for i in axis + 1..tensor.shape().rank() {
                shape.push(tensor.shape().dim(i))
            }

            slices.push(uninit.into().into_tensor(shape));
        }

        prev = cut;
    }


    slices
}

impl<D: Dtype> Tensor<D> {
    #[inline]
    pub fn split(&self, sections: impl IntoSections, axis: impl Into<AxisOpt>) -> Vec<Tensor<D>> {
        split(self, sections, axis)
    }

    #[inline]
    pub fn vsplit(&self, sections: impl IntoSections) -> Vec<Tensor<D>> {
        split(self, sections, 0)
    }

    #[inline]
    pub fn hsplit(&self, sections: impl IntoSections) -> Vec<Tensor<D>> {
        split(self, sections, 1)
    }

    #[inline]
    pub fn dsplit(&self, sections: impl IntoSections) -> Vec<Tensor<D>> {
        split(self, sections, 2)
    }
}

pub trait IntoSections {
    fn into_sections(self) -> Sections;
}

impl IntoSections for usize {
    fn into_sections(self) -> Sections {
        Sections::SplitEqual(self)
    }
}

impl<const N: usize> IntoSections for [usize; N] {
    fn into_sections(self) -> Sections {
        Sections::SplitCuts(Vec::from(self))
    }
}

impl IntoSections for &[usize] {
    fn into_sections(self) -> Sections {
        Sections::SplitCuts(Vec::from(self))
    }
}

impl IntoSections for Vec<usize> {
    fn into_sections(self) -> Sections {
        Sections::SplitCuts(self)
    }
}

impl IntoSections for &Vec<usize> {
    fn into_sections(self) -> Sections {
        Sections::SplitCuts(self.clone())
    }
}

#[derive(Clone)]
pub enum Sections {
    SplitEqual(usize),
    SplitCuts(Vec<usize>),
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
