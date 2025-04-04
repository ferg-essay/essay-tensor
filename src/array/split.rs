use crate::tensor::{Axis, Dtype, Tensor, TensorData};

// TODO: split can't currenly be written as an operation.
pub fn split<D: Dtype>(
    tensor: impl Into<Tensor<D>>, 
    sections: impl IntoSections,
) -> Vec<Tensor<D>> {
    split_axis(None, tensor, sections)
}

// TODO: split can't currenly be written as an operation.
pub fn split_axis<T: Clone + 'static>(
    axis: impl Into<Axis>,
    tensor: impl Into<Tensor<T>>, 
    sections: impl IntoSections,
) -> Vec<Tensor<T>> {
    let tensor = tensor.into();
    let section = sections.into_sections();
    let axis : Axis = axis.into();

    let axis = axis.axis_from_rank(tensor.rank());

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
        let mut slices = Vec::<Tensor<T>>::new();

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
    split_axis(Axis::axis(0), tensor, sections)
}

pub fn hsplit<D: Dtype>(
    tensor: impl Into<Tensor<D>>, 
    sections: impl IntoSections,
) -> Vec<Tensor<D>> {
    split_axis(Axis::axis(0), tensor, sections)
}

pub fn dsplit<D: Dtype>(
    tensor: impl Into<Tensor<D>>, 
    sections: impl IntoSections,
) -> Vec<Tensor<D>> {
    split_axis(Axis::axis(1), tensor, sections)
}

fn split_by_axis<T: Clone + 'static>(
    tensor: Tensor<T>, 
    axis: usize, 
    cuts: Vec<usize>
) -> Vec<Tensor<T>> {
    let n_outer = tensor.shape().sublen(0, axis);
    let n_inner = if axis < tensor.rank() {
        tensor.shape().sublen(axis + 1, tensor.rank())
    } else {
        1
    };
    let axis_len = tensor.shape().dim(axis);

    let mut prev = 0;
    let x = tensor.as_slice();

    let mut slices = Vec::<Tensor<T>>::new();

    for cut in cuts {
        if cut == prev {
            continue;
        }

        let data = unsafe {
            TensorData::<T>::unsafe_init(n_outer * n_inner * (cut - prev), |o| {
                for j in 0..n_outer {
                    for k in prev..cut {
                        for i in 0..n_inner {
                            let v = x[j * axis_len * n_inner + k * n_inner + i].clone();

                            o.add(j * n_inner * (cut - prev) + (k - prev) * n_inner + i)
                                .write(v);
                        }
                    }
                }
            })
        };

        let mut shape = Vec::<usize>::new();
        for i in 0..axis {
            shape.push(tensor.shape().dim(i))
        }
        shape.push(cut - prev);
        for i in axis + 1..tensor.shape().rank() {
            shape.push(tensor.shape().dim(i))
        }

        slices.push(data.into_tensor(shape));

        prev = cut;
    }


    slices
}

impl<D: Dtype> Tensor<D> {
    #[inline]
    pub fn split(&self, sections: impl IntoSections) -> Vec<Tensor<D>> {
        split(self, sections)
    }

    #[inline]
    pub fn split_axis(&self, axis: impl Into<Axis>, sections: impl IntoSections) -> Vec<Tensor<D>> {
        split_axis(axis, self, sections)
    }

    #[inline]
    pub fn vsplit(&self, sections: impl IntoSections) -> Vec<Tensor<D>> {
        split_axis(Axis::axis(0), self, sections)
    }

    #[inline]
    pub fn hsplit(&self, sections: impl IntoSections) -> Vec<Tensor<D>> {
        split_axis(Axis::axis(1), self, sections)
    }

    #[inline]
    pub fn dsplit(&self, sections: impl IntoSections) -> Vec<Tensor<D>> {
        split_axis(Axis::axis(2), self, sections)
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
    use crate::array::split::{vsplit, hsplit, dsplit};

    use crate::{array::split::{split, split_axis}, prelude::*};
    
    #[test]
    fn test_split() {
        assert_eq!(
            split(&tf32!([[1., 2.], [3., 4.]]), 2), 
            vec![tf32!([[1., 2.]]), tf32!([[3., 4.]])],
        );

        assert_eq!(
            split(&tf32!([1., 2., 3., 4.]), [1, 3]), 
            vec![tf32!([1.]), tf32!([2., 3.]), tf32!([4.])],
        );
    }
    
    #[test]
    fn test_split_axis() {
        assert_eq!(
            split_axis(Axis::axis(1), &tf32!([[1., 2.], [3., 4.]]), 2), 
            vec![tf32!([[1.], [3.]]), tf32!([[2.], [4.]])],
        );
    }
    
    #[test]
    fn test_vsplit() {
        assert_eq!(
            vsplit(tf32!([[1., 2.], [3., 4.]]), 2), 
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
