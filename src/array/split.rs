use crate::tensor::{Axis, Type, Tensor, unsafe_init};

impl<D: Type + Clone> Tensor<D> {
    #[inline]
    pub fn split(&self, sections: impl IntoSections) -> Vec<Tensor<D>> {
        split_axis(None, self, sections)
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

fn split_axis<T: Type + Clone>(
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
            assert!(step > 0);
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

fn split_by_axis<T: Type + Clone>(
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

        let mut shape = Vec::<usize>::new();
        for i in 0..axis {
            shape.push(tensor.shape().dim(i))
        }
        shape.push(cut - prev);
        for i in axis + 1..tensor.shape().rank() {
            shape.push(tensor.shape().dim(i))
        }

        let data = unsafe {
            unsafe_init::<T>(n_outer * n_inner * (cut - prev), shape, |o| {
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

        slices.push(data);

        prev = cut;
    }


    slices
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
    use crate::ten;

    #[test]
    fn test_split() {
        assert_eq!(
            ten![[1., 2.], [3., 4.]].split(2), 
            vec![ten![[1., 2.]], ten![[3., 4.]]],
        );

        assert_eq!(
            ten![1., 2., 3., 4.].split([1, 3]), 
            vec![ten![1.], ten![2., 3.], ten![4.]],
        );
    }
    
    #[test]
    fn test_split_axis() {
        assert_eq!(
            ten![[1., 2.], [3., 4.]].split_axis(1, 2), 
            vec![ten![[1.], [3.]], ten![[2.], [4.]]],
        );
    }
    
    #[test]
    fn test_vsplit() {
        assert_eq!(
            ten![[1., 2.], [3., 4.]].vsplit(2), 
            vec![ten![[1., 2.]], ten![[3., 4.]]],
        );

        assert_eq!(
            ten![1., 2., 3., 4.].vsplit([1, 3]), 
            vec![ten![1.], ten![2., 3.], ten![4.]],
        );
    }
    
    #[test]
    fn test_hsplit() {
        assert_eq!(
            ten![[1., 2.], [3., 4.]].hsplit(2), 
            vec![ten![[1.], [3.]], ten![[2.], [4.]]],
        );
    }
    
    #[test]
    fn test_dsplit() {
        assert_eq!(
            ten![[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]].dsplit(2), 
            vec![
                ten![[[1.], [3.]], [[5.], [7.]]],
                ten![[[2.], [4.]], [[6.], [8.]]]
            ],
        );
    }
}
