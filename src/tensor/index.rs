use std::ops::Index;

use crate::Tensor;


impl<T> Index<usize> for Tensor<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.len());

        unsafe {
            self.as_ptr().add(index).as_ref().unwrap()
        }
    }
}

impl<T> Index<(usize, usize)> for Tensor<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let shape = self.shape();
        let len = shape.rank();

        assert!(len > 1);
        assert!(index.0 < shape.idim(-2));
        assert!(index.1 < shape.idim(-1));

        let offset =
            index.1 + shape[len - 1] * index.0;

        assert!(offset < self.len());

        &self[offset]
    }
}

impl<T> Index<(usize, usize, usize)> for Tensor<T> {
    type Output = T;

    fn index(&self, index: (usize, usize, usize)) -> &Self::Output {
        let shape = self.shape();
        let len = shape.size();

        assert!(len > 2);
        assert!(index.0 < shape[len - 3]);
        assert!(index.1 < shape[len - 2]);
        assert!(index.2 < shape[len - 1]);

        let offset =
            index.2 + shape[len - 1] * (
                index.1 + shape[len - 2] * index.0
            );

        // assert!(offset < self.len());

        &self[offset]
    }
}

impl<T> Index<(usize, usize, usize, usize)> for Tensor<T> {
    type Output = T;

    fn index(&self, index: (usize, usize, usize, usize)) -> &Self::Output {
        let shape = self.shape();
        let len = shape.size();

        assert!(len > 3);
        assert!(index.0 < shape[len - 4]);
        assert!(index.1 < shape[len - 3]);
        assert!(index.2 < shape[len - 2]);
        assert!(index.3 < shape[len - 1]);

        let offset =
            index.3 + shape[len - 1] * (
                index.2 + shape[len - 2] * (
                    index.1 + shape[len - 3] * index.0
                )
            );

        // assert!(offset < self.len());

        &self[offset]
    }
}

impl<T> Index<(usize, usize, usize, usize, usize)> for Tensor<T> {
    type Output = T;

    fn index(&self, index: (usize, usize, usize, usize, usize)) -> &Self::Output {
        let shape = self.shape();
        let len = shape.size();

        assert!(len > 4);
        assert!(index.0 < shape[len - 5]);
        assert!(index.1 < shape[len - 4]);
        assert!(index.2 < shape[len - 3]);
        assert!(index.3 < shape[len - 2]);
        assert!(index.4 < shape[len - 1]);

        let offset =
            index.4 + shape[len - 1] * (
                index.3 + shape[len - 2] * (
                    index.2 + shape[len - 3] * (
                        index.1 + shape[len - 4] * index.0
                    )
                )
            );

        // assert!(offset < self.len());

        &self[offset]
    }
}


#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn index_rank_2() {
        let t = tf32!([[1., 2.], [3., 4.], [5., 6.]]);
        assert_eq!(t.shape().as_slice(), &[3, 2]);

        assert_eq!(t[0], 1.);
        assert_eq!(t[1], 2.);
        assert_eq!(t[2], 3.);

        assert_eq!(t[(0, 0)], 1.);
        assert_eq!(t[(0, 1)], 2.);
        assert_eq!(t[(1, 0)], 3.);
        assert_eq!(t[(1, 1)], 4.);
        assert_eq!(t[(2, 0)], 5.);
        assert_eq!(t[(2, 1)], 6.);
    }

    #[test]
    fn index_rank_1() {
        let t = tf32!([1., 2., 3., 4.]);
        assert_eq!(t.shape().as_slice(), &[4]);

        assert_eq!(t[0], 1.);
        assert_eq!(t[1], 2.);
        assert_eq!(t[2], 3.);
        assert_eq!(t[3], 4.);
    }

    #[test]
    fn index_rank_1_slice() {
        let t = tf32!([1., 2., 3., 4.]);
        assert_eq!(t.shape().as_slice(), &[4]);

        assert_eq!(t.slice(0)[0], 1.);
        assert_eq!(t.slice(1)[0], 2.);
        assert_eq!(t.slice(2)[0], 3.);
        assert_eq!(t.slice(3)[0], 4.);
    }
}
