use std::ops::Index;

use crate::tensor::Tensor;

use super::Type;


impl<T: Type> Index<usize> for Tensor<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.size(), "tensor[{}] is larger than size={}", index, self.size);

        unsafe {
            self.as_ptr().add(index).as_ref().unwrap()
        }
    }
}

impl<T: Type> Index<(usize, usize)> for Tensor<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let shape = self.shape();
        let len = shape.rank();

        assert!(len > 1);
        assert!(index.0 < shape.rdim(1));
        assert!(index.1 < shape.rdim(0));

        let offset = index.1 + shape.rdim(0) * index.0;

        assert!(offset < self.size());

        &self[offset]
    }
}

impl<T: Type> Index<(usize, usize, usize)> for Tensor<T> {
    type Output = T;

    fn index(&self, index: (usize, usize, usize)) -> &Self::Output {
        let shape = self.shape();
        let len = shape.size();

        assert!(len > 2);
        assert!(index.0 < shape.rdim(2));
        assert!(index.1 < shape.rdim(1));
        assert!(index.2 < shape.rdim(0));

        let offset =
            index.2 + shape.rdim(0) * (
                index.1 + shape.rdim(1) * index.0
            );

        // assert!(offset < self.len());

        &self[offset]
    }
}

impl<T: Type> Index<(usize, usize, usize, usize)> for Tensor<T> {
    type Output = T;

    fn index(&self, index: (usize, usize, usize, usize)) -> &Self::Output {
        let shape = self.shape();
        let len = shape.size();

        assert!(len > 3);
        assert!(index.0 < shape.rdim(3));
        assert!(index.1 < shape.rdim(2));
        assert!(index.2 < shape.rdim(1));
        assert!(index.3 < shape.rdim(0));

        let offset =
            index.3 + shape.rdim(0) * (
                index.2 + shape.rdim(1) * (
                    index.1 + shape.rdim(2) * index.0
                )
            );

        // assert!(offset < self.len());

        &self[offset]
    }
}

impl<T: Type> Index<(usize, usize, usize, usize, usize)> for Tensor<T> {
    type Output = T;

    fn index(&self, index: (usize, usize, usize, usize, usize)) -> &Self::Output {
        let shape = self.shape();
        let len = shape.rank();

        assert!(len > 4);
        assert!(index.0 < shape.rdim(4));
        assert!(index.1 < shape.rdim(3));
        assert!(index.2 < shape.rdim(2));
        assert!(index.3 < shape.rdim(1));
        assert!(index.4 < shape.rdim(0));

        let offset =
            index.4 + shape.rdim(0) * (
                index.3 + shape.rdim(1) * (
                    index.2 + shape.rdim(2) * (
                        index.1 + shape.rdim(3) * index.0
                    )
                )
            );

        // assert!(offset < self.len());

        &self[offset]
    }
}


#[cfg(test)]
mod test {
    use crate::tf32;

    #[test]
    fn index_rank_2() {
        let t = tf32!([[1., 2.], [3., 4.], [5., 6.]]);
        assert_eq!(t.shape().as_vec(), &[3, 2]);

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
        assert_eq!(t.shape().as_vec(), &[4]);

        assert_eq!(t[0], 1.);
        assert_eq!(t[1], 2.);
        assert_eq!(t[2], 3.);
        assert_eq!(t[3], 4.);
    }

    #[test]
    fn index_rank_1_slice() {
        let t = tf32!([1., 2., 3., 4.]);
        assert_eq!(t.shape().as_vec(), &[4]);

        assert_eq!(t.slice(0)[0], 1.);
        assert_eq!(t.slice(1)[0], 2.);
        assert_eq!(t.slice(2)[0], 3.);
        assert_eq!(t.slice(3)[0], 4.);
    }
}
