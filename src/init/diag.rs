use crate::{tensor::TensorUninit, Tensor};


pub fn diagflat(vec: impl Into<Tensor>) -> Tensor {
    let vec : Tensor = vec.into();

    assert!(vec.rank() == 1, "diagflat currently expects a 1d vector {:?}", vec.shape().as_slice());
    let n = vec.len();
    let size = n * n;

    unsafe {
        let mut uninit = TensorUninit::<f32>::new(size);

        for item in uninit.as_mut_slice() {
            *item = 0.;
        }

        let slice = uninit.as_mut_slice();
        for (i, value) in vec.iter().enumerate() {
            slice[i * n + i] = *value;
        }

        uninit.into_tensor([n, n])
    }
}

impl Tensor {
    pub fn diagflat(&self) -> Tensor {
        diagflat(self)
    }
}

#[cfg(test)]
mod test {
    use crate::{init::diagflat, tf32};

    #[test]
    fn test_diagflat() {
        assert_eq!(
            diagflat(tf32!([1., 2., 3.])), tf32!([
            [1., 0., 0.],
            [0., 2., 0.],
            [0., 0., 3.],
        ]));
    }
}