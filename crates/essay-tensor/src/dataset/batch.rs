use crate::Tensor;

use super::Dataset;


pub fn rebatch(tensor: impl Into<Tensor>, batch_size: usize) -> Dataset<Tensor> {
    let tensor = tensor.into();
    let len = tensor.len();

    let mut vec : Vec<Tensor> = Vec::new();

    let mut offset = 0;
    while offset + batch_size <= len {
        vec.push(tensor.subslice(offset, batch_size));
        offset += batch_size;
    }

    if offset < len {
        vec.push(tensor.subslice(offset, len - offset));
    }

    vec.reverse();

    Dataset::from(vec)
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    use super::rebatch;
    
    #[test]
    fn test_batch() {
        let mut ds = rebatch(tf32!([1., 2., 3., 4., 5.]), 2);
        let vec: Vec<Tensor> = ds.iter().collect();
        assert_eq!(vec, vec![tf32!([1., 2.]), tf32!([3., 4.]), tf32!([5.])]);
    }
}
