use std::{fs::File, io::Read};

use crate::{Tensor, tensor::TensorUninit};

pub fn read_file(filename: impl Into<Tensor<String>>) -> Result<Tensor<u8>, std::io::Error> {
    let mut file = File::open(&filename.into()[0])?;
    let len = file.metadata()?.len();

    unsafe {
        let mut uninit = TensorUninit::<u8>::new(len as usize);

        file.read(uninit.as_mut_slice())?;

        Ok(Tensor::from_uninit(uninit, len as usize))
    }
}

#[cfg(test)]
mod test {
    use crate::Tensor;

    use super::read_file;

    #[test]
    fn test_read_file() {
        let data = read_file(Tensor::from("../../assets/text/quick-brown-fox.txt")).unwrap();

        assert_eq!(
            std::str::from_utf8(data.as_slice()).unwrap(),
            "The quick brown fox jumped over the lazy dog.\n"
        );
    }
}