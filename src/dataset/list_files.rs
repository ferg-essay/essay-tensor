use glob::glob;

use crate::{Tensor, flow::{Source, self, Out}};

use super::dataset::Dataset;

pub fn _list_files(glob: &str) -> Dataset<Tensor<String>> {
    let glob = glob.to_string();

    Dataset::from_flow(|builder| {
        builder.source(move || {
            _ListFiles::_new(&glob)
        }, &())
    })
}

struct _ListFiles {
    file_list: Tensor<String>,
    index: usize,
}

impl _ListFiles {
    fn _new(pattern: &str) -> Self {
        let file_list : Tensor<String> = glob(pattern).unwrap()
            .into_iter()
            .map(|p| p.unwrap().into_os_string().to_string_lossy().to_string())
            .collect();

        Self {
            file_list,
            index: 0,
        }
    }
}

impl Source<(), Tensor<String>> for _ListFiles {
    fn next(&mut self, _input: &mut ()) -> flow::Result<Out<Tensor<String>>> {
        if self.index < self.file_list.len() {
            let value = self.file_list.slice(self.index);

            self.index += 1;

            Ok(Out::Some(value))
        } else {
            Ok(Out::None)
        }
    }
}


#[cfg(test)]
mod test {
    use crate::prelude::*;

    use super::_list_files;

    #[test]
    fn test_file_list() {
        let mut data = _list_files("../../assets/audio/book-24/*.wav");

        let values = data.iter().collect::<Vec<Tensor<String>>>();
        assert_eq!(values[0][0], "../../assets/audio/book-24/237-0000.wav");
        assert_eq!(values[1][0], "../../assets/audio/book-24/237-0001.wav");
        assert_eq!(values[2][0], "../../assets/audio/book-24/237-0002.wav");
        assert_eq!(values[3][0], "../../assets/audio/book-24/237-0003.wav");
    }
}