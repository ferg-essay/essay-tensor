use crate::tensor::Tensor;

pub fn decode_wav(
    contents: impl Into<Tensor<u8>>
) -> (Tensor<f32>, Tensor<usize>) {
    let contents = contents.into();

    let mut cursor = Cursor::new(contents.as_slice());

    assert_eq!(cursor.read_u32_big(), 0x5249_4646); // TIFF

    assert_eq!(cursor.read_u32_little(), (contents.as_slice().len() - 8) as u32);
    assert_eq!(cursor.read_u32_big(), 0x5741_5645); // Wave

    assert_eq!(cursor.read_u32_big(), 0x666d_7420); // fmt
    assert_eq!(cursor.read_u32_little(), 16);

    let meta = Meta {
        fmt: cursor.read_u16_little(),
        n_channels: cursor.read_u16_little(),
        sample_rate: cursor.read_u32_little(),
        _byte_rate: cursor.read_u32_little(),
        _block_align: cursor.read_u16_little(),
        bits_per_sample: cursor.read_u16_little(),
    };

    assert_eq!(meta.fmt, 1);
    assert_eq!(meta.bits_per_sample, 16);

    //println!("Meta: {:?}", &meta);

    assert_eq!(cursor.read_u32_big(), 0x6461_7461);
    let size = cursor.read_u32_little();

    let n_samples = size / 2;
    //let len = meta.n_channels * 

    let mut data = Vec::new();
    data.reserve_exact(n_samples as usize);

    for _ in 0..n_samples as usize {
        let item = cursor.read_i16_little();
        data.push((item as f32) / (0x8000 as f32));
    }

    (
        Tensor::from_vec(data, [
            n_samples as usize / meta.n_channels as usize, 
            meta.n_channels as usize
        ]), 
        Tensor::from(meta.sample_rate as usize)
    )
}

#[derive(Debug)]
struct Meta {
    fmt: u16,
    n_channels: u16,
    sample_rate: u32,
    _byte_rate: u32,
    _block_align: u16,
    bits_per_sample: u16,
}

struct Cursor<'a> {
    slice: &'a [u8],
    index: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            slice: data,
            index: 0,
        }
    }

    fn read(&mut self) -> u8 {
        let v = self.slice[self.index];
        self.index += 1;
        v
    }

    fn _read_u16(&mut self) -> u16 {
        0x100 * self.read() as u16 + self.read() as u16
    }

    fn read_u16_little(&mut self) -> u16 {
        self.read() as u16 + 0x100 * self.read() as u16
    }

    fn read_i16_little(&mut self) -> i16 {
        (self.read() as u16 + 0x100 * self.read() as u16) as i16
    }

    fn read_u32_big(&mut self) -> u32 {
        0x0100_0000 * self.read() as u32 +
        0x01_0000 * self.read() as u32 +
        0x100 * self.read() as u32 +
        self.read() as u32
    }

    fn read_u32_little(&mut self) -> u32 {
        self.read() as u32 +
        0x0100 * self.read() as u32 +
        0x1_0000 * self.read() as u32 +
        0x100_0000 * self.read() as u32
    }

    fn _read_slice(&mut self, slice: &mut [u8]) {
        for i in 0..slice.len() {
            slice[i] = self.slice[i + self.index] 
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{io::{decode_wav::decode_wav, read_file}, ten, tensor::Shape};

    #[test]
    fn test_wav() {
        let data = read_file("../assets/audio/book-24/237-0001.wav").unwrap();

        let (audio, rate) = decode_wav(&data);
        assert_eq!(audio.shape(), &Shape::from([118080, 1]));
        assert_eq!(rate, ten!(16000));
    }
}