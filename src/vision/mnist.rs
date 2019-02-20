use crate::{Kind, Tensor};
use std::fs::File;
use std::io::{self, BufReader, Read, Result};

fn read_u32<T: Read>(reader: &mut T) -> Result<u32> {
    let mut b = vec![0u8; 4];
    reader.read_exact(&mut b)?;
    let (result, _) = b.iter().rev().fold((0u64, 1u64), |(s, basis), &x| {
        (s + basis * u64::from(x), basis * 256)
    });
    Ok(result as u32)
}

fn check_magic_number<T: Read>(reader: &mut T, expected: u32) -> Result<()> {
    let magic_number = read_u32(reader)?;
    if magic_number != expected {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("incorrect magic number {} != {}", magic_number, expected),
        ));
    }
    Ok(())
}

fn read_labels(filename: &std::path::Path) -> Result<Tensor> {
    let mut buf_reader = BufReader::new(File::open(filename)?);
    check_magic_number(&mut buf_reader, 2049)?;
    let samples = read_u32(&mut buf_reader)?;
    let mut data = vec![0u8; samples as usize];
    buf_reader.read_exact(&mut data)?;
    Ok(Tensor::of_data(&data, &Kind::Uint8).to_kind(&Kind::Int64))
}

fn read_images(filename: &std::path::Path) -> Result<Tensor> {
    let mut buf_reader = BufReader::new(File::open(filename)?);
    check_magic_number(&mut buf_reader, 2051)?;
    let samples = read_u32(&mut buf_reader)?;
    let rows = read_u32(&mut buf_reader)?;
    let cols = read_u32(&mut buf_reader)?;
    let data_len = samples * rows * cols;
    let mut data = vec![0u8; data_len as usize];
    buf_reader.read_exact(&mut data)?;
    let tensor = Tensor::of_data(&data, &Kind::Uint8)
        .view(&[i64::from(samples), i64::from(rows * cols)])
        .to_kind(&Kind::Float);
    Ok(tensor / 255.)
}

pub struct Mnist {
    pub train_images: Tensor,
    pub train_labels: Tensor,
    pub test_images: Tensor,
    pub test_labels: Tensor,
}

impl Mnist {
    pub fn load_dir(dir: &std::path::Path) -> Result<Mnist> {
        let train_images = read_images(&dir.join("train-images-idx3-ubyte"))?;
        let train_labels = read_labels(&dir.join("train-labels-idx1-ubyte"))?;
        let test_images = read_images(&dir.join("t10k-images-idx3-ubyte"))?;
        let test_labels = read_labels(&dir.join("t10k-labels-idx1-ubyte"))?;
        Ok(Mnist {
            train_images,
            train_labels,
            test_images,
            test_labels,
        })
    }
}
