use super::dataset::Dataset;
use crate::{kind, Tensor};
use std::fs::File;
use std::io::{BufReader, Read, Result};

static W: i64 = 32;
static H: i64 = 32;
static C: i64 = 3;
static BYTES_PER_IMAGE: i64 = W * H * C + 1;
static SAMPLES_PER_FILE: i64 = 10000;

fn read_file(filename: &std::path::Path) -> Result<(Tensor, Tensor)> {
    let mut buf_reader = BufReader::new(File::open(filename)?);
    let mut data = vec![0u8; (SAMPLES_PER_FILE * BYTES_PER_IMAGE) as usize];
    buf_reader.read_exact(&mut data)?;
    let content = Tensor::of_data(&data, kind::Kind::Uint8);
    let images = Tensor::zeros(&[SAMPLES_PER_FILE, C, H, W], kind::FLOAT_CPU);
    let labels = Tensor::zeros(&[SAMPLES_PER_FILE], kind::INT64_CPU);
    for index in 0..SAMPLES_PER_FILE {
        let content_offset = BYTES_PER_IMAGE * index;
        labels
            .narrow(0, index, 1)
            .copy_(&content.narrow(0, content_offset, 1));
        images.narrow(0, index, 1).copy_(
            &content
                .narrow(0, 1 + content_offset, BYTES_PER_IMAGE - 1)
                .view(&[1, C, H, W])
                .to_kind(kind::Kind::Float),
        );
    }
    Ok((images.to_kind(kind::Kind::Float) / 255.0, labels))
}

pub fn load_dir(dir: &std::path::Path) -> Result<Dataset> {
    let (test_images, test_labels) = read_file(&dir.join("test_batch.bin"))?;
    let train_images_and_labels = [
        "data_batch_1.bin",
        "data_batch_2.bin",
        "data_batch_3.bin",
        "data_batch_4.bin",
        "data_batch_5.bin",
    ]
    .iter()
    .map(|x| read_file(&dir.join(x)))
    .collect::<Result<Vec<_>>>()?;
    let train_images: Vec<_> = train_images_and_labels.iter().map(|x| &x.0).collect();
    let train_labels: Vec<_> = train_images_and_labels.iter().map(|x| &x.1).collect();
    Ok(Dataset {
        train_images: Tensor::cat(&train_images, 0),
        train_labels: Tensor::cat(&train_labels, 0),
        test_images,
        test_labels,
        labels: 10,
    })
}
