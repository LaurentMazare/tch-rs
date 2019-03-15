//! Utility functions to manipulate images.
use crate::wrappers::image::{load_hwc, resize_hwc, save_hwc};
use crate::Tensor;
use failure::Fallible;
use std::path::Path;

fn hwc_to_chw(tensor: &Tensor) -> Tensor {
    tensor.permute(&[2, 0, 1])
}

fn chw_to_hwc(tensor: &Tensor) -> Tensor {
    tensor.permute(&[1, 2, 0])
}

/// Loads an image from a file.
///
/// On success returns a tensor of shape [channel, height, width].
pub fn load<T: AsRef<Path>>(path: T) -> Fallible<Tensor> {
    let tensor = load_hwc(path)?;
    Ok(hwc_to_chw(&tensor))
}

/// Saves an image to a file.
///
/// This expects as input a tensor of shape [channel, height, width].
/// The image format is based on the filename suffix, supported suffixes
/// are jpg, png, tga, and bmp.
pub fn save<T: AsRef<Path>>(t: &Tensor, path: T) -> Fallible<()> {
    match t.size().as_slice() {
        [1, _, _, _] => save_hwc(&chw_to_hwc(&t.squeeze1(0)), path),
        [_, _, _] => save_hwc(&chw_to_hwc(t), path),
        sz => bail!("unexpected size for image tensor {:?}", sz),
    }
}

/// Resizes an image.
///
/// This expects as input a tensor of shape [channel, height, width] and returns
/// a tensor of shape [channel, out_h, out_w].
pub fn resize(t: &Tensor, out_w: i64, out_h: i64) -> Fallible<Tensor> {
    Ok(hwc_to_chw(&resize_hwc(&chw_to_hwc(t), out_w, out_h)?))
}

/// Loads and resize an image, preserve the aspect ratio by taking a center crop.
pub fn load_and_resize<T: AsRef<Path>>(path: T, out_w: i64, out_h: i64) -> Fallible<Tensor> {
    let tensor = load_hwc(path)?;
    let tensor_size = tensor.size();
    let (w, h) = (tensor_size[0], tensor_size[1]);
    if w * out_h == h * out_w {
        Ok(hwc_to_chw(&resize_hwc(&tensor, out_w, out_h)?))
    } else {
        let (resize_w, resize_h) = {
            let ratio_w = out_w as f64 / w as f64;
            let ratio_h = out_h as f64 / h as f64;
            let ratio = ratio_w.max(ratio_h);
            ((ratio * h as f64) as i64, (ratio * w as f64) as i64)
        };
        let tensor = hwc_to_chw(&resize_hwc(&tensor, resize_w, resize_h)?);
        let tensor = if resize_w == out_w {
            tensor
        } else {
            tensor.f_narrow(2, (resize_w - out_w) / 2, out_w)?
        };
        let tensor = if resize_h == out_h {
            tensor
        } else {
            tensor.f_narrow(2, (resize_h - out_h) / 2, out_h)?
        };
        Ok(tensor)
    }
}

fn visit_dirs(dir: &Path, files: &mut Vec<std::fs::DirEntry>) -> Fallible<()> {
    if dir.is_dir() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit_dirs(&path, files)?;
            } else if entry
                .file_name()
                .to_str()
                .map_or(false, |s| s.ends_with(".png") || s.ends_with(".jpg"))
            {
                files.push(entry);
            }
        }
    }
    Ok(())
}

/// Loads all the images in a director.
pub fn load_dir<T: AsRef<Path>>(path: T, out_w: i64, out_h: i64) -> Fallible<Tensor> {
    let mut files: Vec<std::fs::DirEntry> = vec![];
    visit_dirs(&path.as_ref(), &mut files)?;
    ensure!(!files.is_empty(), "no image found in {:?}", path.as_ref());
    let v: Vec<_> = files
        .iter()
        // Silently discard image errors.
        .filter_map(|x| load_and_resize(x.path(), out_w, out_h).ok())
        .collect();
    Ok(Tensor::stack(&v, 0))
}
