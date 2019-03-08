//! Utility functions to manipulate images.
use crate::utils::path_to_cstring;
use crate::Tensor;
use failure::Fallible;
use libc::c_int;
use std::path::Path;

/// On success returns a tensor of shape [width, height, channels].
fn load_hwc<T: AsRef<Path>>(path: T) -> Fallible<Tensor> {
    let path = path_to_cstring(path)?;
    let c_tensor = unsafe_torch_err!({ torch_sys::at_load_image(path.as_ptr()) });
    Ok(Tensor { c_tensor })
}

/// Expects a tensor of shape [width, height, channels].
fn save_hwc<T: AsRef<Path>>(t: &Tensor, path: T) -> Fallible<()> {
    let path = path_to_cstring(path)?;
    let _ = unsafe_torch_err!({ torch_sys::at_save_image(t.c_tensor, path.as_ptr()) });
    Ok(())
}

/// Expects a tensor of shape [width, height, channels].
/// On success returns a tensor of shape [width, height, channels].
fn resize_hwc(t: &Tensor, out_w: i64, out_h: i64) -> Fallible<Tensor> {
    let c_tensor = unsafe_torch_err!({
        torch_sys::at_resize_image(t.c_tensor, out_w as c_int, out_h as c_int)
    });
    Ok(Tensor { c_tensor })
}

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
    save_hwc(&chw_to_hwc(t), path)
}

/// Resizes an image.
///
/// This expects as input a tensor of shape [channel, height, width] and returns
/// a tensor of shape [channel, out_h, out_w].
pub fn resize(t: &Tensor, out_w: i64, out_h: i64) -> Fallible<Tensor> {
    Ok(hwc_to_chw(&resize_hwc(&chw_to_hwc(t), out_w, out_h)?))
}

pub fn load_and_resize<T: AsRef<Path>>(path: T, out_w: i64, out_h: i64) -> Fallible<Tensor> {
    let tensor = load_hwc(path)?;
    Ok(hwc_to_chw(&resize_hwc(&tensor, out_w, out_h)?))
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
