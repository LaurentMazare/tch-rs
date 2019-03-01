use crate::utils::{path_to_str, TorchError};
use crate::Tensor;
use crate::tensor::C_tensor;
use libc::{c_char, c_int};

extern "C" {
    fn at_save_image(arg: *mut C_tensor, filename: *const c_char) -> c_int;
    fn at_load_image(filename: *const c_char) -> *mut C_tensor;
    fn at_resize_image(arg: *mut C_tensor, out_w: c_int, out_h: c_int) -> *mut C_tensor;
}

/// On success returns a tensor of shape [width, height, channels].
fn load_whc(path: &std::path::Path) -> Result<Tensor, TorchError> {
    let path = std::ffi::CString::new(path_to_str(path)?)?;
    let c_tensor = unsafe_torch_err!({ at_load_image(path.as_ptr()) });
    Ok(Tensor { c_tensor })
}

/// Expects a tensor of shape [width, height, channels].
fn save_whc(t: &Tensor, path: &std::path::Path) -> Result<(), TorchError> {
    let path = std::ffi::CString::new(path_to_str(path)?)?;
    let _ = unsafe_torch_err!({ at_save_image(t.c_tensor, path.as_ptr()) });
    Ok(())
}

/// Expects a tensor of shape [width, height, channels].
/// On success returns a tensor of shape [width, height, channels].
fn resize_whc(t: &Tensor, out_w: i64, out_h: i64) -> Tensor {
    let c_tensor = unsafe_torch!({ at_resize_image(t.c_tensor, out_w as c_int, out_h as c_int) });
    Tensor { c_tensor }
}

pub fn load(path: &std::path::Path) -> Result<Tensor, TorchError> {
    let tensor = load_whc(path)?;
    Ok(tensor.transpose(2, 0))
}

pub fn save(t: &Tensor, path: &std::path::Path) -> Result<(), TorchError> {
    save_whc(&t.transpose(2, 0), path)
}

pub fn resize(t: &Tensor, out_w: i64, out_h: i64) -> Tensor {
    resize_whc(&t.transpose(2, 0), out_w, out_h).transpose(2, 0)
}
