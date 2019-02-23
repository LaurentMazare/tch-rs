#[derive(Debug)]
pub struct TorchError {
    c_error: String,
}

extern "C" {
    fn get_and_reset_last_err() -> *mut libc::c_char;
}

pub fn read_and_clean_error() -> Result<(), TorchError> {
    unsafe {
        let torch_last_err = get_and_reset_last_err();
        if !torch_last_err.is_null() {
            let c_error = std::ffi::CStr::from_ptr(torch_last_err)
                .to_string_lossy()
                .into_owned();
            libc::free(torch_last_err as *mut libc::c_void);
            Err(TorchError { c_error })
        } else {
            Ok(())
        }
    }
}

macro_rules! unsafe_torch {
    ($e:expr) => {{
        let v = unsafe { $e };
        read_and_clean_error().unwrap();
        v
    }};
}
