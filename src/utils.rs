extern "C" {
    fn get_and_reset_last_err() -> *mut libc::c_char;
}

pub(crate) fn read_and_clean_error() {
    unsafe {
        let torch_last_err = get_and_reset_last_err();
        if !torch_last_err.is_null() {
            let err = std::ffi::CStr::from_ptr(torch_last_err)
                .to_string_lossy()
                .into_owned();
            libc::free(torch_last_err as *mut libc::c_void);
            panic!(err)
        }
    }
}
