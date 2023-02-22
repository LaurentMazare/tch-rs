use libc::c_int;

extern "C" {
    /// Returns true if Vulkan is available.
    pub fn atc_vulkan_is_available() -> c_int;
}
