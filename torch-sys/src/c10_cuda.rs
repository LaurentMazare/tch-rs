extern "C" {
    /// empty cuda cache
    pub fn emptyCache();
}

pub fn empty_cuda_cache() {
    unsafe { emptyCache() };
}