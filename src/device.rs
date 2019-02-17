pub enum Device {
    Cpu,
    Cuda,
}

impl Device {
    pub(crate) fn c_int(&self) -> libc::c_int {
        match self {
            Device::Cpu => 0,
            Device::Cuda => 1,
        }
    }
}
