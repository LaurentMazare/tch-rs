pub const TORCH_VERSION: &str = "2.0.0";
pub const PYTHON_PRINT_PYTORCH_DETAILS: &str = r"
import torch
from torch.utils import cpp_extension
print('LIBTORCH_VERSION:', torch.__version__.split('+')[0])
print('LIBTORCH_CXX11:', torch._C._GLIBCXX_USE_CXX11_ABI)
for include_path in cpp_extension.include_paths():
  print('LIBTORCH_INCLUDE:', include_path)
for library_path in cpp_extension.library_paths():
  print('LIBTORCH_LIB:', library_path)
";

pub const PYTHON_PRINT_INCLUDE_PATH: &str = r"
import sysconfig
print('PYTHON_INCLUDE:', sysconfig.get_path('include'))
";

pub const NO_DOWNLOAD_ERROR_MESSAGE: &str = r"
Cannot find a libtorch install, you can either:
- Install libtorch manually and set the LIBTORCH environment variable to appropriate path.
- Use a system wide install in /usr/lib/libtorch.so.
- Use a Python environment with PyTorch installed by setting LIBTORCH_USE_PYTORCH=1

See the readme for more details:
https://github.com/LaurentMazare/tch-rs/blob/main/README.md
";
