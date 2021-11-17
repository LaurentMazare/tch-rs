// The LIBTORCH environment variable can be used to specify the directory
// where libtorch has been installed.
// When not specified this script downloads the cpu version for libtorch
// and extracts it in OUT_DIR.
//
// On Linux, the TORCH_CUDA_VERSION environment variable can be used,
// like 9.0, 90, or cu90 to specify the version of CUDA to use for libtorch.

use anyhow::Context;
use std::env;
use torch_build::{LinkType, SystemInfo};

fn main() -> anyhow::Result<()> {
    if !cfg!(feature = "doc-only") {
        let system_info = SystemInfo::new()?;
        // use_cuda is a hacky way to detect whether cuda is available and
        // if it's the case link to it by explicitly depending on a symbol
        // from the torch_cuda library.
        // It would be better to use -Wl,--no-as-needed but there is no way
        // to specify arbitrary linker flags at the moment.
        //
        // Once https://github.com/rust-lang/cargo/pull/8441 is available
        // we should switch to using rustc-link-arg instead e.g. with the
        // following flags:
        //   -Wl,--no-as-needed -Wl,--copy-dt-needed-entries -ltorch
        //
        // This will be available starting from cargo 1.50 but will be a nightly
        // only option to start with.
        // https://github.com/rust-lang/cargo/blob/master/CHANGELOG.md
        //
        // Update: The above doesn't seem to propagate to the downstream binaries
        // so doesn't really help, the comment has been kept though to keep track
        // if this issue.
        // TODO: Try out the as-needed native link modifier when it lands.
        // https://github.com/rust-lang/rust/issues/99424
        let si_lib = &system_info.libtorch_lib_dir;
        let use_cuda =
            si_lib.join("libtorch_cuda.so").exists() || si_lib.join("torch_cuda.dll").exists();
        let use_cuda_cu = si_lib.join("libtorch_cuda_cu.so").exists()
            || si_lib.join("torch_cuda_cu.dll").exists();
        let use_cuda_cpp = si_lib.join("libtorch_cuda_cpp.so").exists()
            || si_lib.join("torch_cuda_cpp.dll").exists();
        let use_hip =
            si_lib.join("libtorch_hip.so").exists() || si_lib.join("torch_hip.dll").exists();
        println!("cargo:rustc-link-search=native={}", si_lib.display());

        system_info.make(use_cuda, use_hip);

        println!("cargo:rustc-link-lib=static=tch");
        if use_cuda {
            system_info.link("torch_cuda")
        }
        if use_cuda_cu {
            system_info.link("torch_cuda_cu")
        }
        if use_cuda_cpp {
            system_info.link("torch_cuda_cpp")
        }
        if use_hip {
            system_info.link("torch_hip")
        }
        if cfg!(feature = "python-extension") {
            system_info.link("torch_python")
        }
        if system_info.link_type == LinkType::Static {
            // TODO: this has only be tried out on the cpu version. Check that it works
            // with cuda too and maybe just try linking all available files?
            system_info.link("asmjit");
            system_info.link("clog");
            system_info.link("cpuinfo");
            system_info.link("dnnl");
            system_info.link("dnnl_graph");
            system_info.link("fbgemm");
            system_info.link("gloo");
            system_info.link("kineto");
            system_info.link("nnpack");
            system_info.link("onnx");
            system_info.link("onnx_proto");
            system_info.link("protobuf");
            system_info.link("pthreadpool");
            system_info.link("pytorch_qnnpack");
            system_info.link("sleef");
            system_info.link("tensorpipe");
            system_info.link("tensorpipe_uv");
            system_info.link("XNNPACK");
        }
        system_info.link("torch_cpu");
        system_info.link("torch");
        system_info.link("c10");
        if use_hip {
            system_info.link("c10_hip");
        }

        let target = env::var("TARGET").context("TARGET variable not set")?;

        if !target.contains("msvc") && !target.contains("apple") {
            println!("cargo:rustc-link-lib=gomp");
        }
    }
    Ok(())
}
