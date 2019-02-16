use std::env;
use std::path::PathBuf;
extern crate cc;

fn main() {
    let libtorch = env::var("LIBTORCH").expect("LIBTORCH not defined");
    let libtorch = PathBuf::from(libtorch);
    let libtorch_lib = libtorch.join("lib").into_os_string().into_string().unwrap();
    println!("cargo:rustc-link-search=native={}", libtorch_lib);
    println!("cargo:rustc-link-lib=c10");
    println!("cargo:rustc-link-lib=caffe2");
    println!("cargo:rustc-link-lib=torch");
    cc::Build::new()
        .cpp(true)
        .include(libtorch.join("include"))
        .include(libtorch.join("include/torch/csrc/api/include"))
        .flag(&format!("-Wl,-rpath={}", libtorch_lib))
        .flag("-std=c++11")
        .flag("-D_GLIBCXX_USE_CXX11_ABI=0")
        .file("c/torch_api.cpp")
        .warnings(false)
        .compile("libtorch");
}
