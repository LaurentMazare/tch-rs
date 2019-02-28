use cmake::Config;
use std::env;
use std::path::PathBuf;

fn make(libtorch: &PathBuf, libtorch_lib: &PathBuf) {
    let libtorch_cxx11_abi = env::var("LIBTORCH_CXX11_ABI").unwrap_or("0".to_string());
    cc::Build::new()
        .cpp(true)
        .include(libtorch.join("include"))
        .include(libtorch.join("include/torch/csrc/api/include"))
        .flag(&format!(
            "-Wl,-rpath={}",
            libtorch_lib.to_string_lossy().into_owned()
        ))
        .flag("-std=c++11")
        .flag(&format!("-D_GLIBCXX_USE_CXX11_ABI={}", libtorch_cxx11_abi))
        .file("libtch/torch_api.cpp")
        .warnings(false)
        .compile("libtorch");
}

fn cmake(libtorch: &PathBuf) {
    let dst = Config::new("libtch")
        .define("CMAKE_PREFIX_PATH", libtorch)
        .build();
    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=static=tch");
    println!("cargo:rustc-link-lib=stdc++");
}

fn main() {
    let libtorch = env::var("LIBTORCH").expect("LIBTORCH not defined");
    let libtorch = PathBuf::from(libtorch);
    let libtorch_lib = libtorch.join("lib");
    println!(
        "cargo:rustc-link-search=native={}",
        libtorch_lib.to_string_lossy().into_owned()
    );
    if env::var("LIBTORCH_USE_CMAKE").is_ok() {
        cmake(&libtorch)
    } else {
        make(&libtorch, &libtorch_lib)
    }
    println!("cargo:rustc-link-lib=c10");
    println!("cargo:rustc-link-lib=caffe2");
    println!("cargo:rustc-link-lib=torch");
}
