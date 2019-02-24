use cmake::Config;
use std::env;
use std::path::PathBuf;

fn main() {
    let libtorch = env::var("LIBTORCH").expect("LIBTORCH not defined");
    let libtorch = PathBuf::from(libtorch);
    let libtorch_lib = libtorch.join("lib").into_os_string().into_string().unwrap();
    let dst = Config::new("libtch")
        .define("CMAKE_PREFIX_PATH", libtorch)
        .build();
    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=static=tch");
    println!("cargo:rustc-link-search=native={}", libtorch_lib);
    println!("cargo:rustc-link-lib=c10");
    println!("cargo:rustc-link-lib=caffe2");
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=stdc++");
}
