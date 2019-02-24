use cmake::Config;
use std::env;
use std::path::PathBuf;

fn main() {
    let libtorch = env::var("LIBTORCH").expect("LIBTORCH not defined");
    let libtorch = PathBuf::from(libtorch);
    let libtorch_lib = libtorch.join("lib");
    let dst = Config::new("libtch")
        .define("CMAKE_PREFIX_PATH", libtorch)
        .build();
    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=static=tch");
    println!(
        "cargo:rustc-link-search=native={}",
        libtorch_lib.to_string_lossy().into_owned()
    );
    println!("cargo:rustc-link-lib=c10");
    println!("cargo:rustc-link-lib=caffe2");
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=stdc++");
}
