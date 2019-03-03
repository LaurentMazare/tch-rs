use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use cmake::Config;
use failure::Fallible;
use reqwest;
use zip;

const TORCH_VERSION: &'static str = "1.0.1";

fn download<P: reqwest::IntoUrl, Q: AsRef<Path>>(source_url: P, target_file: Q) -> Fallible<()> {
    let mut resp = reqwest::get(source_url)?;
    let f = fs::File::create(&target_file)?;
    let mut writer = io::BufWriter::new(f);
    resp.copy_to(&mut writer)?;
    Ok(())
}

fn extract<P: AsRef<Path>>(filename: P, outpath: P) {
    let file = fs::File::open(&filename).unwrap();
    let buf = io::BufReader::new(file);
    let mut archive = zip::ZipArchive::new(buf).unwrap();
    for i in 0..archive.len() {
        let mut file = archive.by_index(i).unwrap();
        let outpath = outpath.as_ref().join(file.sanitized_name());
        if !(&*file.name()).ends_with('/') {
            println!(
                "File {} extracted to \"{}\" ({} bytes)",
                i,
                outpath.as_path().display(),
                file.size()
            );
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    fs::create_dir_all(&p).unwrap();
                }
            }
            let mut outfile = fs::File::create(&outpath).unwrap();
            io::copy(&mut file, &mut outfile).unwrap();
        }
    }
}

fn prepare_libtorch_dir() -> PathBuf {
    if let Ok(libtorch) = env::var("LIBTORCH") {
        PathBuf::from(libtorch)
    } else {
        let libtorch_dir = PathBuf::from(env::var("OUT_DIR").unwrap()).join("libtorch");
        if !libtorch_dir.exists() {
            fs::create_dir(&libtorch_dir).unwrap_or_default();
            let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
            let libtorch_url = match os.as_str() {
                "linux" => format!(
                    "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-{}.zip",
                    TORCH_VERSION
                ),
                "macos" => format!(
                    "https://download.pytorch.org/libtorch/cpu/libtorch-macos-{}.zip",
                    TORCH_VERSION
                ),
                _ => panic!("Unsupported OS"),
            };

            let filename = libtorch_dir.join(format!("v{}.zip", TORCH_VERSION));
            download(&libtorch_url, &filename).unwrap();
            extract(&filename, &libtorch_dir);
        }

        libtorch_dir.join("libtorch")
    }
}

fn make<P: AsRef<Path>>(libtorch: P) {
    let libtorch_cxx11_abi = env::var("LIBTORCH_CXX11_ABI").unwrap_or("0".to_string());
    cc::Build::new()
        .cpp(true)
        .include(libtorch.as_ref().join("include"))
        .include(libtorch.as_ref().join("include/torch/csrc/api/include"))
        .flag(&format!(
            "-Wl,-rpath={}",
            libtorch.as_ref().join("lib").display()
        ))
        .flag("-std=c++11")
        .flag("-fPIC")
        .flag(&format!("-D_GLIBCXX_USE_CXX11_ABI={}", libtorch_cxx11_abi))
        .file("libtch/torch_api.cpp")
        .warnings(false)
        .compile("libtorch");
}

fn cmake<P: AsRef<Path>>(libtorch: P) {
    let dst = Config::new("libtch")
        .define("CMAKE_PREFIX_PATH", libtorch.as_ref())
        .build();

    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=static=tch");
    println!("cargo:rustc-link-lib=stdc++");
}

fn main() {
    let libtorch = prepare_libtorch_dir();
    println!(
        "cargo:rustc-link-search=native={}",
        libtorch.join("lib").display()
    );

    if env::var("LIBTORCH_USE_CMAKE").is_ok() {
        cmake(&libtorch)
    } else {
        make(&libtorch)
    }

    println!("cargo:rustc-link-lib=c10");
    println!("cargo:rustc-link-lib=caffe2");
    println!("cargo:rustc-link-lib=torch");
}
