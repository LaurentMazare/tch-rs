// The LIBTORCH environment variable is used to locate the libtorch
// library. If not set this build script will try downloading and
// extracting a pre-built binary version of libtorch.
use cmake::Config;
use std::env;
use std::path::PathBuf;
use std::process::Command;

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

fn maybe_download(filename: &std::path::Path, url: &str) {
    if !filename.exists() {
        Command::new("wget")
            .args(&[url, "-nv", "-O", &filename.to_string_lossy().into_owned()])
            .status().unwrap();
    }
}

fn unzip_all(absolute_filename: &std::path::Path, outpath: &std::path::Path) {
    let file = std::fs::File::open(&absolute_filename).unwrap();
    let mut archive = zip::ZipArchive::new(file).unwrap();
    for i in 0..archive.len() {
        let mut file = archive.by_index(i).unwrap();
        let outpath = outpath.join(file.sanitized_name());
        if !(&*file.name()).ends_with('/') {
            println!("File {} extracted to \"{}\" ({} bytes)", i, outpath.as_path().display(), file.size());
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    std::fs::create_dir_all(&p).unwrap();
                }
            }
            let mut outfile = std::fs::File::create(&outpath).unwrap();
            std::io::copy(&mut file, &mut outfile).unwrap();
        }
    }
}

fn main() {
    let out_path = PathBuf::from(&env::var("OUT_DIR").unwrap());
    let libtorch =
        if let Ok(libtorch) = env::var("LIBTORCH") { PathBuf::from(libtorch) }
        else {
            let libtorch = out_path.join("libtorch");
            if !libtorch.exists() {
                let url = match env::consts::OS {
                    "macos" => "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.0.1.zip",
                    "linux" => "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.0.1.zip",
                    otherwise => panic!("unsupported os {}", otherwise)
                };
                let filename = url.split("/").last().unwrap();
                let download_dir = PathBuf::from(&env::var("CARGO_MANIFEST_DIR").unwrap()).join("target");
                if !download_dir.exists() {
                    std::fs::create_dir(&download_dir).unwrap();
                }
                let absolute_filename = download_dir.join(filename);
                println!("Downloading {}", url);
                println!("To {:?}", absolute_filename);
                maybe_download(&absolute_filename, &url);
                println!("Extracting in {:?}", libtorch);
                unzip_all(&absolute_filename, &out_path)
            }
            libtorch
        };
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
