// The LIBTORCH environment variable can be used to specify the directory
// where libtorch has been installed.
// When not specified this script downloads the cpu version for libtorch
// and extracts it in OUT_DIR.
//
// On Linux, the TORCH_CUDA_VERSION environment variable can be used,
// like 9.0, 90, or cu90 to specify the version of CUDA to use for libtorch.
use curl::easy::Easy;
use failure::{format_err, Fallible};
use std::{
    env, fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

const TORCH_VERSION: &'static str = "1.2.0";

fn download<P: AsRef<Path>>(source_url: &str, target_file: P) -> Fallible<()> {
    let f = fs::File::create(&target_file)?;
    let mut writer = io::BufWriter::new(f);
    let mut easy = Easy::new();
    easy.url(&source_url)?;
    easy.write_function(move |data| Ok(writer.write(data).unwrap()))?;
    easy.perform()?;
    let response_code = easy.response_code()?;
    if response_code == 200 {
        Ok(())
    } else {
        Err(format_err!(
            "Unexpected response code {} for {}",
            response_code,
            source_url
        ))
    }
}

fn extract<P: AsRef<Path>>(filename: P, outpath: P) -> Fallible<()> {
    let file = fs::File::open(&filename)?;
    let buf = io::BufReader::new(file);
    let mut archive = zip::ZipArchive::new(buf)?;
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
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
                    fs::create_dir_all(&p)?;
                }
            }
            let mut outfile = fs::File::create(&outpath)?;
            io::copy(&mut file, &mut outfile)?;
        }
    }
    Ok(())
}

fn prepare_libtorch_dir() -> PathBuf {
    let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");

    let device = match env::var("TORCH_CUDA_VERSION") {
        Ok(cuda_env) => match os.as_str() {
            "linux" => match cuda_env.trim_start() {
                "9.0" | "90" | "cu90" => "cu90",
                version => panic!("Unsupported CUDA version specified: {}", version),
            },
            os_str => panic!(
                "CUDA was specified with `TORCH_CUDA_VERSION`, but pre-built \
                 binaries with CUDA are only available for Linux, not: {}.",
                os_str
            ),
        },
        Err(_) => "cpu",
    };

    if let Ok(libtorch) = env::var("LIBTORCH") {
        PathBuf::from(libtorch)
    } else {
        let libtorch_dir = PathBuf::from(env::var("OUT_DIR").unwrap()).join("libtorch");
        if !libtorch_dir.exists() {
            fs::create_dir(&libtorch_dir).unwrap_or_default();
            let libtorch_url = match os.as_str() {
                "linux" => format!(
                    "https://download.pytorch.org/libtorch/{}/libtorch-shared-with-deps-{}.zip",
                    device, TORCH_VERSION
                ),
                "macos" => format!(
                    "https://download.pytorch.org/libtorch/cpu/libtorch-macos-{}.zip",
                    TORCH_VERSION
                ),
                "windows" => format!(
                    "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-{}.zip",
                    TORCH_VERSION
                ),
                _ => panic!("Unsupported OS"),
            };

            let filename = libtorch_dir.join(format!("v{}.zip", TORCH_VERSION));
            download(&libtorch_url, &filename).unwrap();
            extract(&filename, &libtorch_dir).unwrap();
        }

        libtorch_dir.join("libtorch")
    }
}

fn compile<P: AsRef<Path>>(libtorch: P) {
    let libtorch_cxx11_abi = env::var("LIBTORCH_CXX11_ABI").unwrap_or("0".to_string());
    cc::Build::new()
        .cpp(true)
        .pic(true)
        .warnings(false)
        .include(libtorch.as_ref().join("include"))
        .include(libtorch.as_ref().join("include/torch/csrc/api/include"))
        .flag("-std=c++11")
        .flag(&format!("-D_GLIBCXX_USE_CXX11_ABI={}", libtorch_cxx11_abi))
        .file("libtch/torch_api.cpp")
        .compile("tch");
}

fn main() {
    let libtorch = prepare_libtorch_dir();
    println!(
        "cargo:rustc-link-search=native={}",
        libtorch.join("lib").display()
    );

    compile(&libtorch);

    let out_dir = env::var("OUT_DIR").unwrap();

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=tch");
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=c10");

    let target = env::var("TARGET").unwrap();

    if target.contains("apple") || target.contains("freebsd") || target.contains("openbsd") {
        println!("cargo:rustc-link-lib=gomp");
        println!("cargo:rustc-link-lib=c++");
    } else if target.contains("linux") {
        println!("cargo:rustc-link-lib=gomp");
        println!("cargo:rustc-link-lib=stdc++");
    } else if target.contains("msvc") {
        println!("cargo:rustc-link-lib=iphlpapi");
    }
}
