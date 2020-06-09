// The LIBTORCH environment variable can be used to specify the directory
// where libtorch has been installed.
// When not specified this script downloads the cpu version for libtorch
// and extracts it in OUT_DIR.
//
// On Linux, the TORCH_CUDA_VERSION environment variable can be used,
// like 9.0, 90, or cu90 to specify the version of CUDA to use for libtorch.
#[macro_use]
extern crate failure;

use std::env;
use std::fs;
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};

use cmake::Config;
use curl::easy::Easy;
use failure::Fallible;
use zip;

const TORCH_VERSION: &'static str = "1.5.0";

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

fn env_var_rerun(name: &str) -> Result<String, env::VarError> {
    println!("cargo:rerun-if-env-changed={}", name);
    env::var(name)
}

fn prepare_libtorch_dir() -> PathBuf {
    let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");

    let device = match env_var_rerun("TORCH_CUDA_VERSION") {
        Ok(cuda_env) => match os.as_str() {
            "linux" | "windows" => cuda_env
                .trim()
                .to_lowercase()
                .trim_start_matches("cu")
                .split(".")
                .take(2)
                .fold("cu".to_string(), |mut acc, curr| {
                    acc += curr;
                    acc
                }),
            os_str => panic!(
                "CUDA was specified with `TORCH_CUDA_VERSION`, but pre-built \
                 binaries with CUDA are only available for Linux and Windows, not: {}.",
                os_str
            ),
        },
        Err(_) => "cpu".to_string(),
    };

    if let Ok(libtorch) = env_var_rerun("LIBTORCH") {
        PathBuf::from(libtorch)
    } else {
        let libtorch_dir = PathBuf::from(env::var("OUT_DIR").unwrap()).join("libtorch");
        if !libtorch_dir.exists() {
            fs::create_dir(&libtorch_dir).unwrap_or_default();
            let libtorch_url = match os.as_str() {
                "linux" => format!(
                    "https://download.pytorch.org/libtorch/{}/libtorch-cxx11-abi-shared-with-deps-{}{}.zip",
                    device, TORCH_VERSION, match device.as_ref() { "cpu" => "%2Bcpu", "cu92" => "%2Bcu92", "cu101"=> "%2Bcu101", _ => "" }
                ),
                "macos" => format!(
                    "https://download.pytorch.org/libtorch/cpu/libtorch-macos-{}.zip",
                    TORCH_VERSION
                ),
                "windows" => format!(
                    "https://download.pytorch.org/libtorch/{}/libtorch-win-shared-with-deps-{}.zip",
                    device,
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

fn make<P: AsRef<Path>>(libtorch: P, use_cuda: bool) {
    let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");

    let cuda_dependency = if use_cuda {
        "libtch/dummy_cuda_dependency.cpp"
    } else {
        "libtch/fake_cuda_dependency.cpp"
    };
    println!("cargo:rerun-if-changed=libtch/torch_api.cpp");
    println!("cargo:rerun-if-changed=libtch/torch_api.h");
    println!("cargo:rerun-if-changed=libtch/torch_api_generated.cpp.h");
    println!("cargo:rerun-if-changed=libtch/torch_api_generated.h");
    println!("cargo:rerun-if-changed=libtch/stb_image_write.h");
    println!("cargo:rerun-if-changed=libtch/stb_image_resize.h");
    println!("cargo:rerun-if-changed=libtch/stb_image.h");
    match os.as_str() {
        "linux" | "macos" => {
            let libtorch_cxx11_abi = env_var_rerun("LIBTORCH_CXX11_ABI").unwrap_or("1".to_string());
            cc::Build::new()
                .cpp(true)
                .pic(true)
                .warnings(false)
                .include(libtorch.as_ref().join("include"))
                .include(libtorch.as_ref().join("include/torch/csrc/api/include"))
                .flag(&format!(
                    "-Wl,-rpath={}",
                    libtorch.as_ref().join("lib").display()
                ))
                .flag("-std=c++14")
                .flag(&format!("-D_GLIBCXX_USE_CXX11_ABI={}", libtorch_cxx11_abi))
                .file("libtch/torch_api.cpp")
                .file(cuda_dependency)
                .compile("tch");
        }
        "windows" => {
            // TODO: Pass "/link" "LIBPATH:{}" to cl.exe in order to emulate rpath.
            //       Not yet supported by cc=rs.
            //       https://github.com/alexcrichton/cc-rs/issues/323
            cc::Build::new()
                .cpp(true)
                .pic(true)
                .warnings(false)
                .include(libtorch.as_ref().join("include"))
                .include(libtorch.as_ref().join("include/torch/csrc/api/include"))
                .file("libtch/torch_api.cpp")
                .file(cuda_dependency)
                .compile("tch");
        }
        _ => panic!("Unsupported OS"),
    };
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
    if !cfg!(feature = "doc-only") {
        let libtorch = prepare_libtorch_dir();
        // use_cuda is a hacky way to detect whether cuda is available and
        // if it's the case link to it by explicitly depending on a symbol
        // from the torch_cuda library.
        // It would be better to use -Wl,--no-as-needed but there is no way
        // to specify arbitrary linker flags at the moment.
        // If https://github.com/rust-lang/cargo/pull/7811 gets released,
        // we should switch to using this instead e.g. with the following
        // flags: -Wl,--no-as-needed -Wl,--copy-dt-needed-entries -ltorch
        let use_cuda = libtorch.join("lib").join("libtorch_cuda.so").exists()
            || libtorch.join("lib").join("torch_cuda.dll").exists();
        println!(
            "cargo:rustc-link-search=native={}",
            libtorch.join("lib").display()
        );

        if env_var_rerun("LIBTORCH_USE_CMAKE").is_ok() {
            cmake(&libtorch)
        } else {
            make(&libtorch, use_cuda)
        }

        println!("cargo:rustc-link-lib=static=tch");
        if use_cuda {
            println!("cargo:rustc-link-lib=torch_cuda");
        }
        println!("cargo:rustc-link-lib=torch");
        println!("cargo:rustc-link-lib=torch_cpu");
        println!("cargo:rustc-link-lib=c10");

        let target = env::var("TARGET").unwrap();

        if !target.contains("msvc") && !target.contains("apple") {
            println!("cargo:rustc-link-lib=gomp");
        }
    }
}
