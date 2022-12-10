// The LIBTORCH environment variable can be used to specify the directory
// where libtorch has been installed.
// When not specified this script downloads the cpu version for libtorch
// and extracts it in OUT_DIR.
//
// On Linux, the TORCH_CUDA_VERSION environment variable can be used,
// like 9.0, 90, or cu90 to specify the version of CUDA to use for libtorch.

use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::str::FromStr;

const TORCH_VERSION: &str = "1.13.0";

#[cfg(feature = "curl")]
fn download<P: AsRef<Path>>(source_url: &str, target_file: P) -> anyhow::Result<()> {
    use curl::easy::Easy;
    use std::io::Write;

    let f = fs::File::create(&target_file)?;
    let mut writer = io::BufWriter::new(f);
    let mut easy = Easy::new();
    easy.url(source_url)?;
    easy.write_function(move |data| Ok(writer.write(data).unwrap()))?;
    easy.perform()?;
    let response_code = easy.response_code()?;
    if response_code == 200 {
        Ok(())
    } else {
        Err(anyhow::anyhow!("Unexpected response code {} for {}", response_code, source_url))
    }
}

#[cfg(not(feature = "curl"))]
fn download<P: AsRef<Path>>(_source_url: &str, _target_file: P) -> anyhow::Result<()> {
    anyhow::bail!("cannot use download without the curl feature")
}

fn extract<P: AsRef<Path>>(filename: P, outpath: P) -> anyhow::Result<()> {
    let file = fs::File::open(&filename)?;
    let buf = io::BufReader::new(file);
    let mut archive = zip::ZipArchive::new(buf)?;
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        #[allow(deprecated)]
        let outpath = outpath.as_ref().join(file.sanitized_name());
        if !file.name().ends_with('/') {
            println!(
                "File {} extracted to \"{}\" ({} bytes)",
                i,
                outpath.as_path().display(),
                file.size()
            );
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    fs::create_dir_all(p)?;
                }
            }
            let mut outfile = fs::File::create(&outpath)?;
            io::copy(&mut file, &mut outfile)?;
        }
    }
    Ok(())
}

fn env_var_target_specific(name: &str) -> Result<String, env::VarError> {
    let target = env::var("TARGET").expect("Unable to get TARGET");

    let name_with_target_hyphenated = name.to_owned() + "_" + &target;
    let name_with_target_underscored = name.to_owned() + "_" + &target.replace("-", "_");

    env_var_rerun(&name_with_target_hyphenated)
        .or_else(|_| env_var_rerun(&name_with_target_underscored))
        .or_else(|_| env_var_rerun(name))
}

fn env_var_rerun(name: &str) -> Result<String, env::VarError> {
    println!("cargo:rerun-if-env-changed={}", name);
    env::var(name)
}

fn check_system_location() -> Option<PathBuf> {
    let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");

    match os.as_str() {
        "linux" => Path::new("/usr/lib/libtorch.so").exists().then(|| PathBuf::from("/usr")),
        _ => None,
    }
}

fn prepare_libtorch_dir() -> PathBuf {
    let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");

    let device = match env_var_rerun("TORCH_CUDA_VERSION") {
        Ok(cuda_env) => match os.as_str() {
            "linux" | "windows" => {
                cuda_env.trim().to_lowercase().trim_start_matches("cu").split('.').take(2).fold(
                    "cu".to_owned(),
                    |mut acc, curr| {
                        acc += curr;
                        acc
                    },
                )
            }
            os_str => panic!(
                "CUDA was specified with `TORCH_CUDA_VERSION`, but pre-built \
                 binaries with CUDA are only available for Linux and Windows, not: {}.",
                os_str
            ),
        },
        Err(_) => "cpu".to_owned(),
    };

    if let Ok(libtorch) = env_var_target_specific("LIBTORCH") {
        PathBuf::from(libtorch)
    } else if let Some(pathbuf) = check_system_location() {
        pathbuf
    } else {
        let libtorch_dir = PathBuf::from(env::var("OUT_DIR").unwrap()).join("libtorch");
        if !libtorch_dir.exists() {
            fs::create_dir(&libtorch_dir).unwrap_or_default();
            let libtorch_url = match os.as_str() {
                "linux" => format!(
                    "https://download.pytorch.org/libtorch/{}/libtorch-cxx11-abi-shared-with-deps-{}{}.zip",
                    device, TORCH_VERSION, match device.as_ref() {
                        "cpu" => "%2Bcpu",
                        "cu102" => "%2Bcu102",
                        "cu113" => "%2Bcu113",
                        "cu116" => "%2Bcu116",
                        "cu117" => "%2Bcu117",
                        _ => panic!("unsupported device {}, TORCH_CUDA_VERSION may be set incorrectly?", device),
                    }
                ),
                "macos" => format!(
                    "https://download.pytorch.org/libtorch/cpu/libtorch-macos-{}.zip",
                    TORCH_VERSION
                ),
                "windows" => format!(
                    "https://download.pytorch.org/libtorch/{}/libtorch-win-shared-with-deps-{}{}.zip",
                    device, TORCH_VERSION, match device.as_ref() {
                        "cpu" => "%2Bcpu",
                        "cu102" => "%2Bcu102",
                        "cu113" => "%2Bcu113",
                        "cu116" => "%2Bcu116",
                        "cu117" => "%2Bcu117",
                        _ => ""
                    }),
                _ => panic!("Unsupported OS"),
            };

            let filename = libtorch_dir.join(format!("v{}.zip", TORCH_VERSION));
            download(&libtorch_url, &filename).unwrap();
            extract(&filename, &libtorch_dir).unwrap();
        }

        libtorch_dir.join("libtorch")
    }
}

fn make(
    includes: impl AsRef<Path>,
    lib: impl AsRef<Path>,
    use_cuda: bool,
    use_hip: bool,
    os: &str,
) {
    let includes = includes.as_ref();
    let lib = lib.as_ref();

    let cuda_dependency = if use_cuda || use_hip {
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
    match os {
        "linux" | "macos" | "android" => {
            let libtorch_cxx11_abi =
                env_var_rerun("LIBTORCH_CXX11_ABI").unwrap_or_else(|_| "1".to_owned());
            cc::Build::new()
                .cpp(true)
                .pic(true)
                .warnings(false)
                .include(includes)
                .include(includes.join("torch/csrc/api/include"))
                .flag(&format!("-Wl,-rpath={}", lib.display()))
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
                .include(includes)
                .include(includes.join("torch/csrc/api/include"))
                .file("libtch/torch_api.cpp")
                .file(cuda_dependency)
                .compile("tch");
        }
        os => panic!("Unsupported OS: {}", os),
    };
}

fn main() {
    if !cfg!(feature = "doc-only") {
        let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");

        let libtorch = prepare_libtorch_dir();

        let libtorch_includes: PathBuf = env_var_target_specific("LIBTORCH_INCLUDE")
            .map(Into::into)
            .unwrap_or_else(|_| libtorch.join("include"));
        let libtorch_lib: PathBuf = env_var_target_specific("LIBTORCH_LIB")
            .map(Into::into)
            .unwrap_or_else(|_| libtorch.join("lib"));
        let libtorch_lite: bool = env_var_target_specific("LIBTORCH_LITE")
            .map(|s| s.parse().unwrap_or(true))
            .unwrap_or(true);

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
        let use_cuda = libtorch_lib.join("libtorch_cuda.so").exists()
            || libtorch_lib.join("torch_cuda.dll").exists();
        let use_cuda_cu = libtorch_lib.join("libtorch_cuda_cu.so").exists()
            || libtorch_lib.join("torch_cuda_cu.dll").exists();
        let use_cuda_cpp = libtorch_lib.join("libtorch_cuda_cpp.so").exists()
            || libtorch_lib.join("torch_cuda_cpp.dll").exists();
        let use_hip = libtorch_lib.join("libtorch_hip.so").exists()
            || libtorch_lib.join("torch_hip.dll").exists();

        println!("cargo:rustc-link-search=native={}", libtorch_lib.display());

        make(&libtorch_includes, &libtorch_lib, use_cuda, use_hip, &os);

        println!("cargo:rustc-link-lib=static=tch");

        match os.as_str() {
            "windows" | "linux" | "macos" => {
                if use_cuda {
                    println!("cargo:rustc-link-lib=torch_cuda");
                }
                if use_cuda_cu {
                    println!("cargo:rustc-link-lib=torch_cuda_cu");
                }
                if use_cuda_cpp {
                    println!("cargo:rustc-link-lib=torch_cuda_cpp");
                }
                if use_hip {
                    println!("cargo:rustc-link-lib=torch_hip");
                }
                println!("cargo:rustc-link-lib=torch_cpu");
                println!("cargo:rustc-link-lib=torch");
                println!("cargo:rustc-link-lib=c10");
                if use_hip {
                    println!("cargo:rustc-link-lib=c10_hip");
                }

                let target = env::var("TARGET").unwrap();

                if !target.contains("msvc") && !target.contains("apple") {
                    println!("cargo:rustc-link-lib=gomp");
                }
            }
            "android" => {
                if libtorch_lite {
                    println!("cargo:rustc-link-lib=pytorch_jni_lite");
                } else {
                    println!("cargo:rustc-link-lib=pytorch_jni");
                }
            }
            other => panic!("unsupported OS: {}", other),
        }
    }
}
