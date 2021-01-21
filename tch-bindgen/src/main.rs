use anyhow::Result;
use std::path::PathBuf;
use structopt::StructOpt;

const TORCH_VERSION: &str = "1.7.0";

#[derive(Debug, Clone, StructOpt)]
#[structopt(name = "tch-bindgen", about = "Rust binding generator for libtorch")]
struct Args {
    /// tch-rs repository directory.
    #[structopt(default_value = ".")]
    pub tch_dir: PathBuf,
    /// custom declaration file.
    pub declaration_file: Option<PathBuf>,
}

fn main() -> Result<()> {
    let Args {
        tch_dir,
        declaration_file,
    } = Args::from_args();

    let declaration_file = declaration_file.unwrap_or_else(|| {
        tch_dir
            .join("third_party")
            .join("pytorch")
            .join(format!("Declarations-v{}.yaml", TORCH_VERSION))
    });
    let cpp_prefix = tch_dir
        .join("torch-sys")
        .join("libtch")
        .join("torch_api_generated");
    let ffi_file = tch_dir.join("torch-sys").join("src").join("c_generated.rs");
    let wrapper_file = tch_dir
        .join("src")
        .join("wrappers")
        .join("tensor_generated.rs");
    let fallible_wrapper_file = tch_dir
        .join("src")
        .join("wrappers")
        .join("tensor_fallible_generated.rs");

    tch_bindgen::generate(
        declaration_file,
        cpp_prefix,
        ffi_file,
        wrapper_file,
        fallible_wrapper_file,
    )?;

    Ok(())
}
