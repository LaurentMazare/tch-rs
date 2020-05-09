use std::ffi::NulError;
use std::io;
use std::num::ParseIntError;

use thiserror::Error;
use zip::result::ZipError;

/// Main library error type.
#[derive(Error, Debug)]
pub enum TchError {
    /// Conversion error.
    #[error("conversion error: {0}")]
    Convert(String),

    /// Invalid file format.
    #[error("invalid file format: {0}")]
    FileFormat(String),

    /// I/O error.
    #[error(transparent)]
    Io(#[from] io::Error),

    /// Tensor kind error.
    #[error("tensor kind error: {0}")]
    Kind(String),

    /// Missing image.
    #[error("no image found in {0}")]
    MissingImage(String),

    /// Null pointer.
    #[error(transparent)]
    Nul(#[from] NulError),

    /// Integer parse error.
    #[error(transparent)]
    ParseInt(#[from] ParseIntError),

    /// Invalid shape.
    #[error("invalid shape: {0}")]
    Shape(String),

    /// Errors returned by the Torch C++ API.
    #[error("Internal torch error: {0}")]
    Torch(String),

    /// Zip file format error.
    #[error(transparent)]
    Zip(#[from] ZipError),
}

impl TchError {
    pub fn path_context(&self, path_name: &str) -> Self {
        match self {
            TchError::Torch(error) => TchError::Torch(format!("{}: {}", path_name, error)),
            _ => unimplemented!(),
        }
    }
}
