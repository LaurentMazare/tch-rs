pub use anyhow::{bail, ensure, format_err, Result};
pub use fstrings::{format_args_f, format_f};
pub use indexmap::IndexMap;
pub use itertools::{Either, Itertools};
pub use lazy_static::lazy_static;
pub use maplit::{hashmap, hashset};
pub use proc_macro2::{Ident, TokenStream};
pub use quote::{format_ident, quote};
pub use serde::{Deserialize, Deserializer};
pub use std::{
    collections::{HashMap, HashSet},
    fs,
    hash::Hash,
    iter,
    ops::Deref,
    path::{Path, PathBuf},
};

pub type Fallible<T> = Result<T>;

unzip_n::unzip_n!(pub 5);
