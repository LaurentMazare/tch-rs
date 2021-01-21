mod common;
mod declarations;
mod utils;

use crate::{
    common::*,
    declarations::{Argument, Declarations, Type},
    utils::{IteratorExt, StrExt},
};

lazy_static! {
    static ref EXCLUDED_FUNCTIONS: HashSet<&'static str> = hashset! {
        "multi_margin_loss",
        "multi_margin_loss_out",
        "log_softmax_backward_data",
        "softmax_backward_data",
        "clone",
        "copy_",
        "conv_transpose2d_backward_out",
        "conv_transpose3d_backward_out",
        "slow_conv_transpose2d_backward_out",
        "slow_conv_transpose3d_backward_out",
        "slow_conv3d_backward_out",
        "normal",
        "_cufft_set_plan_cache_max_size",
        "_cufft_clear_plan_cache",
        "backward",
        "set_data",
        "_amp_non_finite_check_and_unscale_",
        "_cummin_helper",
        "_cummax_helper",
        "retain_grad",
        "_validate_sparse_coo_tensor_args",
    };
    static ref EXCLUDED_PREFIXES: HashSet<&'static str> = hashset! {
        "_thnn_",
        "_th_",
        "thnn_",
        "th_",
        "_foreach"
    };
    static ref EXCLUDED_SUFFIXES: HashSet<&'static str> = hashset! {
        "_forward",
        "_forward_out",
    };
    static ref NO_TENSOR_OPTIONS: HashSet<&'static str> = hashset! {
        "zeros_like",
        "empty_like",
        "full_like",
        "ones_like",
        "rand_like",
        "randint_like",
        "randn_like",
    };
    static ref PREFIXED_FUNCTIONS: HashSet<&'static str> = hashset! {
        "add",
        "add_",
        "div",
        "div_",
        "mul",
        "mul_",
        "sub",
        "sub_",
        "nll_loss",
    };
    static ref REPLACE_MAP: HashMap<&'static str, &'static str> = hashmap! {
        "t" => "tr",
        "where" => "where_",
        "view" => "view_",
        "unsafe" => "unsafe_",
    };
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ReturnKind {
    Tensor,
    TensorArray(usize),
    TensorList,
    Void,
    Single(Type),
    Tuple(Vec<Type>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ItemKind {
    Function,
    Method,
    Other,
}

#[derive(Debug, Clone)]
enum ParseArgumentError {
    Skip,
    NonSimple,
}

#[derive(Debug, Clone)]
struct ItemExt<'a, 'b> {
    pub c_name: &'a str,
    pub item_kind: ItemKind,
    pub return_kind: ReturnKind,
    pub args: Vec<ArgumentExt<'b>>,
}

#[derive(Debug, Clone)]
struct ArgumentExt<'a> {
    pub c_name: &'a str,
    pub rust_ident: Ident,
    pub rust_type: Type,
}

pub fn generate(
    declarations_file: impl AsRef<Path>,
    cpp_prefix: impl AsRef<Path>,
    ffi_file: impl AsRef<Path>,
    wrapper_file: impl AsRef<Path>,
    fallible_wrapper_file: impl AsRef<Path>,
) -> Result<()> {
    let declarations: Declarations = serde_yaml::from_str(&fs::read_to_string(declarations_file)?)?;

    let manual_items = vec![
        ItemExt {
            c_name: "grad",
            item_kind: ItemKind::Method,
            return_kind: ReturnKind::Tensor,
            args: vec![ArgumentExt {
                c_name: "self",
                rust_ident: format_ident!("self"),
                rust_type: Type::Tensor,
            }],
        },
        ItemExt {
            c_name: "set_requires_grad",
            item_kind: ItemKind::Method,
            return_kind: ReturnKind::Tensor,
            args: vec![
                ArgumentExt {
                    c_name: "self",
                    rust_ident: format_ident!("self"),
                    rust_type: Type::Tensor,
                },
                ArgumentExt {
                    c_name: "r",
                    rust_ident: format_ident!("r"),
                    rust_type: Type::Bool,
                },
            ],
        },
        ItemExt {
            c_name: "toType",
            item_kind: ItemKind::Method,
            return_kind: ReturnKind::Tensor,
            args: vec![
                ArgumentExt {
                    c_name: "self",
                    rust_ident: format_ident!("self"),
                    rust_type: Type::Tensor,
                },
                ArgumentExt {
                    c_name: "scalar_type",
                    rust_ident: format_ident!("scalar_type"),
                    rust_type: Type::ScalarType,
                },
            ],
        },
        ItemExt {
            c_name: "to",
            item_kind: ItemKind::Method,
            return_kind: ReturnKind::Tensor,
            args: vec![
                ArgumentExt {
                    c_name: "self",
                    rust_ident: format_ident!("self"),
                    rust_type: Type::Tensor,
                },
                ArgumentExt {
                    c_name: "device",
                    rust_ident: format_ident!("device"),
                    rust_type: Type::Device,
                },
            ],
        },
    ];
    let auto_items = declarations
        .items()
        .iter()
        // skip deprecated and excluded items
        .filter(|item| {
            let name = item.name.as_str();
            !item.deprecated
                && !EXCLUDED_PREFIXES
                    .iter()
                    .any(|prefix| prefix.is_prefix_of(name))
                && !EXCLUDED_SUFFIXES
                    .iter()
                    .any(|suffix| suffix.is_suffix_of(name))
                && !EXCLUDED_FUNCTIONS.contains(name)
        })
        // extract item properties
        .filter_map(|item| {
            // determine the return type
            let return_kind = {
                let returns = &item.returns;
                let num_returns = returns.len();
                let is_all_tensor = returns.iter().all(|arg| arg.dynamic_type == Type::Tensor);
                match (is_all_tensor, num_returns) {
                    (_, 0) => ReturnKind::Void,
                    (true, 1) => ReturnKind::Tensor,
                    (true, len) => ReturnKind::TensorArray(len),
                    (false, 1) => {
                        let type_ = &returns[0].dynamic_type;
                        if type_ == &Type::TensorList {
                            ReturnKind::TensorList
                        } else {
                            ReturnKind::Single(type_.clone())
                        }
                    }
                    (false, _) => {
                        let return_types: Vec<_> =
                            returns.iter().map(|arg| arg.dynamic_type.clone()).collect();
                        ReturnKind::Tuple(return_types)
                    }
                }
            };

            // classify into function, method or other
            let item_kind = if item.method_of.contains("namespace") {
                ItemKind::Function
            } else if item.method_of.contains("Tensor") {
                ItemKind::Method
            } else {
                ItemKind::Other
            };

            // determine argument types in Rust
            let args: Result<Vec<_>, _> = item
                .arguments
                .iter()
                .map(|arg| {
                    let rust_type = match to_rust_arg_type(&item.name, arg) {
                        Ok(arg_type) => arg_type,
                        Err(ParseArgumentError::Skip) => return Ok(None),
                        Err(ParseArgumentError::NonSimple) => {
                            return Err(ParseArgumentError::NonSimple)
                        }
                    };
                    let c_name = match (&rust_type, arg.name.as_str()) {
                        (Type::Scalar, "self") => "self_scalar",
                        (_, name) => name,
                    };
                    let rust_ident = format_ident!("{}", to_rust_name(c_name));

                    Ok(Some(ArgumentExt {
                        c_name,
                        rust_ident,
                        rust_type,
                    }))
                })
                .filter_map(|result| result.transpose())
                .collect();

            // reject unknown item kind
            if item_kind == ItemKind::Other {
                return None;
            }

            // accept only those returning tensors
            match return_kind {
                ReturnKind::Tensor | ReturnKind::TensorArray(_) | ReturnKind::TensorList => (),
                _ => return None,
            }

            // make sure arguments are unsderstood
            let args = match args {
                Ok(args) => args,
                Err(_) => return None,
            };

            Some(ItemExt {
                c_name: item.name.as_str(),
                return_kind,
                item_kind,
                args,
            })
        });

    let collected = manual_items
        .into_iter()
        .chain(auto_items)
        // group items of the same name
        .map(|item| {
            let item_key = item.c_name.to_lowercase();
            (item_key, item)
        })
        .into_group_index_map()
        .into_iter()
        // rename items of the same name with index suffixes
        .flat_map(|(item_key, mut items)| {
            items.sort_by_cached_key(|item| item.args.len());
            let mut items_iter = items.into_iter().enumerate();
            let (_0, first_item) = items_iter.next().unwrap();
            let first_exported_item_name = item_key.clone();

            iter::once((first_exported_item_name, first_item)).chain(items_iter.map(
                move |(index, item)| {
                    let exported_item_name = format!("{}{}", item_key, index);
                    (exported_item_name, item)
                },
            ))
        })
        // generate code
        .map(|(exported_item_name, item)| -> Result<_> {
            let ffi_name = format_ident!("atg_{}", exported_item_name);

            let (fallible_wrapper_fn, wrapper_fn) = {
                let (rust_func_ident, fallible_rust_func_ident) = {
                    let base_name = to_rust_name(&exported_item_name);
                    let (fn_name, fallible_fn_name) = if PREFIXED_FUNCTIONS.contains(item.c_name) {
                        let fn_name = format!("g_{}", base_name);
                        let fallible_fn_name = format!("f_{}", base_name);
                        (fn_name, fallible_fn_name)
                    } else {
                        let fallible_fn_name = format!("f_{}", base_name);
                        let fn_name = base_name;
                        (fn_name, fallible_fn_name)
                    };

                    (
                        format_ident!("{}", fn_name),
                        format_ident!("{}", fallible_fn_name),
                    )
                };

                let (self_arg, remaining_args) = (|| {
                    // find "self" argument
                    let (self_args, remaining_args): (Vec<_>, Vec<_>) =
                        item.args.iter().partition_map(|arg| {
                            if arg.c_name == "self" {
                                Either::Left(arg)
                            } else {
                                Either::Right(arg)
                            }
                        });

                    match self_args.as_slice() {
                        &[self_arg] => return (Some(self_arg), remaining_args),
                        _ => (),
                    }

                    // find "input" argument
                    let (self_args, remaining_args): (Vec<_>, Vec<_>) =
                        item.args.iter().partition_map(|arg| {
                            if arg.c_name == "input" {
                                Either::Left(arg)
                            } else {
                                Either::Right(arg)
                            }
                        });

                    match self_args.as_slice() {
                        &[self_arg] => (Some(&self_arg), remaining_args),
                        _ => (None, remaining_args),
                    }
                })();

                let return_type = to_rust_return_type(&item.return_kind);

                let type_params = {
                    let scalar_type_param = {
                        let has_scalar_arg =
                            item.args.iter().any(|arg| arg.rust_type == Type::Scalar);
                        if has_scalar_arg {
                            Some(quote! { S: Into<Scalar> })
                        } else {
                            None
                        }
                    };

                    let tensor_type_param = {
                        let has_tensor_arg = item.args.iter().any(|arg| match arg.rust_type {
                            Type::TensorList | Type::TensorOptional => true,
                            _ => false,
                        });
                        if has_tensor_arg {
                            Some(quote! { T: Borrow<Tensor> })
                        } else {
                            None
                        }
                    };

                    let type_params: Vec<_> = tensor_type_param
                        .into_iter()
                        .chain(scalar_type_param.into_iter())
                        .collect();

                    if type_params.is_empty() {
                        None
                    } else {
                        Some(quote! { <#(#type_params),*> })
                    }
                };

                let wrapper_fn_params = {
                    if self_arg.is_some() {
                        let self_arg = if item.c_name.ends_with("_") {
                            quote! { &mut self }
                        } else {
                            quote! { &self }
                        };

                        let rust_args: Vec<_> = iter::once(quote! { #self_arg })
                            .chain(to_rust_args(remaining_args.iter().cloned()))
                            .collect();
                        quote! { #(#rust_args),* }
                    } else {
                        let rust_args: Vec<_> =
                            to_rust_args(remaining_args.iter().cloned()).collect();
                        quote! { #(#rust_args),* }
                    }
                };

                let fallible_wrapper_fn = {
                    let arg_convert = {
                        let lines: Vec<_> = item
                            .args
                            .iter()
                            .map(|arg| {
                                let arg_ident = &arg.rust_ident;
                                match arg.rust_type {
                                    Type::DoubleOptional | Type::Int64Optional => {
                                        Some(quote! { let #arg_ident = #arg_ident.into(); })
                                    }
                                    _ => None,
                                }
                            })
                            .collect();

                        quote! { #(#lines)* }
                    };

                    let ffi_call_params = {
                        let params: Vec<_> = item
                            .args
                            .iter()
                            .map(|arg| {
                                let param_ident = {
                                    let is_self_arg = match self_arg {
                                        Some(self_arg) => self_arg.c_name == arg.c_name,
                                        None => false,
                                    };

                                    if is_self_arg {
                                        format_ident!("self")
                                    } else {
                                        arg.rust_ident.clone()
                                    }
                                };
                                match arg.rust_type {
                                    Type::Tensor => quote! { #param_ident.c_tensor },
                                    Type::Scalar => quote! { #param_ident.into().c_scalar },
                                    Type::Bool => {
                                        quote! { if #param_ident { 1 } else { 0 } }
                                    }
                                    Type::ScalarType => quote! { #param_ident.c_int() },
                                    Type::Device => quote! { #param_ident.c_int() },
                                    Type::TensorOptions => quote! {
                                        #param_ident.0.c_int(),
                                        #param_ident.1.c_int()
                                    },
                                    Type::Int64Optional => quote! {
                                        #param_ident.unwrap_or(0i64),
                                        #param_ident.is_none() as i8
                                    },
                                    Type::DoubleOptional => quote! {
                                        #param_ident.unwrap_or(std::f64::NAN),
                                        #param_ident.is_none() as i8
                                    },
                                    Type::String => quote! {
                                        #param_ident.as_ptr(),
                                        #param_ident.len() as i32
                                    },
                                    Type::IntList => quote! {
                                        #param_ident.as_ptr(),
                                        #param_ident.len() as i32
                                    },
                                    Type::TensorList => quote! {
                                        ptr_list(#param_ident).as_ptr(),
                                        #param_ident.len() as i32
                                    },
                                    Type::TensorOptional => quote! {
                                        #param_ident.map_or(
                                            std::ptr::null_mut(),
                                            |t| t.borrow().c_tensor
                                        )
                                    },
                                    Type::Int64 => {
                                        if param_ident == "reduction" {
                                            quote! { #param_ident.to_int() }
                                        } else {
                                            quote! { #param_ident }
                                        }
                                    }
                                    Type::Double => quote! { #param_ident },
                                    Type::Other(_) => unreachable!(),
                                }
                            })
                            .collect();
                        quote! { #(#params),* }
                    };

                    let fallible_wrapper_impl = match item.return_kind {
                        ReturnKind::TensorList => {
                            quote! {
                                let c_tensors = unsafe_torch_err!(
                                    #ffi_name (#ffi_call_params)
                                );
                                let mut r__ = vec![];
                                let mut i = 0;

                                loop {
                                    let c__ = unsafe { *c_tensors.add(i) };
                                    if c__.is_null() {
                                        break
                                    }
                                    r__.push(Tensor { c_tensor: c__ });
                                    i += 1;
                                }

                                unsafe {
                                    libc::free(c_tensors as *mut libc::c_void)
                                }

                                Ok(r__)
                            }
                        }
                        ReturnKind::Tensor => {
                            quote! {
                                let mut c_tensors = [std::ptr::null_mut(); 1];
                                unsafe_torch_err!(
                                    #ffi_name(
                                        c_tensors.as_mut_ptr(),
                                        #ffi_call_params
                                    )
                                );
                                Ok(Tensor { c_tensor: c_tensors[0] })
                            }
                        }
                        ReturnKind::TensorArray(size) => {
                            let return_expr = {
                                let values: Vec<_> = (0..size)
                                    .map(|index| {
                                        quote! {
                                            Tensor { c_tensor: c_tensors[#index] }
                                        }
                                    })
                                    .collect();
                                quote! { Ok((#(#values),*)) }
                            };

                            quote! {
                                let mut c_tensors = [std::ptr::null_mut(); #size];
                                unsafe_torch_err!(
                                    #ffi_name(
                                        c_tensors.as_mut_ptr(),
                                        #ffi_call_params
                                    )
                                );
                                #return_expr
                            }
                        }
                        _ => unreachable!(),
                    };

                    let fallible_wrapper_fn = quote! {
                        pub fn #fallible_rust_func_ident #type_params (
                            #wrapper_fn_params
                        ) -> Result<#return_type, TchError> {
                            #arg_convert
                            #fallible_wrapper_impl
                        }
                    };

                    fallible_wrapper_fn
                };

                let wrapper_fn = {
                    let wrapper_call_params = {
                        let params: Vec<_> =
                            remaining_args.iter().map(|arg| &arg.rust_ident).collect();
                        quote! { #(#params),* }
                    };

                    let wrapper_fn_impl = if self_arg.is_some() {
                        quote! {
                            self.#fallible_rust_func_ident(
                                #wrapper_call_params
                            ).unwrap()
                        }
                    } else {
                        quote! {
                            Tensor::#fallible_rust_func_ident(
                                #wrapper_call_params
                            ).unwrap()
                        }
                    };

                    let wrapper_fn = quote! {
                        pub fn #rust_func_ident #type_params (
                            #wrapper_fn_params
                        ) -> #return_type {
                            #wrapper_fn_impl
                        }
                    };

                    wrapper_fn
                };

                (fallible_wrapper_fn, wrapper_fn)
            };

            let ffi_fn = {
                let ffi_fn_params = {
                    let params: Vec<_> = item
                        .args
                        .iter()
                        .map(|arg| {
                            let arg_ident = format_ident!("{}_", arg.c_name);
                            let ffi_args = match arg.rust_type {
                                Type::Bool => quote! { #arg_ident: c_int },
                                Type::Int64 => quote! { #arg_ident: i64 },
                                Type::Double => quote! { #arg_ident: f64 },
                                Type::Tensor => quote! { #arg_ident: *mut C_tensor },
                                Type::TensorOptional => {
                                    quote! { #arg_ident: *mut C_tensor }
                                }
                                Type::Scalar => quote! { #arg_ident: *mut C_scalar },
                                Type::ScalarType => quote! { #arg_ident: c_int },
                                Type::Device => quote! { #arg_ident: c_int },
                                Type::String => {
                                    let ptr_name = format_ident!("{}_ptr", arg_ident);
                                    let len_name = format_ident!("{}_len", arg_ident);
                                    quote! {
                                        #ptr_name: *const u8,
                                        #len_name: c_int
                                    }
                                }
                                Type::IntList => {
                                    let ptr_name = format_ident!("{}_data", arg_ident);
                                    let len_name = format_ident!("{}_len", arg_ident);
                                    quote! {
                                        #ptr_name: *const i64,
                                        #len_name: c_int
                                    }
                                }
                                Type::TensorList => {
                                    let ptr_name = format_ident!("{}_data", arg_ident);
                                    let len_name = format_ident!("{}_len", arg_ident);
                                    quote! {
                                        #ptr_name: *const *mut C_tensor,
                                        #len_name: c_int
                                    }
                                }
                                Type::Int64Optional => {
                                    let val_name = format_ident!("{}_v", arg_ident);
                                    let nullity_name = format_ident!("{}_null", arg_ident);
                                    quote! {
                                        #val_name: i64,
                                        #nullity_name: i8
                                    }
                                }
                                Type::DoubleOptional => {
                                    let val_name = format_ident!("{}_v", arg_ident);
                                    let nullity_name = format_ident!("{}_null", arg_ident);
                                    quote! {
                                        #val_name: f64,
                                        #nullity_name: i8
                                    }
                                }
                                Type::TensorOptions => {
                                    let kind_name = format_ident!("{}_kind", arg_ident);
                                    let device_name = format_ident!("{}_device", arg_ident);
                                    quote! {
                                        #kind_name: c_int,
                                        #device_name: c_int
                                    }
                                }
                                Type::Other(_) => unreachable!(),
                            };
                            ffi_args
                        })
                        .collect();
                    quote! { #(#params),* }
                };

                let ffi_fn = match item.return_kind {
                    ReturnKind::Tensor | ReturnKind::TensorArray(_) => {
                        quote! {
                            pub fn #ffi_name(out__: *mut *mut C_tensor, #ffi_fn_params);
                        }
                    }
                    ReturnKind::TensorList => {
                        quote! {
                            pub fn #ffi_name(#ffi_fn_params) -> *mut *mut C_tensor;
                        }
                    }
                    _ => unreachable!(),
                };

                ffi_fn
            };

            let (header_fn, cpp_fn) = {
                let cpp_call_params = match item.item_kind {
                    ItemKind::Function => {
                        let c_name = &item.c_name;
                        let c_args = to_c_args(item.args.iter());
                        format_f!("torch::{c_name}({c_args})")
                    }
                    ItemKind::Method => {
                        let c_name = &item.c_name;
                        let first_arg = &item.args[0];
                        let first_arg_name = &first_arg.c_name;
                        let c_args = to_c_args(item.args.iter().skip(1));
                        format_f!("{first_arg_name}->{c_name}({c_args})")
                    }
                    ItemKind::Other => unreachable!(),
                };

                let cpp_fn_params = {
                    item.args
                        .iter()
                        .map(|arg| {
                            let param_ident = format_ident!("{}", arg.c_name);
                            match arg.rust_type {
                                Type::IntList => {
                                    format_f!("int64_t *{param_ident}_data, int {param_ident}_len")
                                }
                                Type::TensorList => {
                                    format_f!("tensor *{param_ident}_data, int {param_ident}_len")
                                }
                                Type::TensorOptions => {
                                    format_f!("int {param_ident}_kind, int {param_ident}_device")
                                }
                                Type::String => {
                                    format_f!("char* {param_ident}_ptr, int {param_ident}_len")
                                }
                                Type::Int64Optional => {
                                    format_f!("int64_t {param_ident}_v, uint8_t {param_ident}_null")
                                }
                                Type::DoubleOptional => {
                                    format_f!("double {param_ident}_v, uint8_t {param_ident}_null")
                                }
                                Type::Bool => format_f!("int  {param_ident}"),
                                Type::Int64 => format_f!("int64_t {param_ident}"),
                                Type::Double => format_f!("double {param_ident}"),
                                Type::Tensor => format_f!("tensor {param_ident}"),
                                Type::TensorOptional => format_f!("tensor {param_ident}"),
                                Type::ScalarType => format_f!("int {param_ident}"),
                                Type::Device => format_f!("int {param_ident}"),
                                Type::Scalar => format_f!("scalar {param_ident}"),
                                Type::Other(_) => unreachable!(),
                            }
                        })
                        .join(", ")
                };

                let (header_fn, cpp_fn) = match item.return_kind {
                    ReturnKind::Tensor => {
                        let header_part =
                            format_f!("void {ffi_name}(tensor *out__, {cpp_fn_params});");
                        let cpp_part = [
                            format_f!("void {ffi_name}(tensor *out__, {cpp_fn_params}) {{"),
                            format_f!("    PROTECT("),
                            format_f!("        auto outputs__ = {cpp_call_params};"),
                            format_f!("        out__[0] = new torch::Tensor(outputs__);"),
                            format_f!("    )"),
                            format_f!("}}"),
                            format_f!(""),
                        ]
                        .join("\n");

                        (header_part, cpp_part)
                    }
                    ReturnKind::TensorArray(size) => {
                        let tensor_assign = {
                            let stmts: Vec<_> = (0..size)
                                .map(|index| {
                                    [
                                        format_f!("out__[{index}] = new torch::Tensor("),
                                        format_f!("    std::get<{index}>(outputs__)"),
                                        format_f!(");"),
                                    ]
                                    .join("\n")
                                })
                                .collect();
                            stmts.join("\n")
                        };
                        let header_part =
                            format_f!("void {ffi_name}(tensor *out__, {cpp_fn_params});");
                        let cpp_part = [
                            format_f!("void {ffi_name}(tensor *out__, {cpp_fn_params}) {{"),
                            format_f!("    PROTECT("),
                            format_f!("        auto outputs__ = {cpp_call_params};"),
                            format_f!("        {tensor_assign}"),
                            format_f!("    )"),
                            format_f!("}}"),
                            format_f!(""),
                        ]
                        .join("\n");

                        (header_part, cpp_part)
                    }
                    ReturnKind::TensorList => {
                        let header_part = format_f!("tensor *{ffi_name}({cpp_fn_params});");
                        let cpp_part = [
                            format_f!("tensor *{ffi_name}({cpp_fn_params}) {{"),
                            format_f!("    PROTECT("),
                            format_f!("        auto outputs__ = {cpp_call_params};"),
                            format_f!("        int sz = outputs__.size();"),
                            format_f!("        torch::Tensor **out__ =  (torch::Tensor**) \\"),
                            format_f!("             malloc((sz + 1) * sizeof(torch::Tensor*));"),
                            format_f!("        for (int i = 0; i < sz; ++i) {{"),
                            format_f!("            out__[i] = new torch::Tensor(outputs__[i]);"),
                            format_f!("        }}"),
                            format_f!("        out__[sz] = nullptr;"),
                            format_f!("        return out__;"),
                            format_f!("    )"),
                            format_f!("    return nullptr;"),
                            format_f!("}}"),
                            format_f!(""),
                        ]
                        .join("\n");
                        (header_part, cpp_part)
                    }
                    _ => unreachable!(),
                };

                (header_fn, cpp_fn)
            };

            Ok((header_fn, cpp_fn, ffi_fn, fallible_wrapper_fn, wrapper_fn))
        })
        .collect::<Fallible<Vec<_>>>()?
        .into_iter()
        .unzip_n_vec();

    let (header_fn_vec, cpp_fn_vec, ffi_fn_vec, fallible_wrapper_fn_vec, wrapper_fn_vec) =
        collected;

    let header_source = vec![
        format_f!("// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!"),
        format_f!(""),
    ]
    .into_iter()
    .chain(header_fn_vec.into_iter())
    .join("\n");

    let cpp_source = vec![
        format_f!("// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!"),
        format_f!(""),
    ]
    .into_iter()
    .chain(cpp_fn_vec.into_iter())
    .join("\n");

    let ffi_source = {
        let source = quote! {
            #[allow(clippy::all)]
            use crate::{C_scalar, C_tensor};
            use libc::c_int;

            extern "C" {
                #(#ffi_fn_vec)*
            }
        };

        format!(
            "/* THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND! */
{}",
            source
        )
    };

    let wrapper_source = {
        let source = quote! {
            #[allow(clippy::all)]
            use crate::{Device, Kind, Scalar, Tensor};
            use std::convert::Into;
            use std::borrow::Borrow;

            impl Tensor {
                #(#wrapper_fn_vec)*
            }
        };

        format!(
            "/* THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND! */
{}",
            source
        )
    };

    let fallible_wrapper_source = {
        let source = quote! {
            #[allow(clippy::all)]
            use torch_sys::*;
            use torch_sys::c_generated::*;
            use crate::{Device, Kind, Scalar, TchError, Tensor};
            use std::convert::Into;
            use std::borrow::Borrow;

            fn ptr_list<T: Borrow<Tensor>>(l: &[T]) -> Vec<*mut C_tensor> {
                l.iter().map(|x| x.borrow().c_tensor).collect()
            }

            impl Tensor {
                #(#fallible_wrapper_fn_vec)*
            }
        };

        format!(
            "/* THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND! */
{}",
            source
        )
    };

    let (header_file, cpp_file) = {
        let cpp_prefix = cpp_prefix.as_ref();
        let parent = cpp_prefix
            .parent()
            .ok_or_else(|| format_err!("invalid prefix '{}'", cpp_prefix.display()))?;
        let file_name = cpp_prefix
            .file_name()
            .ok_or_else(|| format_err!("invalid prefix '{}'", cpp_prefix.display()))?
            .to_str()
            .unwrap();
        let header_file = parent.join(format!("{}.h", file_name));
        let cpp_file = parent.join(format!("{}.cpp.h", file_name));
        (header_file, cpp_file)
    };

    fs::write(header_file, header_source)?;
    fs::write(cpp_file, cpp_source)?;
    fs::write(ffi_file, ffi_source)?;
    fs::write(wrapper_file, wrapper_source)?;
    fs::write(fallible_wrapper_file, fallible_wrapper_source)?;

    Ok(())
}

fn to_rust_name(c_name: &str) -> String {
    let name = REPLACE_MAP
        .get(c_name)
        .map(Deref::deref)
        .unwrap_or(c_name)
        .to_lowercase()
        .replace("__", "_");

    if "_".is_prefix_of(&name) {
        format!("internal{}", name)
    } else {
        name
    }
}

fn to_rust_args<'a, I>(args: I) -> impl Iterator<Item = TokenStream> + 'a
where
    I: IntoIterator<Item = &'a ArgumentExt<'a>>,
    I::IntoIter: 'a,
{
    args.into_iter().map(|arg| {
        let param_ident = &arg.rust_ident;
        let param_type = match arg.rust_type {
            Type::Bool => quote! { bool },
            Type::Int64 => {
                if arg.c_name == "reduction" {
                    quote! { crate::Reduction }
                } else {
                    quote! { i64 }
                }
            }
            Type::Double => quote! { f64 },
            Type::Tensor => quote! { &Tensor },
            Type::TensorOptional => quote! { Option<T> },
            Type::IntList => quote! { &[i64] },
            Type::TensorList => quote! { &[T] },
            Type::String => quote! { &str },
            Type::TensorOptions => quote! { (Kind, Device) },
            Type::Int64Optional => quote! { impl Into<Option<i64>> },
            Type::DoubleOptional => quote! { impl Into<Option<f64>> },
            Type::Scalar => quote! { S },
            Type::ScalarType => quote! { Kind },
            Type::Device => quote! { Device },
            Type::Other(_) => unreachable!(),
        };

        quote! { #param_ident: #param_type }
    })
}

fn to_c_args<'a, I>(args: I) -> String
where
    I: IntoIterator<Item = &'a ArgumentExt<'a>>,
    I::IntoIter: 'a,
{
    args.into_iter()
        .map(|arg| {
            let param_ident = format_ident!("{}", arg.c_name);
            match arg.rust_type {
                Type::Scalar => format_f!("*{param_ident}"),
                Type::Tensor => format_f!("*{param_ident}"),
                Type::TensorOptional => {
                    format_f!("({param_ident} ? *{param_ident} : torch::Tensor())")
                }
                Type::Bool => format_f!("(bool) {param_ident}"),
                Type::IntList => {
                    format_f!("torch::IntArrayRef({param_ident}_data, {param_ident}_len)")
                }
                Type::String => {
                    format_f!("std::string({param_ident}_ptr, {param_ident}_len)")
                }
                Type::TensorList => {
                    format_f!("of_carray_tensor({param_ident}_data, {param_ident}_len)")
                }
                Type::TensorOptions => [
                    format_f!("at::device(device_of_int({param_ident}_device))"),
                    format_f!("    .dtype(at::ScalarType({param_ident}_kind))"),
                ]
                .join("\n"),
                Type::Int64Optional => {
                    format_f!(
                        "{param_ident}_null ? c10::nullopt : c10::optional<int64_t>({param_ident}_v)"
                    )
                }
                Type::DoubleOptional => {
                    format_f!(
                        "{param_ident}_null ? c10::nullopt : c10::optional<double>({param_ident}_v)"
                    )
                }
                Type::ScalarType => {
                    format_f!("at::ScalarType({param_ident})")
                }
                Type::Device => {
                    format_f!("device_of_int({param_ident})")
                }
                Type::Int64 => {
                    format_f!("{param_ident}")
                }
                Type::Double => {
                    format_f!("{param_ident}")
                }
                Type::Other(_) => unreachable!(),
            }
        })
        .join(", ")
}

fn to_rust_return_type(kind: &ReturnKind) -> TokenStream {
    match *kind {
        ReturnKind::Tensor => quote! { Tensor },
        ReturnKind::TensorArray(size) => {
            let types = (0..size).map(|_| quote! { Tensor });
            quote! { (#(#types),*) }
        }
        ReturnKind::TensorList => quote! { Vec<Tensor> },
        _ => unreachable!(),
    }
}

fn to_rust_arg_type(item_key: &str, arg: &Argument) -> Result<Type, ParseArgumentError> {
    let rust_type = match (&arg.dynamic_type, arg.is_nullable, &arg.default) {
        (Type::Bool, _, _) => Type::Bool,
        (Type::Int64, false, _) => Type::Int64,
        (Type::Int64, true, _) => Type::Int64Optional,
        (Type::Double, false, _) => Type::Double,
        (Type::Double, true, _) => Type::DoubleOptional,
        (Type::Tensor, false, _) => Type::Tensor,
        (Type::Tensor, true, _) => Type::TensorOptional,
        (Type::TensorOptions, _, Some(_)) => {
            if NO_TENSOR_OPTIONS.contains(item_key) {
                return Err(ParseArgumentError::Skip);
            } else {
                Type::TensorOptions
            }
        }
        (Type::TensorOptions, _, None) => Type::TensorOptions,
        (Type::IntList, _, _) => Type::IntList,
        (Type::TensorList, _, _) => Type::TensorList,
        (Type::Device, _, _) => Type::Device,
        (Type::Scalar, false, Some(_)) => {
            return Err(ParseArgumentError::Skip);
        }
        (Type::Scalar, _, _) => Type::Scalar,
        (Type::ScalarType, _, _) => Type::ScalarType,
        (Type::String, _, _) => Type::String,
        (Type::Int64Optional, _, Some(_))
        | (Type::DoubleOptional, _, Some(_))
        | (Type::TensorOptional, _, Some(_))
        | (Type::Other(_), _, Some(_)) => {
            return Err(ParseArgumentError::Skip);
        }
        (Type::Int64Optional, _, None)
        | (Type::DoubleOptional, _, None)
        | (Type::TensorOptional, _, None)
        | (Type::Other(_), _, None) => {
            return Err(ParseArgumentError::NonSimple);
        }
    };

    Ok(rust_type)
}
