use crate::{Kind, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BasicKind {
    Float,
    Int,
    Bool,
    Complex,
}

impl BasicKind {
    fn for_tensor(t: &Tensor) -> BasicKind {
        match t.f_kind() {
            Err(_) => BasicKind::Complex,
            Ok(kind) => match kind {
                Kind::Int | Kind::Int8 | Kind::Uint8 | Kind::Int16 | Kind::Int64 => BasicKind::Int,
                Kind::BFloat16
                | Kind::QInt8
                | Kind::QUInt8
                | Kind::QInt32
                | Kind::Half
                | Kind::Float
                | Kind::Double => BasicKind::Float,
                Kind::Bool => BasicKind::Bool,
                Kind::ComplexHalf | Kind::ComplexFloat | Kind::ComplexDouble => BasicKind::Complex,
            },
        }
    }

    fn _is_floating_point(&self) -> bool {
        match self {
            BasicKind::Float => true,
            BasicKind::Bool | BasicKind::Int | BasicKind::Complex => false,
        }
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.defined() {
            match self.f_kind() {
                Err(err) => write!(f, "Tensor[{:?}, {:?}]", self.size(), err),
                Ok(kind) => {
                    let (is_int, is_float) = match kind {
                        Kind::Int | Kind::Int8 | Kind::Uint8 | Kind::Int16 | Kind::Int64 => {
                            (true, false)
                        }
                        Kind::BFloat16
                        | Kind::QInt8
                        | Kind::QUInt8
                        | Kind::QInt32
                        | Kind::Half
                        | Kind::Float
                        | Kind::Double => (false, true),
                        Kind::Bool
                        | Kind::ComplexHalf
                        | Kind::ComplexFloat
                        | Kind::ComplexDouble => (false, false),
                    };
                    match (self.size().as_slice(), is_int, is_float) {
                        ([], true, false) => write!(f, "[{}]", i64::from(self)),
                        ([s], true, false) if *s < 10 => write!(f, "{:?}", Vec::<i64>::from(self)),
                        ([], false, true) => write!(f, "[{}]", f64::from(self)),
                        ([s], false, true) if *s < 10 => write!(f, "{:?}", Vec::<f64>::from(self)),
                        _ => write!(f, "Tensor[{:?}, {:?}]", self.size(), self.f_kind()),
                    }
                }
            }
        } else {
            write!(f, "Tensor[Undefined]")
        }
    }
}

/// Options for Tensor pretty printing
pub struct PrinterOptions {
    precision: usize,
    threshold: usize,
    edge_items: usize,
    line_width: usize,
    sci_mode: Option<bool>,
}

lazy_static! {
    static ref PRINT_OPTS: std::sync::Mutex<PrinterOptions> =
        std::sync::Mutex::new(Default::default());
}

pub fn set_print_options(options: PrinterOptions) {
    *PRINT_OPTS.lock().unwrap() = options
}

pub fn set_print_options_default() {
    *PRINT_OPTS.lock().unwrap() = Default::default()
}

pub fn set_print_options_short() {
    *PRINT_OPTS.lock().unwrap() = PrinterOptions {
        precision: 2,
        threshold: 1000,
        edge_items: 2,
        line_width: 80,
        sci_mode: None,
    }
}

pub fn set_print_options_full() {
    *PRINT_OPTS.lock().unwrap() = PrinterOptions {
        precision: 4,
        threshold: usize::MAX,
        edge_items: 3,
        line_width: 80,
        sci_mode: None,
    }
}

impl Default for PrinterOptions {
    fn default() -> Self {
        Self { precision: 4, threshold: 1000, edge_items: 3, line_width: 80, sci_mode: None }
    }
}

struct FloatFormatter {
    int_mode: bool,
    sci_mode: bool,
    max_width: usize,
    precision: usize,
}

impl FloatFormatter {
    fn new(t: &Tensor) -> Self {
        let mut int_mode = true;
        let mut sci_mode = false;

        let _guard = crate::no_grad_guard();
        let t = t.to_device(crate::Device::Cpu);

        let nonzero_finite_vals = {
            let t = t.reshape(&[-1]);
            t.masked_select(&t.isfinite().logical_and(&t.ne(0.)))
        };

        if nonzero_finite_vals.numel() > 0 {
            let nonzero_finite_abs = nonzero_finite_vals.abs();
            let nonzero_finite_min = nonzero_finite_abs.min().double_value(&[]);
            let nonzero_finite_max = nonzero_finite_abs.max().double_value(&[]);

            let values = Vec::<f64>::from(nonzero_finite_vals);
            for value in values {
                if value.ceil() != value {
                    int_mode = false;
                    break;
                }
            }

            sci_mode = nonzero_finite_max / nonzero_finite_min > 1000.
                || nonzero_finite_max > 1e8
                || nonzero_finite_min < 1e-4
        }

        let print_opts = PRINT_OPTS.lock().unwrap();
        match print_opts.sci_mode {
            None => {}
            Some(v) => sci_mode = v,
        }
        Self { int_mode, sci_mode, max_width: 1, precision: print_opts.precision }
    }

    fn fmt(&self, v: f64, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.sci_mode {
            write!(f, "{v:width$.prec$e}", v = v, width = self.max_width, prec = self.precision)
        } else if self.int_mode {
            if v.is_finite() {
                write!(f, "{v:width$.0}.", v = v, width = self.max_width)
            } else {
                write!(f, "{v:width$.0}", v = v, width = self.max_width)
            }
        } else {
            write!(f, "{v:width$.prec$}", v = v, width = self.max_width, prec = self.precision)
        }
    }
}

impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.defined() {
            let basic_kind = BasicKind::for_tensor(self);
            match self.dim() {
                0 => match basic_kind {
                    BasicKind::Int => {
                        let value = self.int64_value(&[]);
                        write!(f, "{}", value)
                    }
                    BasicKind::Float => {
                        let formatter = FloatFormatter::new(self);
                        let value = self.double_value(&[]);
                        formatter.fmt(value, f)
                    }
                    BasicKind::Bool => {
                        let value = if self.int64_value(&[]) != 0 { "true" } else { "false" };
                        write!(f, "{}", value)
                    }
                    BasicKind::Complex => write!(f, "Tensor[{:?}, Complex]", self.size()),
                },
                _ => write!(f, "Tensor[TODO]"),
            }
        } else {
            write!(f, "Tensor[Undefined]")
        }
    }
}
