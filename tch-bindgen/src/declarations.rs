use crate::common::*;

#[derive(Debug, Clone, Deserialize)]
pub struct Declarations(pub Vec<Item>);

impl Declarations {
    pub fn items(&self) -> &[Item] {
        self.0.as_slice()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Item {
    pub name: String,
    pub operator_name: String,
    pub overload_name: String,
    pub use_c10_dispatcher: UseC10Dispatcher,
    pub manual_kernel_registration: bool,
    pub category_override: String,
    pub matches_jit_signature: bool,
    pub schema_string: String,
    pub arguments: Vec<Argument>,
    pub schema_order_cpp_signature: String,
    pub schema_order_arguments: Vec<Argument>,
    pub method_of: HashSet<String>,
    pub mode: Mode,
    pub python_module: String,
    pub returns: Vec<Return>,
    pub inplace: bool,
    pub is_factory_method: bool,
    pub r#abstract: bool,
    pub device_guard: bool,
    pub with_gil: bool,
    pub deprecated: bool,
    pub has_math_kernel: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Argument {
    pub annotation: String,
    pub dynamic_type: Type,
    pub is_nullable: bool,
    pub name: String,
    pub r#type: String,
    pub default: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Return {
    pub dynamic_type: Type,
    pub name: String,
    pub r#type: Type,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub enum Mode {
    #[serde(rename = "native")]
    Native,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub enum UseC10Dispatcher {
    #[serde(rename = "full")]
    Full,
    #[serde(rename = "with_codegenerated_unboxing_wrapper")]
    WithCodegeneratedUnboxingWrapper,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Bool,
    Int64,
    Int64Optional,
    Double,
    DoubleOptional,
    Tensor,
    TensorOptional,
    TensorList,
    TensorOptions,
    Scalar,
    ScalarType,
    Device,
    String,
    IntList,
    Other(String),
}

impl<'de> Deserialize<'de> for Type {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
        let type_ = match text.as_str() {
            "bool" => Type::Bool,
            "int64_t" => Type::Int64,
            "c10::optional<int64_t>" => Type::Int64Optional,
            "double" => Type::Double,
            "c10::optional<double>" => Type::DoubleOptional,
            "Tensor" => Type::Tensor,
            "c10::optional<Tensor>" => Type::TensorOptional,
            "TensorList" => Type::TensorList,
            "TensorOptions" => Type::TensorOptions,
            "Scalar" => Type::Scalar,
            "ScalarType" => Type::ScalarType,
            "Device" => Type::Device,
            "std::string" => Type::String,
            "IntArrayRef" => Type::IntList,
            _ => Type::Other(text),
        };
        Ok(type_)
    }
}
