pub trait Shape {
    fn to_shape(&self) -> Box<[i64]>;
}

macro_rules! impl_shape {
    ($v:expr) => {
        impl Shape for [i64; $v] {
            fn to_shape(&self) -> Box<[i64]> {
                Box::new(*self)
            }
        }
    };
}

impl_shape!(0);
impl_shape!(1);
impl_shape!(2);
impl_shape!(3);
impl_shape!(4);
impl_shape!(5);
impl_shape!(6);

impl Shape for () {
    fn to_shape(&self) -> Box<[i64]> {
        Box::new([])
    }
}

impl Shape for &[i64] {
    fn to_shape(&self) -> Box<[i64]> {
        (*self).into()
    }
}

impl Shape for i64 {
    fn to_shape(&self) -> Box<[i64]> {
        Box::new([*self])
    }
}

impl Shape for usize {
    fn to_shape(&self) -> Box<[i64]> {
        Box::new([*self as i64])
    }
}

impl Shape for i32 {
    fn to_shape(&self) -> Box<[i64]> {
        Box::new([i64::from(*self)])
    }
}

impl Shape for (i64,) {
    fn to_shape(&self) -> Box<[i64]> {
        Box::new([self.0])
    }
}

impl Shape for (i64, i64) {
    fn to_shape(&self) -> Box<[i64]> {
        Box::new([self.0, self.1])
    }
}

impl Shape for (i64, i64, i64) {
    fn to_shape(&self) -> Box<[i64]> {
        Box::new([self.0, self.1, self.2])
    }
}

impl Shape for (i64, i64, i64, i64) {
    fn to_shape(&self) -> Box<[i64]> {
        Box::new([self.0, self.1, self.2, self.3])
    }
}
