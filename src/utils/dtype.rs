use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

// TODO create my own FP16- if i can
// one fp32 == 2 fp16
pub trait DType: Debug + Clone + Copy + Send + Sync + 'static + Add<Output=Self> + Sub<Output=Self> + Mul<Output=Self> + Div<Output=Self> {
    fn zero() -> Self;
    fn one() -> Self;
    fn sin(self) -> f32;
    fn cos(self) -> f32;
    fn tan(self) -> f32;
    fn to_fp32(self) -> f32;
    fn to_fp64(self) -> f64;
}

impl DType for u8 {
    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }

    fn sin(self) -> f32 {
        (self as f32).sin()
    }

    fn cos(self) -> f32 {
        (self as f32).cos()
    }

    fn tan(self) -> f32 {
        (self as f32).tan()
    }

    fn to_fp32(self) -> f32 {
        self as f32
    }

    fn to_fp64(self) -> f64 {
        self as f64
    }
}

impl DType for i8 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }

    fn sin(self) -> f32 {
        (self as f32).sin()
    }

    fn cos(self) -> f32 {
        (self as f32).cos()
    }

    fn tan(self) -> f32 {
        (self as f32).tan()
    }

    fn to_fp32(self) -> f32 {
        self as f32
    }

    fn to_fp64(self) -> f64 {
        self as f64
    }
}

impl DType for i32 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn sin(self) -> f32 {
        (self as f32).sin()
    }

    fn cos(self) -> f32 {
        (self as f32).cos()
    }

    fn tan(self) -> f32 {
        (self as f32).tan()
    }

    fn to_fp32(self) -> f32 {
        self as f32
    }

    fn to_fp64(self) -> f64 {
        self as f64
    }
}

impl DType for i64 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn sin(self) -> f32 {
        (self as f32).sin()
    }

    fn cos(self) -> f32 {
        (self as f32).cos()
    }

    fn tan(self) -> f32 {
        (self as f32).tan()
    }

    fn to_fp32(self) -> f32 {
        self as f32
    }

    fn to_fp64(self) -> f64 {
        self as f64
    }
}

impl DType for usize {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn sin(self) -> f32 {
        (self as f32).sin()
    }

    fn cos(self) -> f32 {
        (self as f32).cos()
    }

    fn tan(self) -> f32 {
        (self as f32).tan()
    }

    fn to_fp32(self) -> f32 {
        self as f32
    }

    fn to_fp64(self) -> f64 {
        self as f64
    }
}

impl DType for f32 {
    fn zero() -> Self {
        0.
    }
    fn one() -> Self {
        1.
    }
    fn sin(self) -> f32 {
        self.sin()
    }

    fn cos(self) -> f32 {
        self.cos()
    }

    fn tan(self) -> f32 {
        self.tan()
    }

    fn to_fp32(self) -> f32 {
        self
    }

    fn to_fp64(self) -> f64 {
        self as f64
    }
}

impl DType for f64 {
    fn zero() -> Self {
        0.
    }
    fn one() -> Self {
        1.
    }
    fn sin(self) -> f32 {
        self.sin() as f32
    }

    fn cos(self) -> f32 {
        self.cos() as f32
    }

    fn tan(self) -> f32 {
        self.tan() as f32
    }

    fn to_fp32(self) -> f32 {
        self as f32
    }

    fn to_fp64(self) -> f64 {
        self
    }
}