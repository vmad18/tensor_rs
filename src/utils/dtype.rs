use core::fmt;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

// TODO create my own FP16- if i can
// one fp32 == 2 fp16
pub trait DType:
    Debug
    + Clone
    + Copy
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn sin(self) -> f32;
    fn cos(self) -> f32;
    fn tan(self) -> f32;
    fn to_fp32(self) -> f32;
    fn to_fp64(self) -> f64;
    fn to_cmplx(self) -> Complex32;
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

    fn to_cmplx(self) -> Complex32 {
        Complex32::new(self.to_fp32(), 0.)
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

    fn to_cmplx(self) -> Complex32 {
        Complex32::new(self.to_fp32(), 0.)
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

    fn to_cmplx(self) -> Complex32 {
        Complex32::new(self.to_fp32(), 0.)
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

    fn to_cmplx(self) -> Complex32 {
        Complex32::new(self.to_fp32(), 0.)
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

    fn to_cmplx(self) -> Complex32 {
        Complex32::new(self.to_fp32(), 0.)
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

    fn to_cmplx(self) -> Complex32 {
        Complex32::new(self, 0.)
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

    fn to_cmplx(self) -> Complex32 {
        Complex32::new(self.to_fp32(), 0.)
    }
}

impl DType for Complex32 {
    fn zero() -> Self {
        Complex32::new(0., 0.)
    }

    fn one() -> Self {
        Complex32::new(1., 0.)
    }

    fn sin(self) -> f32 {
        (self.mag()).sin()
    }

    fn cos(self) -> f32 {
        (self.mag()).cos()
    }

    fn tan(self) -> f32 {
        (self.mag()).tan()
    }

    fn to_fp32(self) -> f32 {
        self.mag()
    }

    fn to_fp64(self) -> f64 {
        self.mag() as f64
    }

    fn to_cmplx(self) -> Complex32 {
        Complex32::new(self.to_fp32(), 0.)
    }
}

#[derive(Copy, Clone)]
pub struct Complex32 {
    r: f32,
    j: f32,
}

impl Complex32 {
    pub fn new(r: f32, j: f32) -> Self {
        Complex32 { r, j }
    }

    pub fn euler(m: f32, theta: f32) -> Self {
        Complex32 {
            r: m * theta.cos(),
            j: m * theta.sin(),
        }
    }

    pub fn c_mul(&self, other: &Complex32) -> Complex32 {
        let new_r: f32 = self.r * other.r - self.j * other.j;
        let new_j: f32 = self.r * other.j + other.r * self.j;
        Complex32::new(new_r, new_j)
    }

    pub fn s_mul<T: DType>(&self, other: T) -> Complex32 {
        Complex32::new(other.to_fp32() * self.r, other.to_fp32() * self.j)
    }

    pub fn mag(&self) -> f32 {
        (self.r * self.r + self.j * self.j).sqrt()
    }

    pub fn conj(&self) -> Complex32 {
        Complex32::new(self.r, -self.j)
    }

    pub fn resp(&self) -> Complex32 {
        Complex32::new(1. / self.r, 1. / self.j)
    }
}

impl Add for Complex32 {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Complex32::new(self.r + other.r, self.j + other.j)
    }
}

impl Mul<Complex32> for Complex32 {
    type Output = Self;
    fn mul(self, other: Complex32) -> Self::Output {
        self.c_mul(&other)
    }
}

impl Mul<f32> for Complex32 {
    type Output = Complex32;
    fn mul(self, other: f32) -> Self::Output {
        self.s_mul(other)
    }
}

impl Sub for Complex32 {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self + (other * -1.)
    }
}

impl Div for Complex32 {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        self * other.resp()
    }
}

impl fmt::Debug for Complex32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}+j{:.4}", self.r, self.j)
    }
}
