use crate::tensor::{Slice, Tensor};
use dtype::DType;
use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;

pub mod consts;
pub mod dtype;
pub mod ops;
mod test;

pub trait Print {
    fn print(&self);
}

impl<T: Debug> Print for Vec<T> {
    fn print(&self) {
        println!("{:?}", self);
    }
}

pub trait ToTensor<T: DType> {
    fn tnsr(self) -> Tensor<T>;
}

impl ToTensor<i32> for i32 {
    fn tnsr(self) -> Tensor<i32> {
        Tensor::<i32>::new(&[self], &[1])
    }
}

impl ToTensor<i64> for i64 {
    fn tnsr(self) -> Tensor<i64> {
        Tensor::<i64>::new(&[self], &[1])
    }
}

/*impl ToTensor<u16> for u16 {
    fn tnsr(self) -> Tensor<u16> {
        Tensor::<u16>::new(&[self], &[1])
    }
}
*/
impl ToTensor<i8> for i8 {
    fn tnsr(self) -> Tensor<i8> {
        Tensor::<i8>::new(&[self], &[1])
    }
}

impl ToTensor<usize> for usize {
    fn tnsr(self) -> Tensor<usize> {
        Tensor::<usize>::new(&[self], &[1])
    }
}

impl ToTensor<f32> for f32 {
    fn tnsr(self) -> Tensor<f32> {
        Tensor::<f32>::new(&[self], &[1])
    }
}

impl ToTensor<f64> for f64 {
    fn tnsr(self) -> Tensor<f64> {
        Tensor::<f64>::new(&[self], &[1])
    }
}

pub trait ToSlice {
    fn to_slice(&self) -> Slice;
}

impl ToSlice for Vec<(usize, usize)> {
    fn to_slice(&self) -> Slice {
        Slice::new(self.as_slice())
    }
}

/*impl ToTensor<u32> for u32 {
    fn tnsr(self) -> Tensor<u32> {
        Tensor::<u32>::new(&[self], &[1])
    }
}*/

pub trait ToRc {
    fn to_rc(self) -> Rc<RefCell<Self>>;
}

impl<T: DType> ToRc for Tensor<T> {
    fn to_rc(self) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(self))
    }
}
