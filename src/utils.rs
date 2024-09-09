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
