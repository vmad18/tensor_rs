use std::cell::UnsafeCell;
use std::fmt::Debug;
use std::ops::{Add, Deref, DerefMut, Div, Mul, Sub};
use std::thread;
use std::thread::JoinHandle;

use crate::tensor::Tensor;
use crate::utils::dtype::DType;
use crate::utils::{Print, ToRc, ToTensor};
use std::f32::consts::E;

#[derive(Debug)]
pub struct TensorOps {
    pub threaded: bool,
    pub auto: bool,
    pub thread_count: usize,
}

#[derive(Debug)]
pub enum Operation {
    Add,
    Mul,
    Div,
    MatMul,
}

pub trait TensorOpsInit {
    fn tnsr_op(self) -> TensorOps;
}

impl TensorOpsInit for (bool, bool) {
    fn tnsr_op(self) -> TensorOps {
        let (a1, a2) = self;
        TensorOps::init_ops(a1, a2)
    }
}

impl TensorOpsInit for (bool, bool, usize) {
    fn tnsr_op(self) -> TensorOps {
        let (a1, a2, a3) = self;
        TensorOps::init_ops_tc(a1, a2, a3)
    }
}

#[derive(Debug)]
pub struct SharedTensor<T> {
    data: UnsafeCell<T>,
}

unsafe impl<T> Sync for SharedTensor<T> {}
unsafe impl<T> Send for SharedTensor<T> {}

impl<T> SharedTensor<T> {
    pub fn new(data: T) -> Self {
        SharedTensor {
            data: UnsafeCell::new(data),
        }
    }

    pub fn get(&self) -> &T {
        unsafe { &*self.data.get() }
    }

    pub fn get_mut(&self) -> &mut T {
        unsafe { &mut *self.data.get() }
    }
}

impl<T> Deref for SharedTensor<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.data.get() }
    }
}

impl<T> DerefMut for SharedTensor<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.data.get() }
    }
}

impl TensorOps {
    pub fn new<I: TensorOpsInit>(init: I) -> Self {
        init.tnsr_op()
    }

    pub fn init_ops(threaded: bool, auto: bool) -> Self {
        TensorOps {
            threaded,
            auto,
            thread_count: 1,
        }
    }

    pub fn init_ops_tc(threaded: bool, auto: bool, thread_count: usize) -> Self {
        /*        let sys_threads = thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1); // leave one thread left for the system
                let thread_count = thread_count.min(sys_threads); // ensure that we don't fork bomb ourselves :p
        */
        TensorOps {
            threaded,
            auto,
            thread_count,
        }
    }

    pub fn match_tnsrs<T: DType>(
        &self,
        a: Tensor<T>,
        b: Tensor<T>,
        ignore_dim: Option<usize>,
    ) -> (Tensor<T>, Tensor<T>, Vec<usize>) {
        if a.rank() == b.rank() {
            let mut j = 0;
            for i in 0..a.rank() {
                if &a.shape[i] != &b.shape[i] {
                    break;
                }
                j += 1;
            }

            if j == a.rank() {
                let out_shape = (&a.shape).clone();
                return (a, b, out_shape);
            }
        }

        // larger - contains more elements | smaller - opposite of larger :)
        let (mut larger, mut smaller, repeats_l, repeats_s, same) = a
            .match_dims(b, ignore_dim)
            .expect("Could not add tensors- they did not match on any dimension!");

        // matches tensor shape w/ each other
        if !same {
            let _ = larger.repeat_dim(repeats_l.as_slice());
            let _ = smaller.repeat_dim(repeats_s.as_slice());
        }

        let out_shape = larger.clone().shape;
        (larger, smaller, out_shape)
    }

    fn pad<T: DType>(a: &mut Vec<T>, num: &usize) {
        let pad = vec![T::zero(); *num];
        a.extend(pad);
    }

    fn blck_compute_op<T: DType>(
        &self,
        a: Tensor<T>,
        b: Tensor<T>,
        op: fn(&usize, &usize, &Vec<T>, &Vec<T>, &mut Vec<T>),
    ) -> Tensor<T> {
        let BLCK_SIZE = (a.num_elms() / self.thread_count).max(1);

        let (a, b, shape) = self.match_tnsrs(a, b, None);

        let a_m = SharedTensor::new(a.data.clone());
        let b_m = SharedTensor::new(b.data.clone());
        let mut output = vec![T::zero(); a.num_elms()];

        let step_size: usize = if output.len() < BLCK_SIZE {
            output.len()
        } else {
            BLCK_SIZE
        };

        let pad_size = BLCK_SIZE - (output.len() % BLCK_SIZE);

        if a_m.get().len() > BLCK_SIZE && pad_size != BLCK_SIZE {
            TensorOps::pad(&mut a_m.get_mut(), &pad_size);
            TensorOps::pad(&mut b_m.get_mut(), &pad_size);
            TensorOps::pad(&mut output, &pad_size);
        }

        let out_ref = Box::new(SharedTensor::new(output)); // output.as_mut_ptr();
        let data_out_ref: &'static SharedTensor<_> = Box::leak(out_ref);

        let mut handles: Vec<JoinHandle<()>> = vec![];
        for i in (0..data_out_ref.get().len()).step_by(step_size) {
            let a_m_ref = SharedTensor::new(a_m.get()[i..i + step_size].to_vec()); // i don't like
                                                                                   // that I have
                                                                                   // to make a
                                                                                   // copy of the
                                                                                   // chunk of 'a/b'
            let b_m_ref = SharedTensor::new(b_m.get()[i..i + step_size].to_vec());
            let data_out_ref = data_out_ref;

            let handle = thread::spawn(move || {
                let mut value = unsafe { &mut *data_out_ref.data.get() };

                for j in i..(i + step_size) {
                    op(&j, &step_size, a_m_ref.get(), b_m_ref.get(), &mut value);
                }
            });

            handles.push(handle);
        }

        for h in handles {
            h.join().expect("panik!");
        }

        let final_out: Vec<T> = data_out_ref.get_mut().drain(..a.num_elms()).collect();

        Tensor::<T>::new(final_out.as_slice(), shape.as_slice())
    }

    fn compute_op<T: DType>(
        &self,
        a: Tensor<T>,
        b: Tensor<T>,
        op: fn(&usize, &usize, &Vec<T>, &Vec<T>, &mut Vec<T>),
    ) -> Tensor<T> {
        let (a, b, shape) = self.match_tnsrs(a, b, None);

        let a_m = a.clone().data;
        let b_m = b.clone().data;

        let mut output: Vec<T> = vec![T::zero(); a.num_elms()];

        for i in 0..output.len() {
            op(&i, &output.len(), &a_m, &b_m, &mut output);
        }

        Tensor::new(output.as_slice(), shape.as_slice())
    }

    pub fn publish_signal<T: DType>(
        &self,
        a: Tensor<T>,
        b: Tensor<T>,
        sig: fn(&usize, &usize, &Vec<T>, &Vec<T>, &mut Vec<T>),
    ) -> Tensor<T> {
        if self.threaded {
            self.blck_compute_op(a, b, sig)
        } else {
            self.compute_op(a, b, sig)
        }
    }

    pub fn add<T: DType>(&self, a: Tensor<T>, b: Tensor<T>) -> Tensor<T> {
        let add_sig = |i: &usize, step: &usize, a: &Vec<T>, b: &Vec<T>, o: &mut Vec<T>| {
            o[*i] = a[*i % step] + b[*i % step]
        };
        self.publish_signal(a, b, add_sig)
    }

    pub fn sub<T: DType>(&self, a: Tensor<T>, b: Tensor<T>) -> Tensor<T> {
        let add_sig = |i: &usize, step: &usize, a: &Vec<T>, b: &Vec<T>, o: &mut Vec<T>| {
            o[*i] = a[*i % step] - b[*i % step]
        };
        self.publish_signal(a, b, add_sig)
    }

    pub fn mul<T: DType>(&self, a: Tensor<T>, b: Tensor<T>) -> Tensor<T> {
        let mul_sig = |i: &usize, step: &usize, a: &Vec<T>, b: &Vec<T>, o: &mut Vec<T>| {
            o[*i] = a[*i % step] * b[*i % step]
        };
        self.publish_signal(a, b, mul_sig)
    }

    pub fn div<T: DType>(&self, a: Tensor<T>, b: Tensor<T>) -> Tensor<T> {
        let mul_sig = |i: &usize, step: &usize, a: &Vec<T>, b: &Vec<T>, o: &mut Vec<T>| {
            o[*i] = a[*i % step] / b[*i % step]
        };
        self.publish_signal(a, b, mul_sig)
    }

    pub fn sin(&self, a: Tensor<f32>, _b: Tensor<f32>) -> Tensor<f32> {
        let sin_sig = |i: &usize, step: &usize, a: &Vec<f32>, _b: &Vec<f32>, o: &mut Vec<f32>| {
            o[*i] = a[*i % step].sin()
        };
        self.publish_signal::<f32>(a, _b, sin_sig)
    }

    pub fn cos(&self, a: Tensor<f32>, _b: Tensor<f32>) -> Tensor<f32> {
        let sin_sig = |i: &usize, step: &usize, a: &Vec<f32>, _b: &Vec<f32>, o: &mut Vec<f32>| {
            o[*i] = a[*i % step].cos()
        };
        self.publish_signal::<f32>(a, _b, sin_sig)
    }

    pub fn tan(&self, a: Tensor<f32>, _b: Tensor<f32>) -> Tensor<f32> {
        let sin_sig = |i: &usize, step: &usize, a: &Vec<f32>, _b: &Vec<f32>, o: &mut Vec<f32>| {
            o[*i] = a[*i % step].tan()
        };
        self.publish_signal::<f32>(a, _b, sin_sig)
    }

    fn _exp(&self, a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32> {
        let exp_sig = |i: &usize, step: &usize, a: &Vec<f32>, b: &Vec<f32>, o: &mut Vec<f32>| {
            o[*i] = (a[*i % step] * b[*i % step].ln()).exp();
        };
        self.publish_signal(a, b, exp_sig)
    }

    pub fn exp(&self, a: Tensor<f32>) -> Tensor<f32> {
        let b: Tensor<f32> = E.tnsr();
        self._exp(a, b)
    }

    pub fn pow(&self, a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32> {
        self._exp(a, b)
    }

    pub fn asin(&self, a: Tensor<f32>, _b: Tensor<f32>) -> Tensor<f32> {
        let sin_sig = |i: &usize, step: &usize, a: &Vec<f32>, _b: &Vec<f32>, o: &mut Vec<f32>| {
            o[*i] = a[*i % step].asin()
        };
        self.publish_signal::<f32>(a, _b, sin_sig)
    }

    pub fn acos(&self, a: Tensor<f32>, _b: Tensor<f32>) -> Tensor<f32> {
        let sin_sig = |i: &usize, step: &usize, a: &Vec<f32>, _b: &Vec<f32>, o: &mut Vec<f32>| {
            o[*i] = a[*i % step].acos()
        };
        self.publish_signal::<f32>(a, _b, sin_sig)
    }

    pub fn atan(&self, a: Tensor<f32>, _b: Tensor<f32>) -> Tensor<f32> {
        let sin_sig = |i: &usize, step: &usize, a: &Vec<f32>, _b: &Vec<f32>, o: &mut Vec<f32>| {
            o[*i] = a[*i % step].atan()
        };
        self.publish_signal::<f32>(a, _b, sin_sig)
    }
}
