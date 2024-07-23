use std::cell::UnsafeCell;
use std::fmt::Debug;
use std::ops::{Add, Deref, DerefMut, Div, Mul, Sub};
use std::thread;
use std::thread::JoinHandle;

use crate::tensor::Tensor;
use crate::utils::dtype::DType;
use crate::utils::Print;

#[derive(Debug)]
pub struct TensorOps {
    pub threaded: bool,
    pub auto: bool,
    pub thread_count: usize,
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
            data: UnsafeCell::new(data)
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

    pub fn match_tnsrs<'a, T: DType>(
        &self,
        a: Tensor<'a, T>,
        b: Tensor<'a, T>,
        ignore_dim: Option<usize>,
    ) -> (Tensor<'a, T>, Tensor<'a, T>, Vec<usize>) {

        if a.rank() == b.rank() {
            let mut j = 0;
            for i in 0..a.rank() {
                if &a.shape[i] != &b.shape[i] {
                    break;
                }
                j+=1;
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

    fn pad<'a, T: DType>(a: &mut Vec<T>, num: &usize) {
        let pad = vec![T::zero(); *num];
        a.extend(pad);
    }

    fn blck_compute_op<'a, T: DType>(
        &self,
        a: Tensor<'a, T>,
        b: Tensor<'a, T>,
        op: fn(&usize, &usize, &Vec<T>, &Vec<T>, &mut Vec<T>)) -> Tensor<'a, T> {
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
            let a_m_ref = SharedTensor::new(a_m.get()[i..i + step_size].to_vec());
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

        for h in handles { h.join().expect("panik!"); }

        let final_out: Vec<T> = data_out_ref.get_mut()
            .drain(..a.num_elms())
            .collect();

        Tensor::<T>::new(final_out.as_slice(), shape.as_slice())
    }

    fn compute_op<'a, T: DType>(
        &self,
        a: Tensor<'a, T>,
        b: Tensor<'a, T>,
        op: fn(&usize, &usize, &Vec<T>, &Vec<T>, &mut Vec<T>)) -> Tensor<'a, T> {
        let (a, b, shape) = self.match_tnsrs(a, b, None);

        let a_m = a.clone().data;
        let b_m = b.clone().data;

        let mut output: Vec<T> = vec![T::zero(); a.num_elms()];

        for i in 0..output.len() { op(&i, &output.len(), &a_m, &b_m, &mut output); }

        Tensor::new(output.as_slice(), shape.as_slice())
    }

    pub fn publish_signal<'a, T: DType>(&self,
                                        a: Tensor<'a, T>,
                                        b: Tensor<'a, T>,
                                        sig: fn(&usize, &usize, &Vec<T>, &Vec<T>, &mut Vec<T>)) -> Tensor<'a, T> {
        if self.threaded {
            self.blck_compute_op(a, b, sig)
        } else {
            self.compute_op(a, b, sig)
        }
    }

    pub fn add<'a, T: DType>(
        &self,
        a: Tensor<'a, T>,
        b: Tensor<'a, T>,
    ) -> Tensor<'a, T> {
        let add_sig = |i: &usize, step: &usize, a: &Vec<T>, b: &Vec<T>, o: &mut Vec<T>| o[*i] = a[*i % step] + b[*i % step];
        self.publish_signal(a, b, add_sig)
    }

    pub fn sub<'a, T: DType>(
        &self,
        a: Tensor<'a, T>,
        b: Tensor<'a, T>,
    ) -> Tensor<'a, T> {
        let add_sig = |i: &usize, step: &usize, a: &Vec<T>, b: &Vec<T>, o: &mut Vec<T>| o[*i] = a[*i % step] - b[*i % step];
        self.publish_signal(a, b, add_sig)
    }

    pub fn mul<'a, T: DType>(
        &self,
        a: Tensor<'a, T>,
        b: Tensor<'a, T>,
    ) -> Tensor<'a, T> {
        let mul_sig = |i: &usize, step: &usize, a: &Vec<T>, b: &Vec<T>, o: &mut Vec<T>| o[*i] = a[*i % step] * b[*i % step];
        self.publish_signal(a, b, mul_sig)
    }

    pub fn div<'a, T: DType>(
        &self,
        a: Tensor<'a, T>,
        b: Tensor<'a, T>,
    ) -> Tensor<'a, T> {
        let mul_sig = |i: &usize, step: &usize, a: &Vec<T>, b: &Vec<T>, o: &mut Vec<T>| o[*i] = a[*i % step] / b[*i % step];
        self.publish_signal(a, b, mul_sig)
    }

    pub fn sin<'a>(
        &self,
        a: Tensor<'a, f32>,
        b: Tensor<'a, f32>,
    ) -> Tensor<'a, f32> {
        let sin_sig = |i: &usize, step: &usize, a: &Vec<f32>, b: &Vec<f32>, o: &mut Vec<f32>| o[*i] = a[*i % step].sin();
        self.publish_signal::<f32>(a, b, sin_sig)
    }

    pub fn cos<'a>(
        &self,
        a: Tensor<'a, f32>,
        b: Tensor<'a, f32>,
    ) -> Tensor<'a, f32> {
        let sin_sig = |i: &usize, step: &usize, a: &Vec<f32>, b: &Vec<f32>, o: &mut Vec<f32>| o[*i] = a[*i % step].cos();
        self.publish_signal::<f32>(a, b, sin_sig)
    }

    pub fn tan<'a>(
        &self,
        a: Tensor<'a, f32>,
        b: Tensor<'a, f32>,
    ) -> Tensor<'a, f32> {
        let sin_sig = |i: &usize, step: &usize, a: &Vec<f32>, b: &Vec<f32>, o: &mut Vec<f32>| o[*i] = a[*i % step].tan();
        self.publish_signal::<f32>(a, b, sin_sig)
    }
}
