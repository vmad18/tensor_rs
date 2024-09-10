pub mod tensor;
pub mod utils;

use utils::{
    dtype::Complex32,
    ops::{relu, sigmoid, Function},
};

use crate::tensor::Tensor;
use crate::utils::dtype::DType;

use std::f32::consts::PI;
use std::time::SystemTime;

use crate::utils::Print;

pub fn tensor_tests() {
    let shape = vec![128, 128, 64];
    let tnsr = Tensor::<f32>::new_ones(shape.as_slice());
    let tnsr2 = Tensor::<f32>::new_ones(shape.as_slice());

    let tnsr3 = Tensor::<f32>::new(&[2., 2., 1., 1.], &[2, 2]);
    let tnsr4 = Tensor::<f32>::new(&[2., 3., 4., 5.], &[2, 2]);
    let tnsr5 = Tensor::<f32>::new(
        &[
            PI / 2.,
            PI / 6.,
            PI / 3.,
            PI / 4.,
            3. * PI / 2.,
            0.,
            PI,
            3. * PI / 4.,
        ],
        &[2, 2, 2],
    );

    let mut tnsr6 = Tensor::<f32>::new_ones(&[2, 1]);
    let tnsr7 = Tensor::<f32>::new_ones(&[2, 1]);

    let mut tnsr8 = Tensor::<f32>::new(&[1., 3., 6., 7.], &[2, 2]);
    let tnsr9 = Tensor::<f32>::new(&[2., 5., 3., 1.], &[2, 2]);

    let mut tnsr10 = Tensor::<f32>::new(&[-1., 0., 0., 1.], &[2, 2]);
    let tnsr11 = Tensor::<f32>::new_ones(&[2, 1]);

    let tnsr12 = Tensor::<f32>::new(&[3., 4., 10., 11.], &[2, 2]);

    let now = SystemTime::now();

    println!("Matched dim operation test...");
    let _ = tnsr + tnsr2; // don't print this out lol
    println!();
    println!("Scalar mult result...");
    (tnsr3 / 3.0.tnsr()).print();
    println!();
    println!("Inner (dot) product test result...");
    (tnsr6.dot(tnsr7.clone())).unwrap().print();
    println!();
    println!("Outer product test result...");
    (tnsr6.outer(tnsr7)).unwrap().print();
    println!();
    println!("Trig func test...");
    tnsr5.clone().sin().print();
    tnsr5.sin().asin().print();
    println!();
    println!("Matmul test...");
    let mut result = tnsr8.mm(tnsr9.clone()).unwrap();
    result.print();
    println!();
    println!("Flatten test...");
    result.flatten(0, true).unwrap();
    result.print();
    println!();
    println!("Tensor sum test...");
    result.sum(0, false, false).unwrap().unwrap().print();
    println!();
    println!("Tensor power test...");
    tnsr4.clone().pow(&3.0_f32.tnsr()).print();
    println!();
    println!("Tensor exp test...");
    tnsr4.exp().print();
    println!();
    println!("Linear transformation test...");
    tnsr10.mm(tnsr11).unwrap().print();
    println!();
    println!("Tensor to complex test...");
    let c_tnsr = tnsr12.as_cmplx().unwrap();
    c_tnsr.print();
    (c_tnsr.clone() * c_tnsr).print();

    let elapsed = SystemTime::now().duration_since(now).unwrap();
    println!();
    println!("Finished | elapsed time: {:.5?}", elapsed);
}

pub fn dtypes_test() {
    let c1 = Complex32::new(1., 5.);
    let c2 = Complex32::new(3., 4.);
    println!("{:?}", c1 * c2);
    println!("{:?}", c1 * 5.);
}

pub fn ops_test() {
    let t1 = Tensor::<f32>::new_grad(&[1., -3., 4., -5.], &[2, 2]);
    t1.print();
    let s_t1 = sigmoid.call(t1.clone(), None);
    s_t1.print();
    // println!("{}", s_t1);
    // println!();
    // relu.call(t1.clone(), None).print();
}

pub fn grad_test() {
    let a = Tensor::new_grad(&[2, 3, 5, 7], &[2, 2]);
    let b = Tensor::new_grad(&[6, 2, 1, 9], &[2, 2]);
    // println!("{}", a.requires_grad());
    let mut c = a.add(&b);
    let d = a.add(&c);

    // c.print();
    // println!();
    // d.print();

    c.flatten(0, true);
    // c.sum(0, true, true);
    c.print();
}
