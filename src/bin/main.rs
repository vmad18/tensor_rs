use std::f32::consts::PI;
use std::time::SystemTime;

use tensor::tensor::Tensor;
use tensor::utils::{Print, ToTensor};

fn main() {
    let shape = vec![128, 128, 64];
    let tnsr = Tensor::<f32>::new_ones(shape.as_slice());
    let tnsr2 = Tensor::<f32>>::new_ones(shape.as_slice());
    
    let mut tnsr3 =  Tensor::<f32>::new(&[1., 1., 1., 1.], &[2, 2]); 
    let mut tnsr4 =  Tensor::<f32>::new(&[1., 1., 1., 1.], &[2, 2]);
    let mut tnsr5 = Tensor::<f32>::new(&[PI / 2., PI / 6., PI / 3., PI / 4., 3. * PI / 2., 0., PI, 3. * PI / 4.], &[2, 2, 2]);

    let mut tnsr6 = Tensor::<f32>::new_ones(&[2, 1]);
    let mut tnsr7 = Tensor::<f32>::new_ones(&[2, 1]);

    let mut tnsr8 = Tensor::<f32>::new(&[1., 3., 6., 7.],  &[2, 2]);
    let mut tnsr9 = Tensor::<f32>::new(&[2., 5., 3., 1.],  &[2, 2]);

    let now = SystemTime::now();

    (tnsr + tnsr2).print();
    (tnsr3 / 3.0.tnsr()).print();
    (tnsr6.dot(tnsr7)).print();
    (tnsr6.outer(tnsr7)).print();

    tnsr5.sin().print();
    
    let mut result = tnsr8.mm(tnsr9.clone()).unwrap();
    result.flatten(0, true).unwrap();
    result.sum(0, false, false).unwrap().unwrap().print();

    let elapsed = SystemTime::now().duration_since(now).unwrap();

    println!("{:.5?}", elapsed);
}
