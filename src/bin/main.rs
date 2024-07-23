use std::f32::consts::PI;
use std::time::SystemTime;

use tensor::tensor::Tensor;
use tensor::utils::{Print, ToTensor};

fn main() {
    let shape = vec![128, 128, 64];
    let mut tnsr =  Tensor::<f32>::new_zeros(shape.as_slice()); //Tensor::<f32>::new(&[PI / 2., PI / 6., PI / 3., PI / 4., 3. * PI / 2., 0., PI, 3. * PI / 4.], &[2, 2, 2]); //  Tensor::<f32>::new_zeros(&[128, 128, 64]); //  Tensor::<f32>::new(&[PI / 2., PI / 6., PI / 3., PI / 4., 3. * PI / 2., 0., PI, 3. * PI / 4.], &[2, 2, 2]); // Tensor::<f32>::new_zeros(&[32, 255, 255, 64]); //Tensor::<i32>::new(&[1, 2, 3, 4, 5, 6, 7, 8], &[2, 2, 2]); //
    let mut tnsr2 =   Tensor::<f32>::new_zeros(shape.as_slice()); //Tensor::<f32>::new_zeros(&[128, 64]); // Tensor::<i32>::new(&[3], &[1]); // Tensor::<f32>::new_zeros(&[32, 255, 255, 64]); //Tensor::<i32>::new(&[3], &[1]); //
    let mut tnsr3 =  Tensor::<f32>::new(&[1., 1., 1., 1.], &[2, 2]); // Tensor::<f32>::new_zeros(&[32, 255, 255, 64]);
    let mut tnsr4 =  Tensor::<f32>::new(&[1., 1., 1., 1.], &[2, 2]); // Tensor::<f32>::new_zeros(&[32, 255, 255, 64]);
    let mut tnsr5 = Tensor::<f32>::new(&[PI / 2., PI / 6., PI / 3., PI / 4., 3. * PI / 2., 0., PI, 3. * PI / 4.], &[2, 2, 2]);

    let mut tnsr6 = Tensor::<f32>::new_ones(&[2, 1]);
    let mut tnsr7 = Tensor::<f32>::new_ones(&[2, 1]);

    let mut tnsr8 = Tensor::<f32>::new(&[1., 3., 6., 7.],  &[2, 2]);
    let mut tnsr9 = Tensor::<f32>::new(&[2., 5., 3., 1.],  &[2, 2]);

    let now = SystemTime::now();
    let mut result = tnsr8.mm(tnsr9.clone()).unwrap();
    result.flatten(0, true).unwrap();
    result.sum(0, false, false).unwrap().unwrap().print();

    tnsr8.sin().mm(tnsr9).unwrap().print();

    let elapsed = SystemTime::now().duration_since(now).unwrap();

    println!("{:.5?}", elapsed);
}
