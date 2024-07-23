// // use tensor::tensor::{Print, Tensor};
// //
// // fn main() {
// //     // let mut tnsr1: Tensor<i32> =
// //     //     Tensor::<i32>::new(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], &[2, 2, 1, 3]);
// //     // let mut tnsr2: Tensor<i32> = tnsr1.clone(); // Tensor::<i32>::new(&[1, 2, 3, 4], &[2, 1, 2]);
// //     //
// //     // // let result = tnsr1.concat(&tnsr2, 1).expect("Could not concat!");
// //     // // result.print();
// //     // // if let Ok(result) = tnsr1.concat(&tnsr2, 1) {
// //     // //     result.print();
// //     // // }
// //     //
// //     // if let Ok(_r) = tnsr2.permute(&[0, 2, 1, 3]) {
// //     //     println!("Worked");
// //     // }
// //     //
// //     // tnsr2.print();
// //     // tnsr1
// //     //     .transpose((1, 2), false)
// //     //     .expect("no no")
// //     //     .unwrap()
// //     //     .print();
// //     //
// //     // if let Ok(_r) = tnsr1.permute(&[2, 1, 1]) {
// //     //     println!("Done");
// //     // } else {
// //     //     println!("ERROR");
// //     // }
// //
// //     let tnsr = Tensor::<f32>::new_zeros(&[100, 100, 100]); // Tensor::<i32>::new(&[1, 2, 3, 4, 5, 6, 7, 8], &[2, 2, 2]);
// //     let tnsr2 = Tensor::<f32>::new_zeros(&[100, 100, 100]); // Tensor::<i32>::new(&[3], &[1]);
// //
// //     use std::time::Instant;
// //     let now = Instant::now();
// //
// //     (tnsr + tnsr2);
// //
// //     let elapsed = now.elapsed();
// //
// //     println!("{:.5?}", elapsed);
// //
// //     // let (t1, mut t2, a, b, c) = tnsr
// //     //     .match_dims(tnsr2)
// //     //     .expect("Dimensions do not match at all!");
// //     //
// //     // a.print();
// //     // b.print();
// //     // t2.reshape(a.as_slice(), true);
// //     // if let Ok(_r) = t2.repeat_dim(b.as_slice()) {
// //     //     t2.print();
// //     // }
// // }
// use std::thread;
// use std::ptr;
// use std::ops::Add;
// use std::fmt::Debug;
//
// pub trait DType: Debug + Clone + Copy + Add<Output=Self> + Send + Sync + 'static {
//     fn zero() -> Self;
// }
//
// impl DType for u8 {
//     fn zero() -> Self {
//         0
//     }
// }
//
// impl DType for i8 {
//     fn zero() -> Self {
//         0
//     }
// }
//
// impl DType for i32 {
//     fn zero() -> Self {
//         0
//     }
// }
//
// impl DType for i64 {
//     fn zero() -> Self {
//         0
//     }
// }
//
// impl DType for usize {
//     fn zero() -> Self {
//         0
//     }
// }
//
// impl DType for f32 {
//     fn zero() -> Self {
//         0.
//     }
// }
//
// impl DType for f64 {
//     fn zero() -> Self {
//         0.
//     }
// }
//
//
// pub struct Tensor<'a, T> {
//     data: Vec<T>,
//     bro: &'a [T],
//     num_elms: usize,
// }
//
// impl<'a, T> Tensor<'a, T> {
//     pub fn new(data: Vec<T>) -> Self {
//         let num_elms = data.len();
//         let bro = data.as_slice();
//         Tensor {
//             data,
//             bro,
//             num_elms,
//         }
//     }
// }
// pub fn add<'a, T: DType>(
//     a: Tensor<'a, T>,
//     b: Tensor<'a, T>,
// ) -> Tensor<'a, T> {
//     let BLOCK_SIZE: usize = 128;
//
//     let (a_data, b_data) = (a.data, b.data);
//     let num_elms = a.num_elms;
//
//     let mut output = vec![T::zero(); num_elms];
//
//     let output_ptr = output.as_mut_ptr() as usize;
//
//     let step_size: usize = if num_elms < BLOCK_SIZE {
//         num_elms
//     } else {
//         BLOCK_SIZE
//     };
//
//     let mut handles = vec![];
//
//     for i in (0..num_elms).step_by(step_size) {
//         let a_chunk = &a_data[i..(i + step_size).min(num_elms)];
//         let b_chunk = &b_data[i..(i + step_size).min(num_elms)];
//         let output_chunk_ptr = output_ptr + i * std::mem::size_of::<T>();
//
//         let handle = thread::spawn(move || {
//             unsafe {
//                 let output_chunk_ptr = output_chunk_ptr as *mut T;
//                 for j in 0..a_chunk.len() {
//                     ptr::write(output_chunk_ptr.add(j), a_chunk[j] + b_chunk[j]);
//                 }
//             }
//         });
//
//         handles.push(handle);
//     }
//
//     for handle in handles {
//         handle.join().unwrap();
//     }
//
//     Tensor::new(output)
// }
//
// fn main() {
//     // Example usage:
//     let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//     let b = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
//
//     let tensor_a = Tensor::new(a);
//     let tensor_b = Tensor::new(b);
//
//     let result_tensor = add(tensor_a, tensor_b);
//
//     println!("{:?}", result_tensor.data); // Output should be [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
// }