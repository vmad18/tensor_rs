# tensor_rs

create tensors of any dimension, perform operations between them, and dynamically compute gradients

# done
* tensors
* tenosr ops
* autograd
* cpu blocked parallelization
* matmul
* outer/inner prods
* elementary math functions
* complex numbers

# to-do
* softmax
* other useful functions
* other tensor initializations
* gemm
* other optimization
* some examples

# purpose
to learn rust + get better intuition behind tensors and auto differentiation behind
sota ml frameworks.

# example
```rust
fn main() {
  let mut x = Tensor::<f32>::new(&[1., 3., 6., 7.], &[2, 2]); // params - tensor data, tensor shape
  let mut y = Tensor::<f32>::new(&[2., 5., 3., 1.],  &[2, 2]);

  let mut result = x.mm(y).unwrap(); // matmul tensors!

  let mut w1 = Tensor::<i32>::new_ones(&[64, 128, 128, 64]);
  let mut w2 = Tensor::<i32>::new_ones(&[128, 1, 64]);

  let mut tnsr_sum = w1.clone() + w2.clone(); // tensor ops over matched dimensions!

  tnsr_sum.transpose((2, 3), true).except("could not transpose!"); // transpose and permute tensors!
  tnsr_sum.permute(&[0, 1, 3, 2]).except("could not permute!");

  w1.flatten(0, true).expect("flatten unsuccessful!"); // params - dimension, inplace | flatten tensors!
  w2.sum(0, false, false).expect("could not sum!").unwrap(); // params - dimension, inplace | sum tensors across any dimension!

  result.sin(); // apply trig functions across a tensor

  //...and a lot of other stuff :) 
}
```
