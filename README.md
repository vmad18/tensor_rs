# tensor_rs (w.i.p)

Create tensors of any shape and perform operations between them!

# done
* tensors
* tenosr ops
* cpu parallelization
* matmul

# to-do
* autograd
* gemm

# example
```rust
fn main() {
  let mut x = Tensor::<f32>::new(&[1., 3., 6., 7.], &[2, 2]); // params - tensor data, tensor shape
  let mut y = Tensor::<f32>::new(&[2., 5., 3., 1.],  &[2, 2]);

  let mut result = x.mm(y).unwrap(); // matmul tensors!

  let mut w = Tensor::<i32>::new_ones(&[64, 128, 128, 64]); 
  w.flatten(0, true).expect("flatten unsuccessful!"); // params - dimension, inplace | flatten tensor
  w.sum(0, false, false).expect("could not sum!").unwrap(); // params - dimension, inplace | sum across any dimension

  result.sin(); | apply trig functions across a tensor
}
```
