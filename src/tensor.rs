use crate::utils::consts::_e;
use crate::utils::consts::TENSOR_THREADING;
use crate::utils::dtype::{Complex32, DType};
use crate::utils::ops::{Operation, TensorOps};
use crate::utils::{Print, ToRc, ToSlice};
use std::borrow::Borrow;
use std::borrow::BorrowMut;
use std::cell::{RefCell, RefMut};
use std::collections::HashMap;
use std::collections::LinkedList;
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;

// threading - T/F | auto - T/F | thread_count | usize
// TODO make it Tensor<T, R> so that I can make prev_op point to a diff type.
#[derive(Debug)]
pub struct Tensor<T: DType> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub grad: Option<Rc<RefCell<Tensor<f32>>>>,
    pub prev_op: Option<(
        Option<Operation>,
        (Rc<RefCell<Tensor<f32>>>, Rc<RefCell<Tensor<f32>>>),
    )>,
}

//TODO unsqueeze method
impl<T: DType> Tensor<T> {
    // init tensor with data and shape tensor
    pub fn new(data: &[T], shape: &[usize]) -> Self {
        let data = data.to_vec();
        let shape = shape.to_vec();

        assert_eq!(
            Tensor::<T>::num_elm(&shape),
            data.len(),
            "Tensor has mismatched shape and elements!"
        );

        let strides = Tensor::<T>::comp_strides(&shape);
        Tensor {
            data,
            shape,
            strides,
            grad: None,
            prev_op: None,
        }
    }

    // inits tensor w/ gradient
    pub fn new_grad(data: &[T], shape: &[usize]) -> Self {
        let data = data.to_vec();
        let shape = shape.to_vec();

        assert_eq!(
            Tensor::<T>::num_elm(&shape),
            data.len(),
            "Tensor has mismatched shape and elements!"
        );

        let strides = Tensor::<T>::comp_strides(&shape);
        let grad = Some(Rc::new(RefCell::new(Tensor::<f32>::new_zeros(
            shape.as_slice(),
        ))));
        Tensor {
            data,
            shape,
            strides,
            grad,
            prev_op: None,
        }
    }

    // static init for 0 tensor of a shape
    pub fn new_zeros(shape: &[usize]) -> Tensor<T> {
        let elm = Tensor::<T>::num_elm(shape);
        let data: Vec<T> = vec![T::zero(); elm];

        Tensor::<T>::new(data.as_slice(), shape)
    }

    // static init for 1 tensor of a shape
    pub fn new_ones(shape: &[usize]) -> Tensor<T> {
        let elm = Tensor::<T>::num_elm(shape);
        let data: Vec<T> = vec![T::one(); elm];

        Tensor::<T>::new(data.as_slice(), shape)
    }

    // returns cls tensor's rank
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    // returns the num elements in the tensor
    pub fn num_elms(&self) -> usize {
        self.data.len()
    }

    // TODO make Shape stuct
    // compares other shape with cls tensor's shape. optional dimension to skip during comparison
    fn cmp_shape(&self, other_shape: &[usize], ignore_dim: Option<usize>) -> bool {
        if other_shape.len() != self.rank() {
            return false;
        }

        for i in 0..self.rank() {
            if let Some(dim) = ignore_dim {
                if dim == i {
                    continue;
                }
            }
            if self.shape[i] != other_shape[i] {
                return false;
            }
        }

        true
    }

    // returns true if tensor has a gradient
    pub fn requires_grad(&self) -> bool {
        if let Some(_) = &self.grad {
            return true;
        }
        false
    }

    // static method to compute the strides based off tensor shape
    fn comp_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides: Vec<usize> = vec![];
        let mut tot: usize = 1;
        let mut shape_vec = shape.to_vec();

        shape_vec.reverse();

        for i in shape_vec {
            strides.push(tot);
            tot *= i;
        }

        strides.reverse();
        strides
    }

    // returns the number of elements in the tensor
    pub fn num_elm(shape: &[usize]) -> usize {
        let mut total: usize = 1;

        for i in shape {
            total *= i
        }

        total
    }

    // get tensor 1d data array index from tensor index
    fn get_arr_idx(&self, idx: &[usize]) -> Result<usize, TensorOutOfBoundsError> {
        let mut elm_idx: usize = 0;

        if !(idx.len() <= self.rank()) {
            return Err(TensorOutOfBoundsError::new(
                "Index out of bounds!".to_string(),
            ));
        }

        for (i, idx) in idx.iter().enumerate() {
            if !(*idx <= self.shape[i]) {
                return Err(TensorOutOfBoundsError::new(format!(
                    "Index of dimension {} is out of scope. Expected in {}.",
                    i, self.shape[i]
                )));
            }
            elm_idx += *idx * self.strides[i];
        }

        Ok(elm_idx)
    }

    // set the value at tensor 1d data array from tensor index
    pub fn set_elm(&mut self, value: T, idx: &[usize]) {
        let idx = self
            .get_arr_idx(idx)
            .expect("Tensor out of bounds for idx!");
        self.data[idx] = value;
    }

    // get the value at tensor 1d data array from tensor index
    pub fn get_elm(&self, idx: &[usize]) -> &T {
        let idx = self
            .get_arr_idx(idx)
            .expect("Tensor out of bounds for idx!");
        &self.data[idx]
    }

    //// set and get Slice ////
    //////////////////////////////////////////
    pub fn set_slice(
        &mut self,
        idxs: Slice,
        other: Tensor<T>,
    ) -> Result<(), TensorMismatchedShapeError> {
        if !other.cmp_shape(&idxs.shape, None) {
            return Err(TensorMismatchedShapeError);
        }

        let (old_idxs, new_idxs) = idxs
            .get_idxs_from_slice(&self.shape)
            .expect("Could not get slice indexes!");

        for (i, idx) in old_idxs.iter().enumerate() {
            let val = other.get_elm(new_idxs[i].as_slice());
            self.set_elm(val.clone(), idx.as_slice());
        }

        Ok(())
    }

    pub fn get_slice(&self, idxs: Slice) -> Result<Tensor<T>, TensorOutOfBoundsError> {
        let (old_idxs, new_idxs) = idxs
            .get_idxs_from_slice(&self.shape)
            .expect("Could not get slice indexes!");
        let new_shape = idxs.shape;

        let mut empty = Tensor::<T>::new_zeros(&new_shape);

        for (i, idx) in old_idxs.iter().enumerate() {
            let elm = self.get_elm(idx.as_slice()).clone();
            empty.set_elm(elm, new_idxs[i].as_slice())
        }

        Ok(empty)
    }
    //////////////////////////////////////////
    //////////////////////////////////////////

    // reshape tensor to new shape. optional deep copy or returns a new tensor
    pub fn reshape(&mut self, shape: &[usize], inplace: bool) -> Option<Tensor<T>> {
        let strides = Tensor::<T>::comp_strides(shape);

        if inplace {
            self.shape = shape.to_vec();
            self.strides = strides;
        } else {
            return Some(Tensor::new(self.data.clone().as_slice(), shape));
        }

        None
    }

    // 0 indexed concat - combines two tensors at a dim
    // lowkey forgot how i did this lol...
    pub fn concat(
        &mut self,
        other: &Tensor<T>,
        dim: usize,
    ) -> Result<Tensor<T>, TensorMismatchedShapeError> {
        if !self.cmp_shape(&other.shape, Some(dim)) {
            return Err(TensorMismatchedShapeError);
        }

        let mut idxs_1 = Vec::<Vec<(usize, usize)>>::new();
        let mut idxs_2 = Vec::<Vec<(usize, usize)>>::new();
        idxs_1.push(vec![]);
        idxs_2.push(vec![]);

        let mut new_shape = Vec::<usize>::new();

        for d in 0..self.rank() {
            if d == dim {
                new_shape.push(self.shape[d] + other.shape[d]);

                for i in 0..idxs_1.len() {
                    idxs_1[i].push((0, self.shape[d]));
                    idxs_2[i].push((0, other.shape[d]));
                }
            } else if d < dim {
                new_shape.push(self.shape[d]);
                let mut replace1 = Vec::<Vec<(usize, usize)>>::new();
                let mut replace2 = Vec::<Vec<(usize, usize)>>::new();
                for j in 0..idxs_1.len() {
                    for i in 0..self.shape[d] {
                        let mut v1 = idxs_1[j].clone();

                        v1.push((i, i + 1));
                        replace1.push(v1);

                        let mut v2 = idxs_2[j].clone();
                        v2.push((i, i + 1));
                        replace2.push(v2);
                    }
                }

                idxs_1 = replace1;
                idxs_2 = replace2;
            } else {
                new_shape.push(self.shape[d]);

                for i in 0..idxs_1.len() {
                    idxs_1[i].push((0, self.shape[d]));
                    idxs_2[i].push((0, self.shape[d]));
                }
            }
        }

        let mut new_data: Vec<T> = vec![];
        for i in 0..idxs_1.len() {
            let t1 = self
                .get_slice(Slice::new(&idxs_1[i]))
                .expect("Out of bounds error!");
            let t2 = other
                .get_slice(Slice::new(&idxs_2[i]))
                .expect("Out of bounds error!");

            new_data.extend(t1.data);
            new_data.extend(t2.data);
        }

        Ok(Tensor::new(&new_data.as_slice(), new_shape.as_slice()))
    }

    // transposes tensor over any two dims
    pub fn transpose(
        &mut self,
        dim: (usize, usize),
        inplace: bool,
    ) -> Result<Option<Tensor<T>>, TensorOutOfBoundsError> {
        if !(dim.0 < self.rank() && dim.1 < self.rank()) {
            return Err(TensorOutOfBoundsError::new(
                "TensorOutOfBoundsError! Indexing out of Tensor!".to_string(),
            ));
        }

        let mut new_shape = self.shape.clone();
        let cl = new_shape[dim.0];
        new_shape[dim.0] = new_shape[dim.1];
        new_shape[dim.1] = cl;

        let mut empty = Tensor::<T>::new_zeros(&new_shape);
        let mut set_slice: Vec<(usize, usize)> = vec![];

        for (i, s) in new_shape.iter().enumerate() {
            if i == dim.0 || i == dim.1 {
                set_slice.push((0, 0))
            } else {
                set_slice.push((0, *s))
            }
        }

        let mut get_slice = set_slice.clone();

        for j in 0..self.shape[dim.1] {
            for i in 0..self.shape[dim.0] {
                set_slice[dim.0] = (j, j + 1);
                set_slice[dim.1] = (i, i + 1);

                get_slice[dim.0] = (i, i + 1);
                get_slice[dim.1] = (j, j + 1);

                let tnsr_slice = self.get_slice(Slice::new(get_slice.as_slice())).unwrap();
                empty
                    .set_slice(Slice::new(set_slice.as_slice()), tnsr_slice)
                    .expect("Could not transpose!");
            }
        }

        if inplace {
            self.data = empty.data;
            self.shape = new_shape.clone();
            self.strides = Tensor::<T>::comp_strides(&new_shape);
        } else {
            return Ok(Some(empty));
        }

        Ok(None)
    }

    // 0 idx - performs a series of transposes along the tensor's axis
    pub fn permute(&mut self, dim: &[usize]) -> Result<(), TensorMismatchedShapeError> {
        if dim.len() != self.rank() {
            return Err(TensorMismatchedShapeError);
        }

        let r: usize = self.rank().clone();
        let mut visited = vec![0; r];
        let mut new_shape: Vec<usize> = Vec::<usize>::new();
        for i in dim {
            if *i >= self.rank() {
                return Err(TensorMismatchedShapeError);
            }

            if visited[*i] == 1 {
                return Err(TensorMismatchedShapeError);
            }

            new_shape.push(*i);
            visited[*i] = 1;
        }

        let dim: Vec<usize> = dim.to_vec();
        let mut map: HashMap<usize, usize> = HashMap::new();

        for (i, e) in dim.iter().enumerate() {
            if i == *e {
                continue;
            }

            let t_0 = *map.get(&i).unwrap_or(&i);
            let t_1 = *map.get(e).unwrap_or(e);

            // idxs have transposed each other
            if t_1 == i && t_0 == *e {
                continue;
            }

            self.transpose((t_0, t_1), true)
                .expect(format!("Could not transpose tensor at dim {} and {}!", e, i).as_str());

            map.insert(t_0, t_1);
            map.insert(t_1, t_0);
        }

        Ok(())
    }

    // matches tensors across their axes
    pub fn match_dims(
        self,
        other: Tensor<T>,
        ignore_after_dim: Option<usize>,
    ) -> Option<(Self, Tensor<T>, Vec<usize>, Vec<usize>, bool)> {
        let mut first_out = true;
        let (larger, mut smaller) = if self.rank() >= other.rank() {
            (self, other)
        } else {
            first_out = false;
            (other, self)
        };

        let mut matched_axes: Vec<usize> = vec![0; larger.rank()];
        if larger.rank() > smaller.rank() {
            for i in 0..larger.rank() {
                if i >= smaller.rank() {
                    matched_axes[larger.rank() - i - 1] = 1;
                } else {
                    matched_axes[larger.rank() - i - 1] = smaller.shape[smaller.rank() - i - 1];
                }
            }
            smaller.reshape(matched_axes.as_slice(), true);
        }

        let mut repeats_l = Vec::<usize>::new();
        let mut repeats_s = Vec::<usize>::new();

        let mut ignore_dim = 0;
        let mut ignore: bool = false;
        if let Some(dim) = ignore_after_dim {
            ignore_dim = larger.rank() - dim - 1;
            ignore = true;
        }

        for (idx, s) in smaller.shape.iter().rev().enumerate() {
            // don't use rev
            let e_idx = larger.shape.len() - idx - 1;
            if *s == larger.shape[e_idx] || (ignore_dim >= idx && ignore) {
                repeats_l.push(1);
                repeats_s.push(1);
            } else if *s == 1 {
                repeats_l.push(1);
                repeats_s.push(larger.shape[e_idx]);
            } else if larger.shape[e_idx] == 1 {
                repeats_l.push(*s);
                repeats_s.push(1);
            } else {
                return None; // why did i add this line lol? i don't wanna mess w/ it
            }
        }

        let same: bool = if repeats_s.iter().sum::<usize>() == smaller.data.len()
            && repeats_l.iter().sum::<usize>() == larger.data.len()
        {
            true
        } else {
            false
        };

        let repeats_l: Vec<usize> = repeats_l.into_iter().rev().collect();
        let repeats_s: Vec<usize> = repeats_s.into_iter().rev().collect();

        if first_out {
            Some((larger, smaller, repeats_l, repeats_s, same))
        } else {
            Some((smaller, larger, repeats_s, repeats_l, same))
        }
    }

    pub fn repeat_dim(&mut self, repeat: &[usize]) -> Result<(), TensorMismatchedShapeError> {
        if self.rank() != repeat.len() {
            return Err(TensorMismatchedShapeError);
        }

        let mut prev_data = Vec::<Vec<T>>::new();
        let mut repeated_data = self.data.clone();

        let mut new_shape = Vec::<usize>::new();
        let mut global_stride = 1;
        for (idx, i) in repeat.to_vec().iter().rev().enumerate() {
            if *i == 0 {
                return Err(TensorMismatchedShapeError);
            }

            if *i != 1 {
                let dim_shape = global_stride
                    * self.shape[self.rank() - idx - 1]
                    * self.strides[self.rank() - idx - 1];
                let mut group = Vec::<T>::new();

                for (d_idx, e) in repeated_data.iter().enumerate() {
                    if d_idx % dim_shape == 0 && d_idx != 0 {
                        prev_data.push(group);
                        group = Vec::<T>::new();
                    }

                    group.push(e.clone());
                }

                prev_data.push(group);

                for pd in &mut prev_data {
                    let base = pd.clone();
                    for _ in 0..(*i - 1) {
                        pd.extend(base.clone());
                    }
                }
            }

            global_stride *= i;
            new_shape.push(self.shape[self.rank() - idx - 1] * *i);
            if *i == 1 {
                continue;
            }

            let mut collapsed = Vec::<T>::new();
            for pd in prev_data {
                for e in pd {
                    collapsed.push(e);
                }
            }

            repeated_data = collapsed;
            prev_data = vec![];
        }

        let new_shape: Vec<usize> = new_shape.into_iter().rev().collect();

        self.data = repeated_data;
        self.shape = new_shape;
        self.strides = Tensor::<T>::comp_strides(&self.shape);

        Ok(())
    }

    pub fn sum(
        &mut self,
        dim: usize,
        keepdim: bool,
        inplace: bool,
    ) -> Result<Option<Tensor<T>>, TensorMismatchedShapeError> {
        if dim >= self.rank() {
            return Err(TensorMismatchedShapeError);
        }

        let mut base_idx: Vec<(usize, usize)> = vec![(0, 0); self.rank()];
        let mut out_shape: Vec<usize> = vec![];
        for i in 0..self.rank() {
            if i == dim {
                base_idx[i] = (0, 0);
                if dim == 0 && self.rank() == 1 {
                    out_shape.push(1);
                } else if keepdim {
                    out_shape.push(1);
                }
            } else {
                base_idx[i] = (0, self.shape[i]);
                out_shape.push(self.shape[i]);
            }
        }

        let mut aggregator = Tensor::<T>::new_zeros(out_shape.as_slice());

        for i in 0..self.shape[dim] {
            base_idx[dim] = (i, i + 1);
            let mut collapsed = self
                .get_slice(Slice::new(base_idx.as_slice()))
                .expect("Could not slice tensor at given position!");
            collapsed.reshape(out_shape.as_slice(), true);
            aggregator = aggregator + collapsed;
        }

        if inplace {
            self.data = aggregator.data;
            self.shape = out_shape;
            self.strides = Tensor::<T>::comp_strides(self.shape.as_slice());
        } else {
            return Ok(Some(Tensor::<T>::new(
                aggregator.data.as_slice(),
                aggregator.shape.as_slice(),
            )));
        }

        Ok(None)
    }

    pub fn flatten(
        &mut self,
        dim: usize,
        inplace: bool,
    ) -> Result<Option<Tensor<T>>, TensorMismatchedShapeError> {
        if dim >= self.rank() {
            return Err(TensorMismatchedShapeError);
        }

        let mut out_dim: Vec<usize> = vec![];
        let mut agg: usize = 1;

        for (idx, i) in self.shape.iter().enumerate() {
            if idx < dim {
                out_dim.push(*i);
            } else {
                agg *= i;
            }
        }

        out_dim.push(agg);
        if inplace {
            return Ok(self.reshape(out_dim.as_slice(), true));
        }

        Ok(self.reshape(out_dim.as_slice(), false))
    }

    pub fn cast_fp32(self) -> Tensor<f32> {
        let data: Vec<f32> = self.data.iter().map(|x| x.to_fp32()).collect();
        let shape = self.shape;
        let r_g = self.grad;

        let mut r = Tensor::<f32>::new(data.as_slice(), shape.as_slice());
        r.grad = r_g;

        if let Some(po) = &self.prev_op {
            let prev = po.1.clone();
            let op = if let Some(_op) = &po.0 {
                Some(_op.clone())
            } else {
                None
            };
            r.prev_op = Some((op, prev));
        } else {
            r.prev_op = None;
        }
        r
    }

    // element ops
    fn matmul(
        &mut self,
        mut other: Tensor<T>,
        transpose_inner: bool,
        transpose_outer: bool,
    ) -> Result<Tensor<T>, TensorMismatchedShapeError> {
        let mut this = (&mut *self).clone();
        let save_other = other.clone();

        if this.rank() < 2 || other.rank() < 2 {
            return Err(TensorMismatchedShapeError);
        }

        if transpose_inner {
            this.transpose((this.rank() - 2, this.rank() - 1), true)
                .expect("Could not transpose!");
        }

        if transpose_outer {
            other
                .transpose((other.rank() - 2, other.rank() - 1), true)
                .expect("Could not transpose!");
        }

        let (a, mut other, _shape) = TensorOps::new(TENSOR_THREADING).match_tnsrs(
            this,
            other.clone(),
            Some(self.rank() - 2),
        );

        if a.shape[a.rank() - 1] != other.shape[other.rank() - 2] {
            return Err(TensorMismatchedShapeError);
        }

        let mut base_idx_1: Vec<(usize, usize)> = vec![];
        let mut base_idx_2: Vec<(usize, usize)> = vec![];
        let mut base_idx_3: Vec<(usize, usize)> = vec![];

        let mut out_shape: Vec<usize> = vec![];

        for i in 0..a.rank() - 2 {
            base_idx_1.push((0, a.shape[i]));
            base_idx_2.push((0, other.shape[i]));
            base_idx_3.push((0, a.shape[i]));
            out_shape.push(a.shape[i]);
        }

        base_idx_1.push((0, 0));
        base_idx_1.push((0, a.shape[a.rank() - 1]));

        base_idx_2.push((0, 0));
        base_idx_2.push((0, other.shape[other.rank() - 2]));

        base_idx_3.push((0, 0));
        base_idx_3.push((0, 0));

        out_shape.push(a.shape[a.rank() - 2]);
        out_shape.push(other.shape[other.rank() - 1]);

        let mut aggregator = Tensor::new_zeros(out_shape.as_slice());

        other
            .transpose((other.rank() - 2, other.rank() - 1), true)
            .expect("Could not transpose!");
        for i in 0..a.shape[a.rank() - 2] {
            base_idx_1[a.rank() - 2] = (i, i + 1);
            base_idx_3[a.rank() - 2] = (i, i + 1);

            let chunk_1 = a
                .get_slice(Slice::new(base_idx_1.as_slice()))
                .expect("Could not slice tensor!");
            for j in 0..other.shape[a.rank() - 2] {
                base_idx_2[other.rank() - 2] = (j, j + 1);
                base_idx_3[other.rank() - 1] = (j, j + 1);

                let chunk_2 = other
                    .get_slice(base_idx_2.to_slice())
                    .expect("Could not slice tensor!");
                let mut result = chunk_1.clone().mul(&chunk_2);
                result
                    .sum(result.rank() - 1, true, true)
                    .expect("Could not sum over dim!");

                aggregator
                    .set_slice(base_idx_3.to_slice(), result)
                    .expect("Could not set slice!");
            }
        }

        if self.requires_grad() || save_other.requires_grad() {
            aggregator.grad = Some(Tensor::new_zeros(aggregator.shape.as_slice()).to_rc());
            aggregator.prev_op = Some((
                Some(Operation::MatMul),
                (
                    self.clone().cast_fp32().to_rc(),
                    save_other.clone().cast_fp32().to_rc(),
                ),
            ));
            // I really don't like having to clone these tensors
        }
        Ok(aggregator)
    }

    // AB
    pub fn mm(&mut self, other: Tensor<T>) -> Result<Tensor<T>, TensorMismatchedShapeError> {
        self.matmul(other, false, false)
    }

    // (A^T)B
    pub fn dot(&mut self, other: Tensor<T>) -> Result<Tensor<T>, TensorMismatchedShapeError> {
        let mut result = self.matmul(other, true, false).unwrap();
        result
            .flatten(result.rank() - 2, true)
            .expect("Could not flatten for dot prod");
        Ok(result)
    }

    // A(B^T)
    pub fn outer(&mut self, other: Tensor<T>) -> Result<Tensor<T>, TensorMismatchedShapeError> {
        self.matmul(other, false, true)
    }

    fn mm_bckwd(
        &self,
        x: &mut RefMut<Tensor<f32>>,
        y: &mut RefMut<Tensor<f32>>,
        lhs: bool,
        grad: &Tensor<f32>,
    ) {
        if x.requires_grad() {
            let x_g = x.grad.clone().unwrap().clone();
            let mut x_g = x_g.as_ref().borrow_mut();
            let mut grad = grad.clone();

            if lhs {
                let y_t = y
                    .clone()
                    .transpose((y.rank() - 2, y.rank() - 1), false)
                    .expect("could not transpose!")
                    .unwrap();
                grad = grad.mm(y_t).unwrap();
            } else {
                let mut y_t = y
                    .clone()
                    .transpose((y.rank() - 2, y.rank() - 1), false)
                    .expect("could not transpose!")
                    .unwrap();
                grad = y_t.mm(grad).unwrap();
            }

            x_g.data = x_g
                .clone()
                .add(x.backprop(grad).expect("could not compute gradient!"))
                .data;
        }
    }

    pub fn add(&self, other: &Tensor<T>) -> Tensor<T> {
        let mut r = TensorOps::new(TENSOR_THREADING).add(self.clone(), other.clone());
        if self.requires_grad() || other.requires_grad() {
            r.grad = Some(Tensor::new_zeros(r.shape.as_slice()).to_rc());
            r.prev_op = Some((
                Some(Operation::Add),
                (
                    self.clone().cast_fp32().to_rc(),
                    other.clone().cast_fp32().to_rc(),
                ),
            ));
            // I really don't like having to clone these tensors
        }
        r
    }

    fn add_bckwd(&self, x: &mut RefMut<Tensor<f32>>, grad: &Tensor<f32>) {
        if x.requires_grad() {
            let x_g = x.grad.clone().unwrap().clone();
            let mut x_g = x_g.as_ref().borrow_mut();
            x_g.data = x_g
                .clone()
                .add(
                    x.backprop(grad.clone())
                        .expect("could not compute gradient!"),
                )
                .data;
        }
    }

    pub fn sub(&self, other: &Tensor<T>) -> Tensor<T> {
        let mut r = TensorOps::new(TENSOR_THREADING).sub(self.clone(), other.clone());
        if self.requires_grad() || other.requires_grad() {
            r.grad = Some(Tensor::new_zeros(r.shape.as_slice()).to_rc());
            r.prev_op = Some((
                Some(Operation::Sub),
                (
                    self.clone().cast_fp32().to_rc(),
                    other.clone().cast_fp32().to_rc(),
                ),
            ));
            // I really don't like having to clone these tensors
        }
        r
    }

    pub fn mul(&self, other: &Tensor<T>) -> Tensor<T> {
        let mut r = TensorOps::new(TENSOR_THREADING).mul(self.clone(), other.clone());
        if self.requires_grad() || other.requires_grad() {
            r.grad = Some(Tensor::new_zeros(r.shape.as_slice()).to_rc());
            r.prev_op = Some((
                Some(Operation::Mul),
                (
                    self.clone().cast_fp32().to_rc(),
                    other.clone().cast_fp32().to_rc(),
                ),
            ));
            // I really don't like having to clone these tensors
        }
        r
    }

    fn mul_bckwd(
        &self,
        x: &mut RefMut<Tensor<f32>>,
        y: &mut RefMut<Tensor<f32>>,
        grad: &Tensor<f32>,
    ) {
        if x.requires_grad() {
            let x_g = x.grad.clone().unwrap().clone();
            let mut x_g = x_g.as_ref().borrow_mut();
            x_g.data = x_g
                .clone()
                .add(
                    x.backprop(grad.clone().mul(&y))
                        .expect("could not compute gradient!"),
                )
                .data;
        }
    }

    pub fn div(&self, other: &Tensor<T>) -> Tensor<f32> {
        let other_r = other.clone().pow((-1_f32).tnsr());
        let mut r = TensorOps::new(TENSOR_THREADING).mul(self.clone().cast_fp32(), other_r.clone());
        if self.requires_grad() || other.requires_grad() {
            r.grad = Some(Tensor::new_zeros(r.shape.as_slice()).to_rc());
            r.prev_op = Some((
                Some(Operation::Div),
                (
                    self.clone().cast_fp32().to_rc(),
                    other_r.cast_fp32().to_rc(),
                ),
            ));
            // I really don't like having to clone these tensors
        }
        r
    }

    pub fn sin(&self) -> Tensor<f32> {
        let mut r = TensorOps::new(TENSOR_THREADING)
            .sin(self.clone().cast_fp32(), Tensor::<f32>::new(&[1.], &[1]));
        if self.requires_grad() {
            r.grad = Some(Tensor::new_zeros(r.shape.as_slice()).to_rc());
            r.prev_op = Some((
                Some(Operation::Sin),
                (self.clone().cast_fp32().to_rc(), _e.tnsr().to_rc()),
            ));
            // I really don't like having to clone these tensors
        }
        r
    }

    fn sin_bckwd(&self, x: &mut RefMut<Tensor<f32>>, grad: &Tensor<f32>) {
        if x.requires_grad() {
            let x_g = x.grad.clone().unwrap().clone();
            let mut x_g = x_g.as_ref().borrow_mut();
            x_g.data = x_g
                .clone()
                .add(
                    x.backprop(grad.mul(&self.clone().cast_fp32().cos()))
                        .expect("could not compute gradient!"),
                )
                .data;
        }
    }

    pub fn cos(&self) -> Tensor<f32> {
        let mut r = TensorOps::new(TENSOR_THREADING)
            .cos(self.clone().cast_fp32(), Tensor::<f32>::new(&[1.], &[1]));
        if self.requires_grad() {
            r.grad = Some(Tensor::new_zeros(r.shape.as_slice()).to_rc());
            r.prev_op = Some((
                Some(Operation::Cos),
                (self.clone().cast_fp32().to_rc(), _e.tnsr().to_rc()),
            ));
            // I really don't like having to clone these tensors
        }
        r
    }

    fn cos_bckwd(&self, x: &mut RefMut<Tensor<f32>>, grad: &Tensor<f32>) {
        if x.requires_grad() {
            let x_g = x.grad.clone().unwrap().clone();
            let mut x_g = x_g.as_ref().borrow_mut();
            x_g.data = x_g
                .clone()
                .add(
                    x.backprop(grad.mul(&self.clone().cast_fp32().sin().mul(&(-1_f32).tnsr())))
                        .expect("could not compute gradient!"),
                )
                .data;
        }
    }

    pub fn tan(&self) -> Tensor<f32> {
        let mut r = TensorOps::new(TENSOR_THREADING)
            .tan(self.clone().cast_fp32(), Tensor::<f32>::new(&[1.], &[1]));
        if self.requires_grad() {
            r.grad = Some(Tensor::new_zeros(r.shape.as_slice()).to_rc());
            r.prev_op = Some((
                Some(Operation::Tan),
                (self.clone().cast_fp32().to_rc(), _e.tnsr().to_rc()),
            ));
            // I really don't like having to clone these tensors
        }
        r
    }

    fn tan_bckwd(&self, x: &mut RefMut<Tensor<f32>>, grad: &Tensor<f32>) {
        if x.requires_grad() {
            let x_g = x.grad.clone().unwrap().clone();
            let mut x_g = x_g.as_ref().borrow_mut();
            let sec_2 = 1_f32.tnsr().div(&self.clone().cos().pow(2_f32.tnsr()));
            x_g.data = x_g
                .clone()
                .add(
                    x.backprop(grad.mul(&sec_2))
                        .expect("could not compute gradient!"),
                )
                .data;
        }
    }

    pub fn exp(&self) -> Tensor<f32> {
        let mut r = TensorOps::new(TENSOR_THREADING).exp(self.clone().cast_fp32());
        if self.requires_grad() {
            r.grad = Some(Tensor::new_zeros(r.shape.as_slice()).to_rc());
            r.prev_op = Some((
                Some(Operation::Exp),
                (self.clone().cast_fp32().to_rc(), _e.tnsr().to_rc()),
            ));
            // I really don't like having to clone these tensors
        }
        r
    }

    fn exp_bckwd(&self, x: &mut RefMut<Tensor<f32>>, grad: &Tensor<f32>) {
        if x.requires_grad() {
            let x_g = x.grad.clone().unwrap().clone();
            let mut x_g = x_g.as_ref().borrow_mut();
            x_g.data = x_g
                .clone()
                .add(
                    x.backprop(grad.mul(&self.clone().cast_fp32()))
                        .expect("could not compute gradient!"),
                )
                .data;
        }
    }

    pub fn pow(&self, expo: Tensor<f32>) -> Tensor<f32> {
        // let b = base.tnsr();
        // let base = base.tnsr();
        let (a, b, _shape) = TensorOps::new(TENSOR_THREADING).match_tnsrs(
            self.clone().cast_fp32(),
            expo.clone(),
            None,
        );

        let mut r = TensorOps::new(TENSOR_THREADING).pow(b, a);
        if self.requires_grad() {
            r.grad = Some(Tensor::new_zeros(r.shape.as_slice()).to_rc());
            r.prev_op = Some((
                Some(Operation::Pow),
                (self.clone().cast_fp32().to_rc(), expo.to_rc()),
            ));
            // I really don't like having to clone these tensors
        }

        r
    }

    fn pow_bckwd(&self, x: &mut RefMut<Tensor<f32>>, expo: f32, grad: &Tensor<f32>) {
        if x.requires_grad() {
            let x_g = x.grad.clone().unwrap().clone();
            let mut x_g = x_g.as_ref().borrow_mut();
            x_g.data = x_g
                .clone()
                .add(
                    x.backprop(grad.mul(&x.pow((expo - 1.).tnsr()).mul(&expo.tnsr())))
                        .expect("could not compute gradient!"),
                )
                .data;
        }
    }

    pub fn asin(self) -> Tensor<f32> {
        let mut r = TensorOps::new(TENSOR_THREADING)
            .asin(self.clone().cast_fp32(), Tensor::<f32>::new(&[1.], &[1]));
        if self.requires_grad() {
            r.grad = Some(Tensor::new_zeros(r.shape.as_slice()).to_rc());
            r.prev_op = Some((
                Some(Operation::ASin),
                (self.cast_fp32().to_rc(), 0_f32.tnsr().to_rc()),
            ));
            // I really don't like having to clone these tensors
        }

        r
    }

    fn asin_bckwd(&self, x: &mut RefMut<Tensor<f32>>, grad: &Tensor<f32>) {
        if x.requires_grad() {
            let x_g = x.grad.clone().unwrap().clone();
            let mut x_g = x_g.as_ref().borrow_mut();
            let x_sqrt = 1_f32.tnsr().sub(&x.pow(2_f32.tnsr()));
            let x_rsqrt = 1_f32.tnsr().div(&x_sqrt);
            x_g.data = x_g
                .clone()
                .add(
                    x.backprop(grad.mul(&x_rsqrt))
                        .expect("could not compute gradient!"),
                )
                .data;
        }
    }

    pub fn acos(&self) -> Tensor<f32> {
        let mut r = TensorOps::new(TENSOR_THREADING)
            .acos(self.clone().cast_fp32(), Tensor::<f32>::new(&[1.], &[1]));
        if self.requires_grad() {
            r.grad = Some(Tensor::new_zeros(r.shape.as_slice()).to_rc());
            r.prev_op = Some((
                Some(Operation::ACos),
                (self.clone().cast_fp32().to_rc(), 0_f32.tnsr().to_rc()),
            ));
            // I really don't like having to clone these tensors
        }

        r
    }

    fn acos_bckwd(&self, x: &mut RefMut<Tensor<f32>>, grad: &Tensor<f32>) {
        if x.requires_grad() {
            let x_g = x.grad.clone().unwrap().clone();
            let mut x_g = x_g.as_ref().borrow_mut();
            let x_sqrt = 1_f32.tnsr().sub(&x.pow(2_f32.tnsr())).pow((-2_f32).tnsr());
            let x_rsqrt = (-1_f32).tnsr().div(&x_sqrt);
            x_g.data = x_g
                .clone()
                .add(
                    x.backprop(grad.mul(&x_rsqrt))
                        .expect("could not compute gradient!"),
                )
                .data;
        }
    }

    pub fn atan(self) -> Tensor<f32> {
        let mut r = TensorOps::new(TENSOR_THREADING)
            .atan(self.clone().cast_fp32(), Tensor::<f32>::new(&[1.], &[1]));
        if self.requires_grad() {
            r.grad = Some(Tensor::new_zeros(r.shape.as_slice()).to_rc());
            r.prev_op = Some((
                Some(Operation::ATan),
                (self.clone().cast_fp32().to_rc(), 0_f32.tnsr().to_rc()),
            ));
            // I really don't like having to clone these tensors
        }

        r
    }

    fn atan_bckwd(&self, x: &mut RefMut<Tensor<f32>>, grad: &Tensor<f32>) {
        if x.requires_grad() {
            let x_g = x.grad.clone().unwrap().clone();
            let mut x_g = x_g.as_ref().borrow_mut();
            let x_sqrt = 1_f32.tnsr().add(x.pow(2_f32.tnsr()));
            let x_rsqrt = (1_f32).tnsr().div(&x_sqrt);
            x_g.data = x_g
                .clone()
                .add(
                    x.backprop(grad.mul(&x_rsqrt))
                        .expect("could not compute gradient!"),
                )
                .data;
        }
    }

    pub fn greater(&self, other: &Tensor<T>) -> Tensor<T> {
        let r = TensorOps::new(TENSOR_THREADING).cmp_greater(self.clone(), other.clone());
        r
    }

    pub fn less(&self, other: &Tensor<T>) -> Tensor<T> {
        let r = TensorOps::new(TENSOR_THREADING).cmp_greater(other.clone(), self.clone());
        r
    }

    pub fn equals(&self, other: &Tensor<T>) -> Tensor<T> {
        let r = TensorOps::new(TENSOR_THREADING).cmp_equals(self.clone(), other.clone());
        r
    }

    pub fn as_cmplx(self) -> Result<Tensor<Complex32>, TensorMismatchedShapeError> {
        if self.shape[self.rank() - 1] != 2 {
            return Err(TensorMismatchedShapeError);
        };

        let mut o_shape = Vec::<usize>::new();

        for i in 0..self.rank() - 1 {
            o_shape.push(self.shape[i]);
        }

        o_shape.push(1);
        // let agg = Tensor::<Complex32>::new_zeros(o_shape.as_slice());

        let mut slice: Vec<(usize, usize)> = Vec::new();

        for i in 0..self.rank() - 1 {
            slice.push((0, self.shape[i]));
        }

        let mut real_slice = slice.clone();
        let mut img_slice = slice.clone();
        real_slice.push((0, 1));
        img_slice.push((1, 2));

        let real_slice = Slice::new(real_slice.as_slice());
        let img_slice = Slice::new(img_slice.as_slice());

        let real_data = self.get_slice(real_slice).unwrap().data;
        let img_data = self.get_slice(img_slice).unwrap().data;

        let mut agg: Vec<Complex32> = vec![];

        for i in 0..real_data.len() {
            agg.push(Complex32::new(
                real_data[i].to_fp32(),
                img_data[i].to_fp32(),
            ));
        }

        let result = Tensor::<Complex32>::new(agg.as_slice(), o_shape.as_slice());
        Ok(result)
    }

    fn backprop(&self, grad: Tensor<f32>) -> Result<Tensor<f32>, TensorNoGradError> {
        if !self.requires_grad() {
            return Err(TensorNoGradError);
        }

        let prev_op = self.prev_op.clone();

        if let Some(po) = prev_op {
            let operation = po.0;
            let tnsrs = po.1;

            if let Some(op) = operation {
                match op {
                    Operation::Add => {
                        let x = tnsrs.0.clone();
                        let y = tnsrs.1.clone();
                        let mut x = x.as_ref().borrow_mut();
                        let mut y = y.as_ref().borrow_mut();

                        self.add_bckwd(&mut x, &grad);
                        self.add_bckwd(&mut y, &grad);
                    }

                    Operation::Sub => {
                        let x = tnsrs.0.clone();
                        let y = tnsrs.1.clone();
                        let mut x = x.as_ref().borrow_mut();
                        let y = y.as_ref().borrow_mut().mul(&(-1_f32).tnsr());

                        let y = RefCell::new(y);

                        self.add_bckwd(&mut x, &grad);
                        self.add_bckwd(&mut y.borrow_mut(), &grad);
                    }

                    Operation::Mul | Operation::Div => {
                        let x = tnsrs.0.clone();
                        let y = tnsrs.1.clone();
                        let mut x = x.as_ref().borrow_mut();
                        let mut y = y.as_ref().borrow_mut();

                        self.mul_bckwd(&mut x, &mut y, &grad);
                        self.mul_bckwd(&mut y, &mut x, &grad);
                    }

                    Operation::Exp => {
                        let x = tnsrs.0.clone();
                        let mut x = x.as_ref().borrow_mut();

                        self.exp_bckwd(&mut x, &grad);
                    }

                    Operation::Pow => {
                        let x = tnsrs.0.clone();
                        let y = tnsrs.1.clone();
                        let mut x = x.as_ref().borrow_mut();
                        let y = y.as_ref().borrow_mut().data[0];

                        self.pow_bckwd(&mut x, y, &grad);
                    }

                    Operation::Sin => {
                        let x = tnsrs.0.clone();
                        let mut x = x.as_ref().borrow_mut();

                        self.sin_bckwd(&mut x, &grad);
                    }

                    Operation::Cos => {
                        let x = tnsrs.0.clone();
                        let mut x = x.as_ref().borrow_mut();

                        self.cos_bckwd(&mut x, &grad);
                    }

                    Operation::Tan => {
                        let x = tnsrs.0.clone();
                        let mut x = x.as_ref().borrow_mut();

                        self.tan_bckwd(&mut x, &grad);
                    }

                    Operation::ASin => {
                        let x = tnsrs.0.clone();
                        let mut x = x.as_ref().borrow_mut();

                        self.asin_bckwd(&mut x, &grad);
                    }

                    Operation::ACos => {
                        let x = tnsrs.0.clone();
                        let mut x = x.as_ref().borrow_mut();

                        self.acos_bckwd(&mut x, &grad);
                    }

                    Operation::ATan => {
                        let x = tnsrs.0.clone();
                        let mut x = x.as_ref().borrow_mut();

                        self.atan_bckwd(&mut x, &grad);
                    }

                    Operation::MatMul => {
                        let x = tnsrs.0.clone();
                        let y = tnsrs.1.clone();
                        let mut x = x.as_ref().borrow_mut();
                        let mut y = y.as_ref().borrow_mut();

                        self.mm_bckwd(&mut x, &mut y, true, &grad);
                        self.mm_bckwd(&mut y, &mut x, false, &grad);
                    }

                    _ => {}
                }
            }
        }

        Ok(Tensor::new_ones(self.shape.as_slice()).mul(&grad))
    }

    pub fn cmp_grad(&self) {
        self.backprop(Tensor::<f32>::new_ones(self.shape.as_slice()))
            .expect("panik!");
    }
}

impl<T: DType> Add for Tensor<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        TensorOps::new(TENSOR_THREADING).add(self, other)
    }
}

// impl<T: DType> Mul for Tensor<T> {
//     type Output = Self;
//
//     fn mul(self, other: Self) -> Self {
//         TensorOps::new(TENSOR_THREADING).mul(self, other)
//     }
// }

/*impl<T: DType> Div for Tensor<T> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        TensorOps::new(TENSOR_THREADING).div(self, other)
    }
}*/

/*impl<T: DType> Sub for Tensor<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        TensorOps::new(TENSOR_THREADING).sub(self, other)
    }
}*/

impl<T: DType> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        let mut r = Tensor::new(self.data.clone().as_slice(), self.shape.clone().as_slice());
        if self.requires_grad() {
            r.grad = self.grad.clone();
            if let Some(po) = &self.prev_op {
                let prev = po.1.clone();
                let op = if let Some(_op) = &po.0 {
                    Some(_op.clone())
                } else {
                    None
                };
                r.prev_op = Some((op, prev));
            } else {
                r.prev_op = None;
            }
        }
        r
    }
}

impl<T: DType> fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.data)
    }
}

impl<T: DType> Print for Tensor<T> {
    fn print(&self) {
        println!("{:?}", self);
    }
}

#[derive(Debug)]
pub struct TensorMismatchedShapeError;

impl fmt::Display for TensorMismatchedShapeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tensors have mismatched shapes!")
    }
}

#[derive(Debug)]
pub struct TensorNoGradError;

impl fmt::Display for TensorNoGradError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tensor doesn't have a gradient!")
    }
}

#[derive(Debug)]
pub struct TensorOutOfBoundsError {
    pub msg: String,
}

impl fmt::Display for TensorOutOfBoundsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            format!(
                "Indexing part of tensor that is out of bounds! {}",
                self.msg
            )
        )
    }
}

impl TensorOutOfBoundsError {
    pub fn new(msg: String) -> Self {
        TensorOutOfBoundsError { msg }
    }
}

pub struct Slice<'b> {
    pub slice: &'b [(usize, usize)],
    pub shape: Vec<usize>,
}

impl<'b> Slice<'b> {
    pub fn new(slice: &'b [(usize, usize)]) -> Self {
        let mut shape: Vec<usize> = vec![];
        for idx in slice {
            shape.push(idx.1 - idx.0);
        }

        Slice { slice, shape }
    }

    fn rank(&self) -> usize {
        return self.shape.len();
    }

    // returns two sets of tensor indexes from tensor slice.
    // first set is w.r.t the tensor's shape.
    // second set returns a zero ref'd slice.
    pub fn get_idxs_from_slice(
        &self,
        shape: &[usize],
    ) -> Result<(Vec<Vec<usize>>, Vec<Vec<usize>>), TensorOutOfBoundsError> {
        let mut old_idxs: Vec<Vec<usize>> = vec![];
        let mut new_idxs: Vec<Vec<usize>> = vec![];

        assert!(
            self.rank() <= shape.len(),
            "Indexing element out of bounds for Tensor."
        );

        // TODO? handle case for if when s_e.1 == s_e.0 - i don think it act. matter now
        for (i, s_e) in self.slice.iter().enumerate() {
            if !(s_e.1 <= shape[i] && s_e.1 >= s_e.0) {
                return Err(TensorOutOfBoundsError::new(format!(
                    "Index of dimension {} is not within range. Expected {} but got ({}, {}).",
                    i, shape[i], s_e.0, s_e.1
                )));
            }

            let mut replace = Vec::<Vec<usize>>::new();
            let mut new_replace = Vec::<Vec<usize>>::new();
            for j in s_e.0..s_e.1 {
                if i == 0 {
                    old_idxs.push(vec![j]);
                    new_idxs.push(vec![j - s_e.0]);
                } else {
                    for v in &mut old_idxs.clone() {
                        v.push(j);
                        replace.push(v.clone());
                    }

                    for v in &mut new_idxs.clone() {
                        v.push(j - s_e.0);
                        new_replace.push(v.clone());
                    }
                }
            }

            if i != 0 {
                old_idxs = replace;
                new_idxs = new_replace;
            }
        }
        Ok((old_idxs, new_idxs))
    }
}

impl fmt::Display for Slice<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.slice)
    }
}
