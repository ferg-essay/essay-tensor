use core::fmt;
use std::{ops::{Deref, self}, cell::UnsafeCell, sync::{atomic::{AtomicU64, Ordering}, Mutex, Arc}};

use crate::{tensor::{Dtype, TensorId}, Tensor};

use super::Tape;

pub struct Var<D: Dtype=f32> {
    id: VarId,
    name: String,
    tensor: Arc<Mutex<TensorShare<D>>>,
    last_tensor: UnsafeCell<Tensor<D>>, // clone() to allow deref
}

impl<D: Dtype> Var<D> {
    pub fn new(name: &str, tensor: impl Into<Tensor<D>>) -> Self {
        let id = VarId::alloc();

        let tensor = Into::into(tensor); // .with_var(id);

        let last_tensor = tensor.clone();

        Self {
            id,
            tensor: Arc::new(Mutex::new(TensorShare(tensor))),
            name: name.to_string(),
            last_tensor: UnsafeCell::new(last_tensor),
        }
    }

    #[inline]
    pub fn id(&self) -> VarId {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn set(&mut self, tensor: impl Into<Tensor<D>>) {
        let tensor = Into::into(tensor); // .with_var(self.id);

        self.tensor.lock().unwrap().set(tensor);
    }

    pub(crate) fn tensor_with_id(&self, id: TensorId) -> Tensor<D> {
        self.tensor.lock().unwrap().tensor_with_id(id)
    }
}

impl Var<f32> {
    pub(crate) fn _assign_sub(&self, grad: Tensor) -> &Self {
        self.tensor.lock().unwrap()._assign_sub(grad);

        self
    }
}

impl Var {
    pub fn tensor(&self) -> Tensor {
        Tape::var(&self)
    }

    pub(crate) fn tensor_raw(&self) -> Tensor {
        self.tensor.lock().unwrap().get()
    }
}

impl Deref for Var {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        // Tape::set_var(&self.name, &self.tensor);
        let tensor = Tape::var(&self);

        unsafe {
            *self.last_tensor.get() = tensor;

            &self.last_tensor.get().as_ref().unwrap()
        }
    }
}

impl fmt::Debug for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Var")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("tensor", &self.tensor.lock().unwrap().get())
            .finish()
    }
}

impl Clone for Var {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            name: self.name.clone(),
            tensor: self.tensor.clone(),
            last_tensor: unsafe { 
                UnsafeCell::new(self.last_tensor.get().as_ref().unwrap().clone()) 
            },
        }
    }
}

impl From<Var> for Tensor {
    fn from(var: Var) -> Self {
        Tape::var(&var)
    }
}

impl From<&Var> for Tensor {
    fn from(var: &Var) -> Self {
        Tape::var(var)
    }
}

macro_rules! var_ops {
    ($ty:ty, $op:ident, $fun:ident) => {
        impl ops::$op<Tensor<$ty>> for &Var<$ty> {
            type Output = Tensor;
        
            fn $fun(self, rhs: Tensor) -> Self::Output {
                self.deref().$fun(&rhs)
            }
        }

        impl ops::$op<&Tensor<$ty>> for &Var<$ty> {
            type Output = Tensor<$ty>;
        
            fn $fun(self, rhs: &Tensor<$ty>) -> Self::Output {
                self.deref().$fun(rhs)
            }
        }

        impl ops::$op<&Var<$ty>> for Tensor<$ty> {
            type Output = Tensor<$ty>;
        
            fn $fun(self, rhs: &Var<$ty>) -> Self::Output {
                self.$fun(rhs.deref())
            }
        }

        impl ops::$op<&Var<$ty>> for &Tensor<$ty> {
            type Output = Tensor;
        
            fn $fun(self, rhs: &Var<$ty>) -> Self::Output {
                self.$fun(rhs.deref())
            }
        }

        impl ops::$op<&Var<$ty>> for &Var<$ty> {
            type Output = Tensor;
        
            fn $fun(self, rhs: &Var<$ty>) -> Self::Output {
                self.$fun(rhs.deref())
            }
        }

        impl ops::$op<&Var<$ty>> for $ty {
            type Output = Tensor;
        
            fn $fun(self, rhs: &Var<$ty>) -> Self::Output {
                Tensor::<$ty>::from(self).$fun(rhs.deref())
            }
        }

        impl ops::$op<$ty> for &Var<$ty> {
            type Output = Tensor;
        
            fn $fun(self, rhs: $ty) -> Self::Output {
                self.deref().$fun(Tensor::<$ty>::from(rhs))
            }
        }
    }
}

var_ops!(f32, Add, add);
var_ops!(f32, Sub, sub);
var_ops!(f32, Mul, mul);
var_ops!(f32, Div, div);
var_ops!(f32, Rem, rem);

static VAR_ID: AtomicU64 = AtomicU64::new(0);

///
/// VarId is globally unique to avoid name collisions.
///
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VarId(u64);

impl VarId {
    fn alloc() -> VarId {
        let id = VAR_ID.fetch_add(1, Ordering::SeqCst);

        VarId(id)
    }

    pub fn index(&self) -> u64 {
        self.0
    }
}

struct TensorShare<D>(Tensor<D>);

impl<D> TensorShare<D> {
    fn get(&self) -> Tensor<D> {
        self.0.clone()
    }

    fn tensor_with_id(&mut self, id: TensorId) -> Tensor<D> {
        self.0 = self.0.clone().with_id(id);

        assert!(self.0.id().is_some());

        self.0.clone()
    }

    fn set(&mut self, tensor: Tensor<D>) {
        self.0 = tensor;
    }
}

impl TensorShare<f32> {
    fn _assign_sub(&mut self, tensor: Tensor<f32>) {
        self.0 = &self.0 - &tensor;
    }
}

#[cfg(test)]
mod test {
    use crate::{prelude::*, model::Var};

    #[test]
    fn test_var() {
        let t1 = tensor!([1., 2., 3.]);
        let v1 = Var::new("t1", t1);

        let t2 = &v1.exp();
        println!("t2: {:#?}", t2);

        println!("t2: {:#?}", v1);

        let t3 = tensor!([1., 2., 3.]);
        let t3 = t3.exp();
        println!("t3: {:#?}", t3);
    }

    #[test]
    fn var_sum() {
        let v1 = Var::new("t1", tf32!([1., 2., 3.]));

        assert_eq!(&v1 + 2., tf32!([3., 4., 5.]));
        assert_eq!(2. + &v1, tf32!([3., 4., 5.]));

        assert_eq!(&v1 * 2., tf32!([2., 4., 6.]));
        assert_eq!(2. * &v1, tf32!([2., 4., 6.]));
    }
}
