use core::fmt;
use std::{ops::{Deref, self}, rc::Rc, cell::RefCell};

use crate::{tensor::{Dtype}, Tensor, prelude::IntoTensor};

use super::{module::ModuleTape};

pub struct Var<D:Dtype=f32> {
    name: String,
    tensor_share: Rc<RefCell<Tensor<D>>>,
    tensor: Tensor<D>,
}

impl<D:Dtype> Var<D> {
    pub fn new(name: &str, tensor: Tensor<D>) -> Self {
        let tensor = tensor.to_var(name);

        Self {
            tensor_share: Rc::new(RefCell::new(tensor.clone())),
            tensor: tensor,
            name: name.to_string(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn set(&mut self, tensor: Tensor<D>) {
        let tensor = tensor.to_var(&self.name);

        self.tensor = tensor.clone();
        self.tensor_share.replace(tensor);
    }
}

impl Var {
    pub fn tensor(&self) -> &Tensor {
        // Tape::set_var(&self.name, &self.tensor);
        ModuleTape::var(&self);
    
        &self.tensor
    }

    pub(crate) fn tensor_raw(&self) -> &Tensor {
        &self.tensor
    }
}

impl Deref for Var {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        // Tape::set_var(&self.name, &self.tensor);
        ModuleTape::var(&self);

        &self.tensor
    }
}

impl Tensor {
    pub fn as_var(self, name: &str) -> Var {
        Var::new(name, self)
    }
}

impl fmt::Debug for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Var")
            .field("name", &self.name)
            .field("tensor", &self.tensor)
            .finish()
    }
}

impl Clone for Var {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            tensor_share: self.tensor_share.clone(),
            tensor: self.tensor.clone(),
        }
    }
}

impl<D:Dtype> IntoTensor<D> for Var<D> {
    fn into_tensor(&self) -> Tensor<D> {
        self.tensor.clone()
    }
}

impl<D:Dtype> From<Var<D>> for Tensor<D> {
    fn from(value: Var<D>) -> Self {
        value.tensor.clone()
    }
}

macro_rules! var_ops {
    ($op:ident, $fun:ident) => {
        impl ops::$op<&Tensor> for &Var {
            type Output = Tensor;
        
            fn $fun(self, rhs: &Tensor) -> Self::Output {
                self.deref().$fun(rhs)
            }
        }

        impl ops::$op<Tensor> for &Var {
            type Output = Tensor;
        
            fn $fun(self, rhs: Tensor) -> Self::Output {
                self.deref().$fun(&rhs)
            }
        }

        impl ops::$op<&Var> for &Tensor {
            type Output = Tensor;
        
            fn $fun(self, rhs: &Var) -> Self::Output {
                self.$fun(rhs.deref())
            }
        }

        impl ops::$op<&Var> for Tensor {
            type Output = Tensor;
        
            fn $fun(self, rhs: &Var) -> Self::Output {
                self.$fun(rhs.deref())
            }
        }

        impl ops::$op<&Var> for &Var {
            type Output = Tensor;
        
            fn $fun(self, rhs: &Var) -> Self::Output {
                self.$fun(rhs.deref())
            }
        }
    }
}

var_ops!(Add, add);
var_ops!(Sub, sub);
var_ops!(Mul, mul);

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_var() {
        let t1 = tensor!([1., 2., 3.]);
        let v1 = t1.as_var("t1");

        let t2 = v1.exp();
        println!("t2: {:#?}", t2);

        println!("t2: {:#?}", v1);

        let t3 = tensor!([1., 2., 3.]);
        let t3 = t3.exp();
        println!("t3: {:#?}", t3);
    }
}
