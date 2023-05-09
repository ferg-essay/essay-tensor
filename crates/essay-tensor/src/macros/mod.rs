#[macro_export]
macro_rules! tensor {
    ([ $([ $([ $( $x:expr),* $(,)?]),* $(,)?]),* $(,)?]) => {
        Tensor::<f32>::from([$([$([$($x),*]),*]),*])
    };
    ([ $([ $($x:expr),* $(,)?]),* $(,)?]) => {
        Tensor::<f32>::from([$([$($x),*]),*])
    };
    ([ $( $x:expr),* $(,)?]) => {
        Tensor::<f32>::from([$($x),*])
    };
    ( $x:expr ) => {
        Tensor::<f32>::from($x)
    };
}

#[macro_export]
macro_rules! tensor_uop {
    ($fun:ident, $op:expr) => {
        pub fn $fun(a: &Tensor) -> Tensor {
            uop(a, $op)
        }

        impl Tensor {
            pub fn $fun(&self) -> Tensor {
                uop(self, $op)
            }
        }
    }
}

#[macro_export]
macro_rules! tensor_binop {
    ($fun:ident, $op:expr) => {
        pub fn $fun(a: &Tensor, b: &Tensor) -> Tensor {
            a.binop(b, $op)
        }

        impl Tensor {
            pub fn $fun(&self, b: &Tensor) -> Tensor {
                self.binop(b, $op)
            }
        }
    }
}
