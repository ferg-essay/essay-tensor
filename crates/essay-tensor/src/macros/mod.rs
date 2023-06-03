#[macro_export]
macro_rules! tensor {
    ([ $([ $([ $( $x:expr),* $(,)?]),* $(,)?]),* $(,)?]) => {
        Tensor::<f32>::from([$([$([$($x),*]),*]),*])
    };
    ([ $([ $($x:expr),* $(,)?]),* $(,)?]) => {
        Tensor::<f32>::from([$([$($x),*]),*])
    };
    ([ $( $x:expr),* $(,)?]) => {
        Tensor::from([$($x),*])
    };
    ( $x:expr ) => {
        Tensor::from($x)
    };
}

#[macro_export]
macro_rules! tf32 {
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
            unary_op(a, $op)
        }

        impl Tensor {
            pub fn $fun(&self) -> Tensor {
                unary_op(self, $op)
            }
        }
    }
}

#[macro_export]
macro_rules! tensor_binop {
    ($fun:ident, $op:expr) => {
        pub fn $fun(a: impl Into<Tensor>, b: impl Into<Tensor>) -> Tensor {
            binary_op(a, b, $op)
        }

        impl Tensor {
            pub fn $fun(&self, b: impl Into<Tensor>) -> Tensor {
                binary_op(self, b, $op)
            }
        }
    }
}
