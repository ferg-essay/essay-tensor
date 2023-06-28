#[macro_export]
macro_rules! tensor {
    ([ $([ $([ $( $x:expr),* $(,)?]),* $(,)?]),* $(,)?]) => {
        $crate::Tensor::from([$([$([$($x),*]),*]),*])
    };
    ([ $([ $($x:expr),* $(,)?]),* $(,)?]) => {
        $crate::Tensor::from([$([$($x),*]),*])
    };
    ([ $( $x:expr),* $(,)?]) => {
        $crate::Tensor::from([$($x),*])
    };
    ( $x:expr ) => {
        $crate::Tensor::from($x)
    };
}

#[macro_export]
macro_rules! tf32 {
    ([ $([ $([ $( $x:expr),* $(,)?]),* $(,)?]),* $(,)?]) => {
        $crate::Tensor::<f32>::from([$([$([$($x),*]),*]),*])
    };
    ([ $([ $($x:expr),* $(,)?]),* $(,)?]) => {
        $crate::Tensor::<f32>::from([$([$($x),*]),*])
    };
    ([ $( $x:expr),* $(,)?]) => {
        $crate::Tensor::<f32>::from([$($x),*])
    };
    ( $x:expr ) => {
        $crate::Tensor::<f32>::from($x)
    };
    ( ) => {
        $crate::Tensor::<f32>::empty()
    };
}

#[macro_export]
macro_rules! tc32 {
    ([ $([ $( ($re:expr, $im:expr) ),* $(,)?]), * $(,)?]) => {
        $crate::Tensor::<$crate::tensor::C32>::from([
            $([
                $( $crate::tensor::C32 { re: $re, im: $im } ),*
            ]),*
        ])
    };

    ([ $( ($re:expr, $im:expr) ),* $(,)?]) => {
        $crate::Tensor::<$crate::tensor::C32>::from([$( $crate::tensor::C32 { re: $re, im: $im } ),*])
    };

    ( $re:expr, $im:expr  ) => {
        $crate::Tensor::<$crate::tensor::C32>::from($crate::tensor::C32 { re: $re, im: $im })
    };

    ( ) => {
        $crate::Tensor::<$crate::tensor::C32>::empty()
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
