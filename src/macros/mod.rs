#[macro_export]
macro_rules! tensor {
    [ $([ $([ $( $x:expr),* $(,)?]),* $(,)?]),* $(,)?] => {
        $crate::Tensor::from([$([$([$($x),*]),*]),*])
    };
    [ $([ $($x:expr),* $(,)?]),* $(,)?] => {
        $crate::Tensor::from([$([$($x),*]),*])
    };
    [ $($x:expr),* $(,)?] => {
        $crate::Tensor::from([$($x),*])
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
