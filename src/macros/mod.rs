#[macro_export]
macro_rules! ten {
    [ $([ $([ $( $x:expr),* $(,)?]),* $(,)?]),* $(,)?] => {
        $crate::tensor::Tensor::from([$([$([$($x),*]),*]),*])
    };
    [ $([ $($x:expr),* $(,)?]),* $(,)?] => {
        $crate::tensor::Tensor::from([$([$($x),*]),*])
    };
    [ $($x:expr),* $(,)?] => {
        $crate::tensor::Tensor::from([$($x),*])
    };
}
/*
#[macro_export]
macro_rules! ten {
    ([ $([ $([ $( $x:expr),* $(,)?]),* $(,)?]),* $(,)?]) => {
        $crate::tensor::Tensor::<f32>::from([$([$([$($x),*]),*]),*])
    };
    ([ $([ $($x:expr),* $(,)?]),* $(,)?]) => {
        $crate::tensor::Tensor::<f32>::from([$([$($x),*]),*])
    };
    ([ $( $x:expr),* $(,)?]) => {
        $crate::tensor::Tensor::<f32>::from([$($x),*])
    };
    ( $x:expr ) => {
        $crate::tensor::Tensor::<f32>::from($x)
    };
    ( ) => {
        $crate::Tensor::<f32>::empty()
    };
}
    */
