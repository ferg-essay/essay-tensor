#[macro_export]
macro_rules! tensor {
    ([ $([ $([ $( $x:expr),* $(,)?]),* $(,)?]),* $(,)?]) => {
        Tensor::<3>::from([$([$([$($x),*]),*]),*])
    };
    ([ $([ $($x:expr),* $(,)?]),* $(,)?]) => {
        Tensor::<2>::from([$([$($x),*]),*])
    };
    ([ $( $x:expr),* $(,)?]) => {
        Tensor::<1>::from([$($x),*])
    };
    ( $x:expr ) => {
        Tensor::<0>::from($x)
    };
}
