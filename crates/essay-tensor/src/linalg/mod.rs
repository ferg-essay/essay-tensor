mod blas_naive;
mod blas;
mod matmul;
mod matvec;

pub use matmul::{
    matmul,
};

pub use matvec::{
    matvec, matvec_t,
};

