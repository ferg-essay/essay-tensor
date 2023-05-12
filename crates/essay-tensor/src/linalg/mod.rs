mod sgemm;
mod vecmul;
mod matvec;
mod matmul;

pub use matmul::{
    matmul,
};

pub use matvec::{
    matvec, matvec_t,
};

pub use vecmul::{
    // inner_product, 
    outer_product,
};

