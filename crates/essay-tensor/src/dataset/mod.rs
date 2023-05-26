mod tensor;
mod take;
mod range;
mod dataset;

pub use range::range;
pub use dataset::{
    Dataset, DatasetIter, IntoFlowBuilder,
};
pub use tensor::{
    from_tensors, from_tensor_slices,
};
