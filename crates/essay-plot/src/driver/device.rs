use crate::{graph::{CoordMarker, Bounds}, graph::FigureInner, };

use super::{Renderer, backend::Backend, wgpu::WgpuBackend};

#[derive(Debug)]
pub enum DeviceErr {
    NotImplemented,
}

pub type Result<T, E = DeviceErr> = std::result::Result<T, E>;

struct NoneBackend;

impl Backend for NoneBackend {
    /*
    fn renderer(&mut self) -> &dyn Renderer {
        todo!()
    }
    */
}
