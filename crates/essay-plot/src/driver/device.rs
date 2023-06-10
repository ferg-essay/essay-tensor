use crate::{graph::{CoordMarker, Bounds}, graph::FigureInner, };

use super::{Renderer, backend::Backend, wgpu::WgpuBackend};

pub struct Canvas {
    backend: Box<dyn Backend>,
}

impl Canvas {
    pub fn new(backend: impl Backend + 'static) -> Self {
        Self {
            backend: Box::new(backend)
        }
    }

    /*
    pub fn renderer(&mut self) -> &dyn Renderer {
        self.backend.renderer()
    }
    */

    pub fn main_loop(&mut self, figure: FigureInner) -> Result<()> {
        self.backend.main_loop(figure)
    }
}

impl Default for Canvas {
    fn default() -> Self {
        // let backend = NoneBackend;

        let backend = WgpuBackend::new();

        Self { 
            backend: Box::new(backend),
        }
    }
}

impl CoordMarker for Canvas {}

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
