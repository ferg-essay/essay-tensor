use crate::axes::{CoordMarker, Bounds};

use super::{Renderer, backend::Backend};

pub struct Device {
    backend: Box<dyn Backend>,
}

impl Device {
    pub fn new(backend: impl Backend + 'static) -> Self {
        Self {
            backend: Box::new(backend)
        }
    }

    pub fn renderer(&mut self) -> &dyn Renderer {
        self.backend.renderer()
    }

    pub fn main_loop(&mut self) -> Result<()> {
        self.backend.main_loop()
    }
}

impl Default for Device {
    fn default() -> Self {
        let backend = NoneBackend;

        Self { 
            backend: Box::new(backend),
        }
    }
}

impl CoordMarker for Device {}

#[derive(Debug)]
pub enum DeviceErr {
    NotImplemented,
}

pub type Result<T, E = DeviceErr> = std::result::Result<T, E>;

struct NoneBackend;

impl Backend for NoneBackend {
    fn renderer(&mut self) -> &dyn Renderer {
        todo!()
    }
}
