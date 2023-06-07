use crate::{driver::{Backend, Renderer, DeviceErr}, figure::FigureInner};

use super::main_loop::main_loop;

pub struct WgpuBackend {
}

impl WgpuBackend {
    pub fn new() -> Self {
        Self {

        }
    }
}

impl Backend for WgpuBackend {
    fn main_loop(&mut self, figure: FigureInner) -> Result<(), DeviceErr> {
        main_loop(figure);

        Ok(())
    }

    fn renderer(&mut self) -> &dyn Renderer {
        todo!()
    }
}
