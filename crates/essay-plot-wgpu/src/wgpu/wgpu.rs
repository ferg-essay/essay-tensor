use essay_plot_base::driver::{Backend, FigureApi, DeviceErr};

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
    fn main_loop(&mut self, figure: Box<dyn FigureApi>) -> Result<(), DeviceErr> {
        main_loop(figure);

        Ok(())
    }
    /*
    fn renderer(&mut self) -> &dyn Renderer {
        todo!()
    }
    */
}
