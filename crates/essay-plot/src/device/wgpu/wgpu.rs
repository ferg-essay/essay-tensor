use wgpu::{InstanceDescriptor, Instance};

use crate::device::{Backend, Renderer, DeviceErr};

use super::main_loop::main_loop;

pub struct WgpuBackend {
    instance: Instance,
}

impl WgpuBackend {
    pub fn new() -> Self {
        let desc = InstanceDescriptor::default();
        
        let instance = Instance::new(desc);

        Self {
            instance,
        }
    }
}

impl Backend for WgpuBackend {
    fn main_loop(&mut self) -> Result<(), DeviceErr> {
        main_loop();

        Ok(())
    }

    fn renderer(&mut self) -> &dyn Renderer {
        todo!()
    }
}