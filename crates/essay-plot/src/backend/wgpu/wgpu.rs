use wgpu::{InstanceDescriptor, Instance};

use crate::backend::{Backend, BackendErr};

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
    fn main_loop(&mut self) -> Result<(), BackendErr> {
        main_loop();

        Ok(())
    }
}