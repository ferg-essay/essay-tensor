use super::{Renderer, device, DeviceErr};


pub trait Backend {
    fn renderer(&mut self) -> &dyn Renderer;

    fn main_loop(&mut self) -> device::Result<()> {
        Err(DeviceErr::NotImplemented)
    }
}
