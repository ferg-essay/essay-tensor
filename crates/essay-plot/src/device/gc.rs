use std::marker::PhantomData;

use crate::{axes::{Bounds}, artist::Path};

use super::Device;

pub struct GraphicsContext<'a> {
    clip_bounds: Option<Bounds<Device>>,
    clip_path: Option<Path<Device>>,

    marker: PhantomData<&'a u8>,
}

impl GraphicsContext<'_> {
    pub fn clip_rectangle(&mut self, clip_bounds: Bounds<Device>) -> &mut Self {
        self.clip_bounds = Some(clip_bounds);

        self
    }

    pub fn get_clip_rectangle(&self) -> &Option<Bounds<Device>> {
        &self.clip_bounds
    }

    pub fn clip_path(&mut self, path: Path<Device>) -> &mut Self {
        self.clip_path = Some(path);

        self
    }

    pub fn get_clip_path(&self) -> &Option<Path<Device>> {
        &self.clip_path
    }
}

impl Default for GraphicsContext<'_> {
    fn default() -> Self {
        Self { 
            clip_bounds: None,
            clip_path: None,
            marker: Default::default() 
        }
    }
}
