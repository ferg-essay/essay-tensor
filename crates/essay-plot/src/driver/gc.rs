use std::marker::PhantomData;

use crate::{figure::{Bounds}, artist::Path};

use super::Device;

pub struct GraphicsContext {
    clip_bounds: Option<Bounds<Device>>,
    clip_path: Option<Path<Device>>,

    linewidth: f32,
    rgba: u32, // rgba
}

impl GraphicsContext {
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

    pub fn linewidth(&mut self, linewidth: f32) -> &mut Self {
        self.linewidth = linewidth;

        self
    }

    pub fn get_linewidth(&self) -> f32 {
        self.linewidth
    }

    pub fn rgba(&mut self, rgba: u32) -> &mut Self {
        self.rgba = rgba;

        self
    }

    pub fn get_rgba(&self) -> u32 {
        self.rgba
    }
}

impl Default for GraphicsContext {
    fn default() -> Self {
        Self { 
            clip_bounds: None,
            clip_path: None,

            linewidth: 4.,
            rgba: 0x000000ff,
        }
    }
}
