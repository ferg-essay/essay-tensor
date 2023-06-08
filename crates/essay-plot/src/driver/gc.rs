use std::marker::PhantomData;

use crate::{frame::{Bounds}, artist::{Path, Color}};

use super::Canvas;

pub struct GraphicsContext {
    clip_bounds: Option<Bounds<Canvas>>,
    clip_path: Option<Path<Canvas>>,

    linewidth: f32,
    color: Color, // rgba
}

impl GraphicsContext {
    pub fn clip_rectangle(&mut self, clip_bounds: Bounds<Canvas>) -> &mut Self {
        self.clip_bounds = Some(clip_bounds);

        self
    }

    pub fn get_clip_rectangle(&self) -> &Option<Bounds<Canvas>> {
        &self.clip_bounds
    }

    pub fn clip_path(&mut self, path: Path<Canvas>) -> &mut Self {
        self.clip_path = Some(path);

        self
    }

    pub fn get_clip_path(&self) -> &Option<Path<Canvas>> {
        &self.clip_path
    }

    pub fn linewidth(&mut self, linewidth: f32) -> &mut Self {
        self.linewidth = linewidth;

        self
    }

    pub fn get_linewidth(&self) -> f32 {
        self.linewidth
    }

    pub fn color(&mut self, color: impl Into<Color>) -> &mut Self {
        self.color = color.into();

        self
    }

    pub fn get_color(&self) -> Color {
        self.color
    }
}

impl Default for GraphicsContext {
    fn default() -> Self {
        Self { 
            clip_bounds: None,
            clip_path: None,

            linewidth: 4.,
            color: Color(0x000000ff),
        }
    }
}
