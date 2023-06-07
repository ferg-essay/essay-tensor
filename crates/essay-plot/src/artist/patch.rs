use core::fmt;

use crate::{figure::{Point, Data, Affine2d, Bounds}, driver::{Renderer, Device}};

use super::{Path, path::Angle, ArtistTrait, Color, StyleOpt};

pub trait Patch {
    fn get_path(&mut self) -> &Path;
}

pub struct Wedge {
    center: Point,
    radius: f32,
    angle: Angle,

    path: Option<Path<Data>>,
}

impl Wedge {
    pub fn new(
        center: Point,
        radius: f32,
        angle: Angle,
    ) -> Self {
        //println!("Wedge {:?} angle {:?}-{:?}", center, angle.0, angle.1);

        Self {
            center,
            radius,
            angle,
            path: None,
        }
    }

    //pub fn color(&mut self, color: Color) {
    //    self.color = Some(color);
    //}
}

impl Patch for Wedge {
    fn get_path(&mut self) -> &Path {
        if self.path.is_none() {
            let wedge = Path::<Data>::wedge(self.angle);
            
            //println!("Wedge {:?}", wedge.codes());
            let transform = Affine2d::eye()
                .scale(self.radius, self.radius)
                .translate(self.center.x(), self.center.y());

            let wedge = wedge.transform::<Data>(&transform);

            self.path = Some(wedge);
        }

        match &self.path {
            Some(path) => path,
            None => todo!(),
        }        
    }
}

impl ArtistTrait for Wedge {
    fn get_data_bounds(&mut self) -> Bounds<Data> {
        self.get_path().get_bounds()
    }

    fn draw(
        &mut self, 
        renderer: &mut dyn Renderer,
        to_device: &Affine2d,
        clip: &Bounds<Device>,
        style: &dyn StyleOpt,
    ) {
        if let Some(path) = &self.path {
            renderer.draw_path(style, path, to_device, clip).unwrap();
        }
    }
}

impl fmt::Debug for Wedge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Wedge(({}, {}), {}, [{}, {}])",
            self.center.x(), self.center.y(),
            self.radius,
            self.angle.0, self.angle.1)
    }
}
