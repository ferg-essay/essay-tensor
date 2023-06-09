use essay_tensor::Tensor;

use crate::driver::Canvas;

use super::{tick_locator::{LinearLocator, Locator}, Bounds, Data, databox::DataBox};

pub struct Axis {
    locator: Box<dyn Locator>,
}

impl Axis {
    pub fn new() -> Self {
        Self {
            locator: Box::new(LinearLocator::new(None)),
        }
    }

    //pub fn ticks(&self, data: &DataBox) -> Tensor<f32> {
    //}

    pub fn x_ticks(&self, data: &DataBox) -> Vec<(f32, f32)> {
        let v_width = data.get_view_bounds().width();
        let c_width = data.get_pos().width();

        let view = data.get_view_bounds();

        let (min, max) = (view.xmin(), view.xmax());
        let (min, max) = self.locator.view_limits(min, max);

        // self.locator.tick_values(min, max)

        let mut x_vec = Vec::<(f32, f32)>::new();

        for x in self.locator.tick_values(min, max).iter() {
            x_vec.push((*x, ((x - min) * c_width / v_width).round()));
        }

        x_vec
    }

    pub fn y_ticks(&self, data: &DataBox) -> Vec<(f32, f32)> {
        let v_height = data.get_view_bounds().height();
        let c_height = data.get_pos().height();

        let view = data.get_view_bounds();

        let (min, max) = (view.ymin(), view.ymax());
        let (min, max) = self.locator.view_limits(min, max);
        
        // self.locator.tick_values(min, max)

        let mut y_vec = Vec::<(f32, f32)>::new();

        for y in self.locator.tick_values(min, max).iter() {
            y_vec.push((*y, ((y - min) * c_height / v_height).round()));
        }

        y_vec
    }
}