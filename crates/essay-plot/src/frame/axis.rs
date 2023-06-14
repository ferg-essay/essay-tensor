use super::{tick_locator::{LinearLocator, TickLocator, MaxNLocator}, databox::DataBox};

pub struct Axis {
    locator: Box<dyn TickLocator>,
}

impl Axis {
    pub fn new() -> Self {
        Self {
            //locator: Box::new(LinearLocator::new(None)),
            locator: Box::new(MaxNLocator::new(None)),
        }
    }

    //pub fn ticks(&self, data: &DataBox) -> Tensor<f32> {
    //}

    pub fn x_ticks(&self, data: &DataBox) -> Vec<(f32, f32)> {
        let c_width = data.get_pos().width();

        let view = data.get_view_bounds();
        let v_width = view.width();

        let (vmin, vmax) = (view.xmin(), view.xmax());
        let (min, max) = self.locator.view_limits(vmin, vmax);

        // self.locator.tick_values(min, max)

        let mut x_vec = Vec::<(f32, f32)>::new();

        for x in self.locator.tick_values(min, max).iter() {
            x_vec.push((*x, ((x - vmin) * c_width / v_width).round()));
        }

        x_vec
    }

    pub fn y_ticks(&self, data: &DataBox) -> Vec<(f32, f32)> {
        let v_height = data.get_view_bounds().height();
        let c_height = data.get_pos().height();

        let view = data.get_view_bounds();

        let (vmin, vmax) = (view.ymin(), view.ymax());
        let (min, max) = self.locator.view_limits(vmin, vmax);
        
        // self.locator.tick_values(min, max)

        let mut y_vec = Vec::<(f32, f32)>::new();

        for y in self.locator.tick_values(min, max).iter() {
            y_vec.push((*y, ((y - vmin) * c_height / v_height).round()));
        }

        y_vec
    }
}