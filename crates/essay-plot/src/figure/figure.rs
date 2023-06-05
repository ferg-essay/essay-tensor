use essay_tensor::Tensor;

use crate::{device::{Backend, Device}, plot::{PlotOpt, Plot}, axes::{Axes, CoordMarker, Bounds, Point}};

use super::gridspec::GridSpec;

pub struct Figure {
    device: Device,
    gridspec: Option<GridSpec>,

    axes: Vec<Axes>,
}

impl Figure {
    pub fn new() -> Self {
        Self {
            device: Default::default(),
            gridspec: None,
            axes: Default::default(),
        }
    }

    pub fn plot(
        &mut self, 
        x: impl Into<Tensor>, 
        y: impl Into<Tensor>, 
        opt: impl Into<PlotOpt>
    ) -> &Axes {
        let axes = Axes::new(Bounds::<Figure>::new(Point(0., 0.), Point(1., 1.)));

        self.axes.push(axes);

        let len = self.axes.len();
        &self.axes[len - 1]
    }

    pub fn show(&self) {

    }
}

impl CoordMarker for Figure {}
