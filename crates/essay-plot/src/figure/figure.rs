use essay_tensor::Tensor;

use crate::{backend::Backend, plot::{PlotOpt, Plot}, axes::Axes};

use super::gridspec::GridSpec;

pub struct Figure {
    backend: Option<Box<dyn Backend>>,
    gridspec: Option<GridSpec>,

    axes: Vec<Axes>,
}

impl Figure {
    pub fn new() -> Self {
        Self {
            backend: None,
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
        let axes = Axes::new([0., 0., 1., 1.]);

        self.axes.push(axes);

        let len = self.axes.len();
        &self.axes[len - 1]
    }

    pub fn show(&self) {

    }
}
