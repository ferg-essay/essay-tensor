use essay_tensor::Tensor;

use crate::{
    driver::{Backend, Canvas, Renderer}, plot::{PlotOpt}, 
    frame::{Axes, CoordMarker, Bounds}
};

use super::gridspec::GridSpec;

pub struct Figure {
    device: Canvas,
    // inner: Arc<Mutex<FigureInner>>,
    inner: FigureInner,
}

impl Figure {
    pub fn new() -> Self {
        Self {
            device: Default::default(),
            // inner: Arc::new(Mutex::new(FigureInner::new())),
            inner: FigureInner::new(),
        }
    }

    pub fn axes(&mut self, axes: impl Into<Axes>) -> &mut Axes {
        self.inner.axes(axes)
    }

    pub fn plot(
        &mut self, 
        x: impl Into<Tensor>, 
        y: impl Into<Tensor>, 
        opt: impl Into<PlotOpt>
    ) -> &Axes {
        let axes = self.axes(());
        
        axes.plot(x, y, opt);

        axes
    }

    pub fn show(self) {
        // let mut figure = self;
        let inner = self.inner;
        let mut device = self.device;

        device.main_loop(inner).unwrap();

        todo!();
    }
}

pub struct FigureInner {
    gridspec: Bounds<GridSpec>,

    frames: Vec<Axes>,
}

impl FigureInner {
    fn new() -> Self {
        Self {
            gridspec: Bounds::none(),
            frames: Default::default(),
        }
    }

    fn axes(
        &mut self, 
        axes: impl Into<Axes>, 
    ) -> &mut Axes {
        let axes = Axes::new(Bounds::<GridSpec>::none());

        self.frames.push(axes);

        let len = self.frames.len();
        &mut self.frames[len - 1]
    }

    pub fn draw(&mut self, renderer: &mut impl Renderer) {
        let bounds = renderer.get_canvas_bounds();

        for frame in &mut self.frames {
            frame.set_pos(&bounds);

            frame.draw(renderer);
        }
    }
}

impl CoordMarker for Figure {}
