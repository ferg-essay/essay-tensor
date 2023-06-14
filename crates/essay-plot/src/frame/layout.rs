use std::{sync::Arc, ops::{Deref, DerefMut}, cell::{RefCell, Ref, RefMut}, rc::Rc};

use essay_plot_base::{Bounds, Canvas, Point, CoordMarker, driver::Renderer, CanvasEvent};

use super::Frame;

#[derive(Clone)]
pub struct LayoutArc(Rc<RefCell<Layout>>);

pub struct Layout {
    extent: Bounds<Layout>,

    frames: Vec<LayoutBox>,
}

impl Layout {
    pub fn new() -> Self {
        Self {
            extent: Bounds::unit(),

            frames: Vec::new(),
        }
    }

    pub fn add_frame(&mut self, bound: impl Into<Bounds<Layout>>) -> &mut Frame {
        let bound = bound.into();

        assert!(bound.xmin() >= 0.);
        assert!(bound.ymin() >= 0.);

        // arbitrary limit for now
        assert!(bound.width() <= 11.);
        assert!(bound.height() <= 11.);

        self.extent = self.extent.union(&bound);

        let id = FrameId(self.frames.len());

        let frame = Frame::new(id);

        self.frames.push(LayoutBox::new(frame, bound));

        self.frame_mut(id)
    }

    #[inline]
    pub fn frame(&self, id: FrameId) -> &Frame {
        self.frames[id.index()].frame()
    }

    #[inline]
    pub fn frame_mut(&mut self, id: FrameId) -> &mut Frame {
        self.frames[id.index()].frame_mut()
    }

    pub fn bounds(&self) -> &Bounds<Layout> {
        &self.extent
    }

    pub fn layout(&mut self, canvas: &Canvas) {
        let bounds = self.layout_bounds();

        assert!(bounds.xmin() == 0.);
        assert!(bounds.ymin() == 0.);

        assert!(1. <= bounds.width() && bounds.width() <= 11.);
        assert!(1. <= bounds.height() && bounds.height() <= 11.);
        
        // TODO: nonlinear grid sizes
        let h = canvas.height();
        let dw = canvas.width() / bounds.width();
        let dh = canvas.height() / bounds.height();

        for item in &mut self.frames {
            let pos_layout = &item.pos_layout;

            item.pos_canvas = Bounds::new(
                Point(dw * pos_layout.xmin(), h - dh * pos_layout.ymax()),
                Point(dw * pos_layout.xmax(), h - dh * pos_layout.ymin()),
            );
        }
    }

    fn layout_bounds(&self) -> Bounds<Layout> {
        let mut bounds = Bounds::unit();

        for item in &self.frames {
            bounds = bounds.union(&item.pos_layout);
        }

        bounds
    }

    pub(crate) fn draw(&mut self, renderer: &mut dyn Renderer) {
        self.layout(renderer.get_canvas());

        for item in &mut self.frames {
            item.draw(renderer);
        }
    }

    pub(crate) fn event(&mut self, renderer: &mut dyn Renderer, event: &CanvasEvent) {
        for item in &mut self.frames {
            let frame = item.frame_mut();

            if frame.pos().contains(event.point()) {
                frame.event(renderer, event);
            }
        }
    }
}

impl CoordMarker for Layout {}

impl LayoutArc {
    pub(crate) fn new() -> LayoutArc {
        LayoutArc(Rc::new(RefCell::new(Layout::new())))
    }
}

impl LayoutArc {
    #[inline]
    pub fn bounds(&self) -> Bounds<Layout> {
        self.0.borrow().bounds().clone()
    }

    #[inline]
    pub fn add_frame(&mut self, bound: impl Into<Bounds<Layout>>) -> FrameId {
        self.0.borrow_mut().add_frame(bound).id()
    }

    #[inline]
    pub fn borrow(&self) -> Ref<Layout> {
        self.0.borrow()
    }

    #[inline]
    pub fn borrow_mut(&self) -> RefMut<Layout> {
        self.0.borrow_mut()
    }

    /*
    #[inline]
    pub fn frame(&self, id: FrameId) -> &Frame {
        self.0.borrow().frame(id)
    }

    #[inline]
    pub fn frame_mut(&mut self, id: FrameId) -> &mut Frame {
        self.0.borrow_mut().frame_mut(id)
    }
    */

    pub(crate) fn draw(&mut self, renderer: &mut dyn Renderer) {
        self.0.borrow_mut().draw(renderer);
    }

    pub(crate) fn event(&mut self, renderer: &mut dyn Renderer, event: &CanvasEvent) {
        self.0.borrow_mut().event(renderer, event);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameId(usize);

impl FrameId {
    pub fn index(&self) -> usize {
        self.0
    }
}

pub struct LayoutBox {
    pos_layout: Bounds<Layout>,
    pos_canvas: Bounds<Canvas>,

    frame: Frame,
}

impl LayoutBox {
    fn new(frame: Frame, bounds: impl Into<Bounds<Layout>>) -> Self {
        Self {
            pos_layout: bounds.into(),
            pos_canvas: Bounds::unit(),

            frame,
        }
    }

    #[inline]
    pub fn id(&self) -> FrameId {
        self.frame.id()
    }

    #[inline]
    pub fn layout(&self) -> &Bounds<Layout> {
        &self.pos_layout
    }

    #[inline]
    pub fn pos_canvas(&self) -> &Bounds<Canvas> {
        &self.pos_canvas
    }

    #[inline]
    pub fn frame(&self) -> &Frame {
        &self.frame
    }

    #[inline]
    pub fn frame_mut(&mut self) -> &mut Frame {
        &mut self.frame
    }

    fn draw(&mut self, renderer: &mut dyn Renderer) {
        self.frame.update_extent(renderer.get_canvas());
        let pos_frame = self.pos_canvas.clone();
        self.frame.set_pos(&pos_frame);

        self.frame.draw(renderer);
    }
}


