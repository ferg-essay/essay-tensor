use essay_plot_base::{Bounds, Canvas, Point, CoordMarker};

pub struct Layout {
    boxes: Vec<Box>,

    bounds: Bounds<Layout>,
}

impl Layout {
    pub fn new() -> Self {
        Self {
            boxes: Vec::new(),
            bounds: Bounds::unit(),
        }
    }

    pub fn push(&mut self, id: usize, bound: impl Into<Bounds<Layout>>) -> &mut Self {
        let bound = bound.into();

        assert!(bound.xmin() >= 0.);
        assert!(bound.ymin() >= 0.);

        // arbitrary limit for now
        assert!(bound.width() <= 11.);
        assert!(bound.height() <= 11.);

        self.bounds = self.bounds.union(&bound);

        self.boxes.push(Box::new(LayoutId(id), bound));

        self
    }

    pub fn bounds(&self) -> &Bounds<Layout> {
        &self.bounds
    }

    pub fn layout(&mut self, canvas: &Bounds<Canvas>) -> &Vec<Box> {
        let bounds = self.layout_bounds();

        assert!(bounds.xmin() == 0.);
        assert!(bounds.ymin() == 0.);

        assert!(1. <= bounds.width() && bounds.width() <= 11.);
        assert!(1. <= bounds.height() && bounds.height() <= 11.);
        
        // TODO: nonlinear grid sizes
        let h = canvas.height();
        let dw = canvas.width() / bounds.width();
        let dh = canvas.height() / bounds.height();

        for item in &mut self.boxes {
            let pos_layout = &item.pos_layout;

            item.pos_canvas = Bounds::new(
                Point(dw * pos_layout.xmin(), h - dh * pos_layout.ymax()),
                Point(dw * pos_layout.xmax(), h - dh * pos_layout.ymin()),
            );
        }


        &self.boxes
    }

    fn layout_bounds(&self) -> Bounds<Layout> {
        let mut bounds = Bounds::unit();

        for item in &self.boxes {
            bounds = bounds.union(&item.pos_layout);
        }

        bounds
    }
}

impl CoordMarker for Layout {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LayoutId(usize);

impl LayoutId {
    pub fn index(&self) -> usize {
        self.0
    }
}

pub struct Box {
    id: LayoutId,
    pos_layout: Bounds<Layout>,
    pos_canvas: Bounds<Canvas>,
}

impl Box {
    fn new(id: LayoutId, bounds: impl Into<Bounds<Layout>>) -> Self {
        Self {
            id,
            pos_layout: bounds.into(),
            pos_canvas: Bounds::unit(),
        }
    }

    #[inline]
    pub fn id(&self) -> LayoutId {
        self.id
    }

    #[inline]
    pub fn layout(&self) -> &Bounds<Layout> {
        &self.pos_layout
    }

    #[inline]
    pub fn pos_canvas(&self) -> &Bounds<Canvas> {
        &self.pos_canvas
    }
}


