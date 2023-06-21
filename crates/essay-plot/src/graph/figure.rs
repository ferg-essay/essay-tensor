use std::ops;

use essay_plot_wgpu::WgpuBackend;

use essay_plot_base::{
    driver::{Renderer, Backend, FigureApi},
    Coord, Bounds, Point, CanvasEvent, Canvas,
};

use crate::{graph::Graph, frame::{Layout, LayoutArc}};

use super::config::read_config;

pub struct Figure {
    // inner: Arc<Mutex<FigureInner>>,
    device: Box<dyn Backend>,
    inner: FigureInner,
}

impl Figure {
    pub fn new() -> Self {
        Self {
            // inner: Arc::new(Mutex::new(FigureInner::new())),
            inner: FigureInner::new(),
            device: Box::new(WgpuBackend::new()),
        }
    }

    pub fn new_graph(&mut self, grid: impl Into<Bounds<Layout>>) -> &mut Graph {
        self.inner.new_graph(grid)
    }

    pub fn graph(&mut self, id: GraphId) -> &mut Graph {
        self.inner.graph_mut(id)
    }

    pub fn poly_graphs<'a, R: PolyRow<'a>>(&'a mut self, layout: R) -> R::Item {
        let mut row = 0;
        //R::axes(self, layout, &mut row)
        todo!()
    }

    pub fn show(self) {
        // let mut figure = self;
        let inner = self.inner;
        let mut device = self.device;

        device.main_loop(Box::new(inner)).unwrap();

        todo!();
    }
}

impl ops::Index<GraphId> for Figure {
    type Output = Graph;

    fn index(&self, index: GraphId) -> &Self::Output {
        self.inner.graph(index)
    }
}

impl ops::IndexMut<GraphId> for Figure {
    fn index_mut(&mut self, index: GraphId) -> &mut Self::Output {
        self.inner.graph_mut(index)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct GraphId(usize);

impl GraphId {
    #[inline]
    pub fn index(&self) -> usize {
        self.0
    }
}

pub struct FigureInner {
    gridspec: Bounds<Layout>,
    layout: LayoutArc,

    graphs: Vec<Graph>,
}

impl FigureInner {
    fn new() -> Self {
        let config = read_config();
        Self {
            layout: LayoutArc::new(config),
            gridspec: Bounds::none(),
            graphs: Default::default(),
        }
    }

    fn new_graph(
        &mut self, 
        grid: impl Into<Bounds<Layout>>, 
    ) -> &mut Graph {
        let len = self.graphs.len();
        let id = GraphId(len);

        let mut grid : Bounds<Layout> = grid.into();

        if grid.is_zero() || grid.is_none() {
            if id.index() == 0 {
                grid = Bounds::unit();
            } else {
                let layout = self.layout.bounds();
                grid = Bounds::new(
                    Point(0., layout.ymax()),
                    Point(1., layout.ymax() + 1.),
                );
            }
        }

        let frame_id = self.layout.add_frame(grid.clone());

        let graph = Graph::new(frame_id, self.layout.clone());

        self.graphs.push(graph);

        &mut self.graphs[len]
    }

    fn graph(&self, id: GraphId) -> &Graph {
        &self.graphs[id.index()]
    }

    fn graph_mut(&mut self, id: GraphId) -> &mut Graph {
        &mut self.graphs[id.index()]
    }
}

impl FigureApi for FigureInner {
    fn draw(&mut self, renderer: &mut dyn Renderer, _bounds: &Bounds<Canvas>) {
        self.layout.draw(renderer);
    }

    fn event(&mut self, renderer: &mut dyn Renderer, event: &CanvasEvent) {
        self.layout.event(renderer, event);
    }
}

impl Coord for Figure {}

pub trait PolyRow<'a> {
    type Item;

    fn axes(figure: &'a mut Figure, layout: Self, row: &mut Counter) -> Self::Item;
}

pub trait PolyCol<'a> {
    type Item;

    fn axes(figure: &'a mut Figure, layout: Self, row: usize, col: &mut Counter) -> Self::Item;
}

impl<'a> PolyRow<'a> for [usize; 0] {
    type Item = GraphId;

    fn axes(figure: &'a mut Figure, layout: Self, row: &mut Counter) -> Self::Item {
        PolyRow::axes(figure, [1, 1], row)
    }
}

impl<'a> PolyRow<'a> for [usize; 1] {
    type Item = GraphId;

    fn axes(figure: &'a mut Figure, layout: Self, row: &mut Counter) -> Self::Item {
        PolyRow::axes(figure, [layout[0], 1], row)
    }
}

impl<'a> PolyRow<'a> for [usize; 2] {
    type Item = GraphId;

    fn axes(figure: &'a mut Figure, layout: Self, row: &mut Counter) -> Self::Item {
        let rows = layout[0];
        let cols = layout[1];

        let graph = figure.new_graph(Bounds::new(
            Point(0., row.0 as f32), 
            Point(0., (row.0 + rows) as f32),
        ));

        row.0 += rows;

        //graph.id()
        todo!()
    }
}

impl<'a> PolyCol<'a> for [usize; 0] {
    type Item = &'a mut Graph;

    fn axes(figure: &'a mut Figure, layout: Self, row: usize, col: &mut Counter) -> Self::Item {
        PolyCol::axes(figure, [1], row, col)
    }
}

impl<'a> PolyCol<'a> for [usize; 1] {
    type Item = &'a mut Graph;

    fn axes(figure: &'a mut Figure, layout: Self, row: usize, col: &mut Counter) -> Self::Item {
        let cols = layout[0];

        let axes = figure.new_graph(Bounds::new(
            Point(col.0 as f32, row as f32), 
            Point((col.0 + cols) as f32, row as f32),
        ));

        col.0 += cols;

        axes
    }
}

impl<'a> PolyCol<'a> for [usize; 2] {
    type Item = &'a mut Graph;

    fn axes(figure: &'a mut Figure, layout: Self, row: usize, col: &mut Counter) -> Self::Item {
        let cols = layout[0];

        let axes = figure.new_graph(Bounds::new(
            Point(col.0 as f32, row as f32), 
            Point((col.0 + cols) as f32, row as f32),
        ));

        col.0 += cols;

        axes
    }
}

impl<'a> PolyRow<'a> for () {
    type Item = ();

    fn axes(figure: &'a mut Figure, layout: Self, row: &mut Counter) -> Self::Item {
        row.0 += 1;

        ()
    }
}

impl<'a, R1:PolyCol<'a>> PolyRow<'a> for (R1,) {
    type Item = (R1::Item,);

    fn axes(figure: &'a mut Figure, layout: Self, row: &mut Counter) -> Self::Item {
        let (r1,) = layout;
        (
            R1::axes(figure, r1, row.0, &mut Counter(0)),
        )
    }
}

impl<'a, R1:PolyCol<'a>, R2:PolyCol<'a>> PolyRow<'a> for (R1, R2) {
    type Item = (R1::Item, R2::Item);

    fn axes(figure: &'a mut Figure, layout: Self, row: &mut Counter) -> Self::Item {
        let (r1, r2) = layout;
        todo!();
        /*
        (
            R1::axes(figure, r1, row.0, &mut Counter(0)),
            R2::axes(figure, r2, row.0, &mut Counter(0)),
        )
        */
    }
}

impl<'a, R1:PolyCol<'a>> PolyCol<'a> for (R1,) {
    type Item = (R1::Item,);

    fn axes(figure: &'a mut Figure, layout: Self, row: usize, col: &mut Counter) -> Self::Item {
        let (r1,) = layout;
        (
            R1::axes(figure, r1, row, col),
        )
    }
}

impl<'a, R1:PolyCol<'a>, R2:PolyCol<'a>> PolyCol<'a> for (R1, R2) {
    type Item = (R1::Item, R2::Item);

    fn axes(figure: &'a mut Figure, layout: Self, row: usize, col: &mut Counter) -> Self::Item {
        let (r1, r2) = layout;
        todo!();
        /*
        (
            R1::axes(figure, r1, row, col),
            R2::axes(figure, r2, row, col),
        )
        */
    }
}

pub struct Counter(usize);

#[cfg(test)]
mod test {
    use super::Figure;

    #[test]
    fn test_polyaxes() {
        let mut figure = Figure::new();

        let axes = figure.poly_graphs([]);
        let axes = figure.poly_graphs(([], [2]));
        let axes = figure.poly_graphs((
            ([], []),
            ([2, 2]),
        ));
    }
}

