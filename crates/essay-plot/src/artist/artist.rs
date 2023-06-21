use essay_plot_base::{
    Coord, Bounds, Affine2d, Canvas, PathOpt,
    driver::Renderer, Clip,
};

pub trait Artist<M: Coord> {
    fn update(&mut self, canvas: &Canvas);

    fn get_extent(&mut self) -> Bounds<M>;
    
    fn draw(
        &mut self, 
        renderer: &mut dyn Renderer,
        to_canvas: &Affine2d,
        clip: &Clip,
        style: &dyn PathOpt,
    );
}
