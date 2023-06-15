use essay_plot_base::{Affine2d, Path};
use essay_tensor::Tensor;

use crate::{artist::{Lines2d, Collection, Container, paths, patch::{DataPatch, PathPatch}, ArtistStyle}, graph::{Graph, PlotOpt}, frame::{Data, ArtistId}};

pub fn bar_y(
    graph: &mut Graph, 
    y: impl Into<Tensor>, 
) -> PlotOpt {
    let y : Tensor = y.into();

    let mut container = Container::new();
    let width = 1.;

    for (i, value) in y.iter().enumerate() {
        let scale = Affine2d::eye()
            .scale(width, *value)
            .translate(i as f32 - 0.4, 0.);
        let path: Path<Data> = paths::unit().transform(&scale);
        
        let id = ArtistId::new(0);
        container.push(ArtistStyle::new(id, PathPatch::<Data>::new(path)));
    }

    //self.artist(lines)
    graph.add_data_artist(container)
}
// #[derive_opt(BarOpt)]
pub struct BarStyle {
    width: f32,
}
