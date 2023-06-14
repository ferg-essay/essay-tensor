use essay_plot_base::{Affine2d, Path};
use essay_tensor::Tensor;

use crate::{artist::{Lines2d, Collection, Container, paths, patch::{DataPatch, PathPatch}, Artist}, graph::{Graph, PlotOpt}, frame::{Data, ArtistId}};

pub fn bar_y(
    graph: &mut Graph, 
    y: impl Into<Tensor>, 
) -> PlotOpt {
    let y : Tensor = y.into();

    let mut container = Container::new();

    for (i, value) in y.iter().enumerate() {
        let scale = Affine2d::eye()
            .scale(0.8, *value)
            .translate(i as f32 - 0.4, 0.);
        let path: Path<Data> = paths::unit().transform(&scale);
        
        let id = ArtistId::new(0);
        container.push(Artist::new(id, PathPatch::<Data>::new(path)));
    }

    //self.artist(lines)
    graph.add_data_artist(container)
}

