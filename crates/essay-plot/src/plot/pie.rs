use essay_plot_base::{Angle, Point};
use essay_tensor::Tensor;

use crate::{artist::{patch, ColorCycle, Container, ArtistStyle}, graph::{Graph, PlotOpt}, frame::ArtistId};

pub fn pie(
    graph: &mut Graph, 
    x: impl Into<Tensor>, 
) -> PlotOpt {
    let x = x.into();
    
    assert!(x.rank() == 1, "pie chart must have rank 1 data");

    let sum = x.reduce_sum()[0];

    let x = x / sum;

    let radius = 1.;

    let startangle = 0.;
    let mut theta1 = startangle / 360.;
    let center = Point(0., 0.);

    let mut container = Container::new();
    let colors = ColorCycle::tableau();
    let mut i = 0;
    for frac in x.iter() {
        let theta2 = (theta1 + frac + 1.) % 1.;

        let patch = patch::Wedge::new(
            center, 
            radius, 
            (Angle::Unit(theta1), Angle::Unit(theta2))
        );

        //patch.color(colors[i]);
        let id = ArtistId::empty();

        let mut artist = ArtistStyle::new(id, patch);
        //artist.color(colors[i]);
        artist.style_mut().face_color(colors[i]);
        //artist.style_mut().facecolor(Color(0x0));
        
        container.push(artist);

        theta1 = theta2;
        i += 1;
    }

    graph.add_data_artist(container)

    // todo!()
}
