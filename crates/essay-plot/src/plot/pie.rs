use essay_plot_base::{Angle, Point};
use essay_tensor::Tensor;

use crate::{artist::{patch, Container, ContainerOpt}, graph::{Graph}};

pub fn pie(
    graph: &mut Graph, 
    x: impl Into<Tensor>, 
) -> ContainerOpt {
    let x = x.into();
    
    assert!(x.rank() == 1, "pie chart must have rank 1 data");

    let sum = x.reduce_sum()[0];

    let x = x / sum;

    let radius = 1.;

    let startangle = 0.;
    let mut theta1 = startangle / 360.;
    let center = Point(0., 0.);

    let mut container = Container::new();

    for frac in x.iter() {
        let theta2 = (theta1 + frac + 1.) % 1.;

        let patch = patch::Wedge::new(
            center, 
            radius, 
            (Angle::Unit(theta1), Angle::Unit(theta2))
        );

        container.push(patch);

        theta1 = theta2;
    }

    graph.add_plot_artist(container)
}
