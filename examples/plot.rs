use essay_plot::{prelude::*, artist::{Bezier3, Bezier2, PColor, patch::PathPatch, Markers}, graph::{Graph, PlotOpt}, plot::bar_y};
use essay_plot_base::{Point, Color, PathCode, Path, JoinStyle, CapStyle, LineStyle};
use essay_tensor::{prelude::*, init::{linspace, meshgrid}};

fn main() {
    //let mut gui = WgpuBackend::new();


    //let x = linspace(0., 2. * PI, 30);
    //let y = x.sin();

    let x = linspace(-1., 1., 5);
    let y = linspace(-1., 1., 5);
    let [x, y] = meshgrid([x, y]);

    let z = 1. - &x * &x + 1. - &y * &y;
    println!("X: {:?}", x);
    println!("Y: {:?}", y);
    println!("z: {:?}", z);
    //let y = x.sin();

    let x = linspace(0., 1., 3);
    let y = x.clone();
    // gui.main_loop().unwrap();
    let mut figure = Figure::new();
    let graph = figure.new_graph([1., 1.]);
    //axes.pcolor();

    // axes.title("My Title").style().color(0x008033);
    // axes.xlabel("My X-Label").style().color(0x0030ff);
    // axes.ylabel("Y-Label").style().color("r");

    graph.title("My Title").color(0x008033).size(18.);
    graph.xlabel("My X-Label").color("brown");
    graph.ylabel("Y-Label").color("teal").size(8.);
/*
    graph.scatter(&x, &y).color("blue").marker("X")
        .line_color(0xff8000)
        .size(2500.)
        .fill_color("none")
        .line_width(5.);
    */
    let x = linspace(0., 6.28, 100);
    let y = x.sin();
    graph.plot(&x, &y).color(0xc08000).line_width(4.)
        .line_style("--");
    graph.x().show_grid(true);
    graph.x().major_grid().color(0xc04040).line_width(1.5);
    graph.x().major().color(0xc04040).line_width(1.5);
    graph.y().show_grid(true);
    //bar_y(graph, &y)
    //    .edgecolor(0x400080)
    //    .facecolor(0x80c0e0)
    //    .width(0.2);
    
    //axes.scatter(&x, &y, ());
    //let x = tf32!([40., 30., 20., 5., 5.]);
    //let x = tf32!([40., 30.]);
    let x = tf32!([25., 25., 50.]);
    // let axes = figure.new_graph(());
     //graph.pie(x);
    // let x = linspace(0., 20., 21);
    // let axes = figure.new_graph([1., 1., 2., 2.]);
    // axes.plot(&x, &x.exp(), ());
    //bezier2(graph, [-0.5, 0.], [-1.0, 1.0], [-1.5, 0.0]).color(Color(0x0080c080));
    
    /*
    plot_quad(graph, [0.0, 0.0], [1.0, 0.0], [1., 1.], [0., 1.])
        .facecolor(Color(0))
        .edgecolor(0xe08000)
        .linewidth(20.)
        .joinstyle(JoinStyle::Bevel);
    */
    
    /*
    plot_line(graph, 
        [0.0, 0.0], [1.0, 0.0],
        [1., 1.], [0., 1.],
        [0.5, 0.25], [0.5, 0.75],
    ).facecolor(Color(0))
        .edgecolor(0xe08000)
        .linewidth(20.)
        .joinstyle(JoinStyle::Miter)
        .capstyle(CapStyle::Round);
    */
    
    //bezier2(graph, [0.5, 0.], [1.0, 1.0], [1.5, 0.0]).color(Color(0x0080c080));
    //bezier2(graph, [-1.5, 0.], [-1.0, -1.0], [-0.5, 0.0]).color(Color(0x0080c080));
    //bezier2(graph, [0., -0.5], [1.0, -1.0], [0.0, -1.5]).color(Color(0x0080c080));

    //axes.bezier2([-1., 0.], [0.5, 1.0], [1.0, 0.0]);
    // axes.bezier2([0., -1.], [-0.5, 0.0], [0.0, 1.]);
    //axes.bezier3([0., 0.], [0.25, 1.0], [0.5, -1.0], [1.0, 0.0]);
    //axes.bezier3([-1., 0.], [1.0, 1.0], [-1.0, -1.0], [1.0, 0.0]);
    figure.show();

    /*
    let mut figure = Figure::new();

    let x = tf32!([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
    let y = &x * &x + tf32!(1.);

    figure.plot(&x, &y, ()); // , ().label("My Item"));
    //plot.plot(&x, &y, ().label("My Item"));
    //plot.scatter(&x, &x * &x * &x, ().label("My Item 3"));
    //plot.set_title("My Title");
    //plot.set_xlabel("My x-axis");

    figure.show();
    */
}

pub fn bezier3(
    graph: &mut Graph, 
    p0: impl Into<Point>,
    p1: impl Into<Point>,
    p2: impl Into<Point>,
    p3: impl Into<Point>
) {
    graph.add_data_artist(Bezier3(p0.into(), p1.into(), p2.into(), p3.into()));
}

pub fn bezier2(
    graph: &mut Graph, 
    p0: impl Into<Point>,
    p1: impl Into<Point>,
    p2: impl Into<Point>,
) -> PlotOpt {
    graph.add_data_artist(Bezier2(p0.into(), p1.into(), p2.into()))
}

pub fn plot_quad(
    graph: &mut Graph, 
    p0: impl Into<Point>,
    p1: impl Into<Point>,
    p2: impl Into<Point>,
    p3: impl Into<Point>,
) -> PlotOpt {
    graph.add_data_artist(PathPatch::new(Path::new(vec![
        PathCode::MoveTo(p0.into()),
        PathCode::LineTo(p1.into()),
        PathCode::LineTo(p2.into()),
        PathCode::ClosePoly(p3.into()),
    ])))
}

pub fn plot_line(
    graph: &mut Graph, 
    p0: impl Into<Point>,
    p1: impl Into<Point>,
    p2: impl Into<Point>,
    p3: impl Into<Point>,
    p4: impl Into<Point>,
    p5: impl Into<Point>,
) -> PlotOpt {
    graph.add_data_artist(PathPatch::new(Path::new(vec![
        PathCode::MoveTo(p0.into()),
        PathCode::LineTo(p1.into()),
        PathCode::MoveTo(p2.into()),
        PathCode::LineTo(p3.into()),
        PathCode::MoveTo(p4.into()),
        PathCode::LineTo(p5.into()),
    ])))
}

pub fn pcolor(
    graph: &mut Graph, 
) {
    graph.add_data_artist(PColor {});
}
