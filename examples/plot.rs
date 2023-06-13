use std::f32::consts::PI;

use essay_plot::{prelude::*};
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

    let x = linspace(0., 100., 20);
    let y = x.clone();
    // gui.main_loop().unwrap();
    let mut figure = Figure::new();
    let axes = figure.new_graph([2., 1.]);
    //axes.pcolor();
    axes.title("My Title").style().color(0x008033);
    axes.xlabel("My X-Label").style().color(0x0030ff);
    axes.ylabel("Y-Label").style().color("r");
    //axes.scatter(&x, &y, ()).color(0x003fc0);
    axes.plot(&x, &y, ());
    
    // axes.scatter(&x, &y, ());
    // let x = tf32!([40., 30., 20., 5., 5.]);
    // let axes = figure.new_graph(());
    // axes.pie(x, ());

    // let x = linspace(0., 20., 21);
    // let axes = figure.new_graph([1., 1., 2., 2.]);
    // axes.plot(&x, &x.exp(), ());
    //axes.bezier2([0., 0.], [0.5, 1.0], [1.0, 0.0]);
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
