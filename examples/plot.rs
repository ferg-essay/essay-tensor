use essay_plot::{prelude::*};
use essay_tensor::{prelude::*, init::linspace};

fn main() {
    //let mut gui = WgpuBackend::new();


    let x = linspace(0., 10., 10);
    let y = x.sin();

    // gui.main_loop().unwrap();
    let mut figure = Figure::new();
    let axes = figure.new_graph(());
    axes.title("My Title").style().color(0x008033);
    axes.xlabel("X-Label").style().color(0x0030ff);
    axes.ylabel("Y-Label").style().color("r");
    axes.plot(&x, &y, ()).color(0x003fc0);
    // axes.scatter(&x, &y, ());
    let x = tf32!([40., 30., 20., 5., 5.]);
    //axes.pie(x, ());
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
