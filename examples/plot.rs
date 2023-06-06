use essay_plot::{prelude::*, driver::{egui::EguiBackend, Backend, wgpu::WgpuBackend}};
use essay_tensor::{prelude::*, init::linspace};

fn main() {
    println!("Hello");

    //let mut gui = WgpuBackend::new();


    let x = linspace(0., 10., 3);
    let y = x.clone();

    // gui.main_loop().unwrap();
    let mut figure = Figure::new();
    let axes = figure.axes(());
    // axes.plot(&x, &y, ());
    // axes.scatter(&x, &y, ());
    let x = tf32!([40., 30., 20., 5., 5.]);
    axes.pie(x, ());
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
