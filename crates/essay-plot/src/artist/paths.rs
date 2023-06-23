use std::f32::consts::{PI, TAU};

use essay_plot_base::{Angle, Coord, Path, PathCode, Point, Canvas, Bounds};
use essay_tensor::{init::linspace, tensor::TensorVec};

///
/// Unit coordinates centered around 0: [-1, 1] x [-1, 1]
/// 

pub struct Unit;

impl Coord for Unit {}

pub fn square() -> Path<Unit> {
    Path::new(vec![
        PathCode::MoveTo(Point(-1., -1.)),
        PathCode::LineTo(Point(1., -1.)),
        PathCode::LineTo(Point(1., 1.)),
        PathCode::ClosePoly(Point(-1., 1.)),
    ])
}

pub fn unit_pos() -> Path<Unit> {
    Path::new(vec![
        PathCode::MoveTo(Point(0., 0.)),
        PathCode::LineTo(Point(1., 0.)),
        PathCode::LineTo(Point(1., 1.)),
        PathCode::ClosePoly(Point(0., 1.)),
    ])
}

pub fn unit_polygon(n: usize) -> Path<Unit> {
    assert!(n > 2);

    let mut tensor = TensorVec::<f32>::new();

    for i in 0..n {
        let theta = (i as f32 * 360. / n as f32 + 90.).to_radians();
        
        tensor.push(theta.cos());
        tensor.push(theta.sin());
    }
    let tensor = tensor.into_tensor().reshape([n, 2]);

    Path::closed_poly(tensor)
}

pub fn unit_polygon_alt(n: usize) -> Path<Unit> {
    assert!(n > 2);

    let mut tensor = TensorVec::<f32>::new();

    for i in 0..n {
        let theta = (i as f32 * 360. / n as f32 + 180. / n as f32).to_radians();
        
        tensor.push(theta.cos());
        tensor.push(theta.sin());
    }
    let tensor = tensor.into_tensor().reshape([n, 2]);
    
    Path::closed_poly(tensor)
}

pub fn unit_star(n: usize, r_inner: f32) -> Path<Unit> {
    let mut tensor = TensorVec::<f32>::new();

    for i in 0..n {
        let theta = (i as f32 * 360. / n as f32 + 90.).to_radians();
        
        tensor.push(theta.cos());
        tensor.push(theta.sin());

        let theta2 = theta + PI / n as f32;

        tensor.push(theta2.cos() * r_inner);
        tensor.push(theta2.sin() * r_inner);
    }
    let tensor = tensor.into_tensor().reshape([2 * n, 2]);

    Path::closed_poly(tensor)
}

pub fn unit_asterisk(n: usize) -> Path<Unit> {
    let mut codes = Vec::<PathCode>::new();

    for i in 0..n {
        let theta = (i as f32 * 360. / n as f32 + 90.).to_radians();
        
        codes.push(PathCode::MoveTo(Point(0., 0.)));
        codes.push(PathCode::LineTo(Point(theta.cos(), theta.sin())));
    }

    Path::new(codes)
}

pub fn wedge(angle: (Angle, Angle)) -> Path<Unit> {
    let halfpi = 0.5 * PI;

    let (t0, t1) = (angle.0.to_radians(), angle.1.to_radians());
    
    let t1 = if t0 < t1 { t1 } else { t1 + TAU };

    // TODO:
    let n = 2.0f32.powf(((t1 - t0) / halfpi).ceil()) as usize;

    let steps = linspace(t0, t1, n + 1);

    let cos = steps.cos();
    let sin = steps.sin();

    let dt = (t1 - t0) / n as f32;
    let t = (0.5 * dt).tan();
    let alpha = dt.sin() * ((4. + 3. * t * t).sqrt() - 1.) / 3.;
    // let mut vec = TensorVec::<[f32; 2]>::new();
    let mut codes = Vec::new();

    codes.push(PathCode::MoveTo(Point(cos[0], sin[0])));

    for i in 1..=n {
        // TODO: switch to quad bezier
        // vec.push([cos[i], sin[i]]);
        codes.push(PathCode::Bezier3(
            Point(
                cos[i - 1] - alpha * sin[i - 1],
                sin[i - 1] + alpha * cos[i - 1],
            ),
            Point(cos[i] + alpha * sin[i], sin[i] - alpha * cos[i]),
            Point(cos[i], sin[i]),
        ));
    }

    // vec.push([0., 0.]);
    codes.push(PathCode::ClosePoly(Point(0., 0.)));

    //let path = Path::new(vec.into_tensor(), codes);
    let path = Path::new(codes);

    path
}

// Via matplotlib
// Lancaster, Don.  `Approximating a Circle or an Ellipse Using Four
// Bezier Cubic Splines <https://www.tinaja.com/glib/ellipse4.pdf>`_.
pub fn circle() -> Path<Unit> {
    let magic = 0.2652031;
    let sqrt_half = 0.5f32.sqrt();
    let magic_45 = sqrt_half * magic;

    Path::from([
        PathCode::MoveTo(Point(0., -1.)),
        PathCode::Bezier3(
            Point(magic, -1.),
            Point(sqrt_half - magic_45, -sqrt_half - magic_45),
            Point(sqrt_half, -sqrt_half),
        ),
        PathCode::Bezier3(
            Point(sqrt_half + magic_45, -sqrt_half + magic_45),
            Point(1., -magic),
            Point(1., 0.),
        ),
        PathCode::Bezier3(
            Point(1.0, magic),
            Point(sqrt_half + magic_45, sqrt_half - magic_45),
            Point(sqrt_half, sqrt_half),
        ),
        PathCode::Bezier3(
            Point(sqrt_half - magic_45, sqrt_half + magic_45),
            Point(magic, 1.),
            Point(0., 1.),
        ),
        PathCode::Bezier3(
            Point(-magic, 1.0),
            Point(-sqrt_half + magic_45, sqrt_half + magic_45),
            Point(-sqrt_half, sqrt_half),
        ),
        PathCode::Bezier3(
            Point(-sqrt_half - magic_45, sqrt_half - magic_45),
            Point(-1.0, magic),
            Point(-1., 0.),
        ),
        PathCode::Bezier3(
            Point(-1., -magic),
            Point(-sqrt_half - magic_45, -sqrt_half + magic_45),
            Point(-sqrt_half, -sqrt_half),
        ),
        PathCode::Bezier3(
            Point(-sqrt_half + magic_45, -sqrt_half - magic_45),
            Point(-magic, -1.0),
            Point(0., -1.),
        ),
        PathCode::ClosePoly(Point(0., -1.)),
    ])
}

pub(crate) fn bounds(pos: &Bounds<Canvas>) -> Path<Canvas> {
    Path::new(vec![
        PathCode::MoveTo(Point(pos.xmin(), pos.ymin())),
        PathCode::LineTo(Point(pos.xmax(), pos.ymin())),
        PathCode::LineTo(Point(pos.xmax(), pos.ymax())),
        PathCode::ClosePoly(Point(pos.xmin(), pos.ymax())),
    ])
}
