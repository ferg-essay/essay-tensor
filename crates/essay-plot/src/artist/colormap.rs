use essay_plot_base::Color;

pub struct ColorMap {
    colors: Vec<[f32; 4]>,
    factor: f32,
}

impl ColorMap {
    fn from_colors(colors: &[(f32, Color)]) -> Self {
        let mut colors = Vec::from(colors);

        if colors.len() == 0 {
            colors.push((0., Color(0xff)));
        };

        if colors[0].0 > 0. {
            colors.insert(0, (0., colors[0].1.clone()));
        }

        if colors[colors.len() - 1].0 < 1. {
            colors.push((1., colors[colors.len() - 1].1.clone()));
        }

        let mut full_colors = Vec::<[f32; 4]>::new();

        // TODO: add lab/grey midpoints?

        //for color in raw_colors {
        //    full_colors.push([color.red(), color.green(), color.blue(), 1.])
        //}

        for i in 0..256 {
            full_colors.push(raw_to_full(i as f32 / 255., colors.as_slice()));
        }

        let factor = (full_colors.len() as f32 - 1.).max(1.);

        let last = full_colors[full_colors.len() - 1].clone();
        full_colors.push(last);


        Self {
            colors: full_colors,
            factor,
        }
    }

    pub fn map(&self, v: f32) -> Color {
        let offset = v * self.factor;
        let i = (offset as usize).min(self.colors.len() - 2);
        let offset = offset - i as f32;

        let c0 = self.colors[i];
        let c1 = self.colors[i + 1];

        let r = (1. - offset) * c0[0] + offset * c1[0];
        let g = (1. - offset) * c0[1] + offset * c1[1];
        let b = (1. - offset) * c0[2] + offset * c1[2];
        let a = (1. - offset) * c0[3] + offset * c1[3];

        let color = Color::from((r, g, b, a));

        color

    }
}

pub fn raw_to_full(v: f32, raw_colors: &[(f32, Color)]) -> [f32; 4] {
    let i = match raw_colors.iter().position(|c| v <= c.0) {
        Some(pos) => pos.max(1) - 1,
        None => panic!(),
    };

    let (f0, c0) = raw_colors[i];
    let (f1, c1) = raw_colors[i + 1];

    // let v = v * (len - 1) as f32;
    //let i = v as usize;
    //let offset = v - i as f32;
    let offset = (v - f0) / (f1 - f0).max(f32::EPSILON);

    //if len - 1 <= i {
    //    let last = raw_colors[raw_colors.len() - 1];
        
    //    return [last.red(), last.green(), last.blue(), 1.];
    //}

    // TODO: attempting to interpolate in msh space or lab space produced odd
    // effects. Check to see if this is a bug in conversion or a problem 
    // specific to saturation to low saturation interpolation.
    //
    // Specifically, interpolating white or near-white's hue, which isn't
    // an accurate value because it's extremely desaturated, causes odd
    // colors in between because the bogus hue from white is interpolated
    // as if it's a fully saturated color.

    let c0 = raw_colors[i].1.to_xyz();
    let c1 = raw_colors[i + 1].1.to_xyz();

    //let c0 = [c0.red(), c0.green(), c0.blue()];
    //let c1 = [c1.red(), c1.green(), c1.blue()];

    let x = (1. - offset) * c0[0] + offset * c1[0];
    let y = (1. - offset) * c0[1] + offset * c1[1];
    let z = (1. - offset) * c0[2] + offset * c1[2];
    //let s0 = c0[1] / (c0[1] + c1[1]);
    //let s1 = c1[1] / (c0[1] + c1[1]);
    //let z = (1. - offset) * c0[2] * s0 + offset * c1[2] * s1;
    //[x, y, z, 1.]

    let c = Color::from_xyz(x, y, z);
    [c.red(), c.green(), c.blue(), 1.]
}

impl From<&[(f32, Color)]> for ColorMap {
    fn from(value: &[(f32, Color)]) -> Self {
        ColorMap::from_colors(value)
    }
}

impl<const N: usize> From<[(f32, Color); N]> for ColorMap {
    fn from(value: [(f32, Color); N]) -> Self {
        ColorMap::from_colors(Vec::from(value).as_slice())
    }
}

impl From<&[Color]> for ColorMap {
    fn from(value: &[Color]) -> Self {
        let mut colors = Vec::<(f32, Color)>::new();

        let factor = 1. / (value.len().max(2) - 1) as f32;
        for (i, name) in value.iter().enumerate() {
            colors.push((i as f32 * factor, Color::from(*name)));
        }

        ColorMap::from_colors(colors.as_slice())
    }
}

impl<const N: usize> From<[Color; N]> for ColorMap {
    fn from(value: [Color; N]) -> Self {
        let mut colors = Vec::<(f32, Color)>::new();

        let factor = 1. / (value.len().max(2) - 1) as f32;
        for (i, name) in value.iter().enumerate() {
            colors.push((i as f32 * factor, Color::from(*name)));
        }

        ColorMap::from_colors(colors.as_slice())
    }
}

impl From<&[(f32, &str)]> for ColorMap {
    fn from(value: &[(f32, &str)]) -> Self {
        let mut colors = Vec::<(f32, Color)>::new();

        for (v, name) in value.iter() {
            colors.push((*v, Color::from(*name)));
        }

        ColorMap::from_colors(colors.as_slice())
    }
}

impl<const N: usize> From<[(f32, &str); N]> for ColorMap {
    fn from(value: [(f32, &str); N]) -> Self {
        let mut colors = Vec::<(f32, Color)>::new();

        for (v, name) in value.iter() {
            colors.push((*v, Color::from(*name)));
        }

        ColorMap::from_colors(colors.as_slice())
    }
}

impl From<&[&str]> for ColorMap {
    fn from(value: &[&str]) -> Self {
        let mut colors = Vec::<(f32, Color)>::new();

        let factor = 1. / (value.len().max(2) - 1) as f32;
        for (i, name) in value.iter().enumerate() {
            colors.push((i as f32 * factor, Color::from(*name)));
        }

        ColorMap::from_colors(colors.as_slice())
    }
}

impl<const N: usize> From<[&str; N]> for ColorMap {
    fn from(value: [&str; N]) -> Self {
        let mut colors = Vec::<(f32, Color)>::new();

        let factor = 1. / (value.len().max(2) - 1) as f32;
        for (i, name) in value.iter().enumerate() {
            colors.push((i as f32 * factor, Color::from(*name)));
        }

        ColorMap::from_colors(colors.as_slice())
    }
}
