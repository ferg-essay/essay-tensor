use essay_plot_base::Color;

pub struct ColorMap {
    colors: Vec<[f32; 4]>,
    factor: f32,
}

impl ColorMap {
    fn from_colors(colors: &[Color]) -> Self {
        let colors = if colors.len() > 0 {
            colors
        } else {
            &[Color(0xff)]
        };


        let mut full_colors = Vec::<[f32; 4]>::new();
        //for color in raw_colors {
        //    full_colors.push([color.red(), color.green(), color.blue(), 1.])
        //}
        for i in 0..256 {
            full_colors.push(raw_to_full(i as f32 / 255., colors));
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

pub fn raw_to_full(v: f32, raw_colors: &[Color]) -> [f32; 4] {
    let len = raw_colors.len();

    let v = v * (len - 1) as f32;
    let i = v as usize;
    let offset = v - i as f32;

    if len - 1 <= i {
        let last = raw_colors[raw_colors.len() - 1];
        
        return [last.red(), last.green(), last.blue(), 1.];
    }

    // TODO: attempting to interpolate in msh space or lab space produced odd
    // effects. Check to see if this is a bug in conversion or a problem 
    // specific to saturation to low saturation interpolation.
    //let c0 = raw_colors[i].to_lab();
    //let c1 = raw_colors[i + 1].to_lab();

    let c0 = [raw_colors[i].red(), raw_colors[i].green(), raw_colors[i].blue()];
    let c1 = [raw_colors[i + 1].red(), raw_colors[i + 1].green(), raw_colors[i + 1].blue()];

    let x = (1. - offset) * c0[0] + offset * c1[0];
    let y = (1. - offset) * c0[1] + offset * c1[1];
    let z = (1. - offset) * c0[2] + offset * c1[2];
    //let s0 = c0[1] / (c0[1] + c1[1]);
    //let s1 = c1[1] / (c0[1] + c1[1]);
    //let z = (1. - offset) * c0[2] * s0 + offset * c1[2] * s1;

    //let c = Color::from_lab(x, y, z);
    //[c.red(), c.green(), c.blue(), 1.]

    [x, y, z, 1.]
}

impl From<&[Color]> for ColorMap {
    fn from(value: &[Color]) -> Self {
        ColorMap::from_colors(value)
    }
}

impl<const N: usize> From<[Color; N]> for ColorMap {
    fn from(value: [Color; N]) -> Self {
        ColorMap::from_colors(Vec::from(value).as_slice())
    }
}

impl From<&[&str]> for ColorMap {
    fn from(value: &[&str]) -> Self {
        let mut colors = Vec::<Color>::new();

        for name in value {
            colors.push(Color::from(*name));
        }

        ColorMap::from_colors(colors.as_slice())
    }
}

impl<const N: usize> From<[&str; N]> for ColorMap {
    fn from(value: [&str; N]) -> Self {
        let mut colors = Vec::<Color>::new();

        for name in value {
            colors.push(Color::from(name));
        }

        ColorMap::from_colors(colors.as_slice())
    }
}
