use core::fmt;
use std::ops::Index;

#[derive(Clone, Copy, PartialEq)]
pub struct Color(pub u32);

impl fmt::Debug for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Color({:8x})", self.0)
    }
}

impl Color {
    #[inline]
    pub fn from_rgb(red: f32, green: f32, blue: f32) -> Color {
        Color::from_rgba(red, green, blue, 1.0)
    }

    #[inline]
    pub fn from_rgba(red: f32, green: f32, blue: f32, alpha: f32) -> Color {
        let r = (red * 255.).clamp(0., 255.) as u32;
        let g = (green * 255.).clamp(0., 255.) as u32;
        let b = (blue * 255.).clamp(0., 255.) as u32;
        let a = (alpha * 255.).clamp(0., 255.) as u32;

        Color((r << 24) | (g << 16) | (b << 8) | a)
    }

    pub fn from_hsv(h: f32, s: f32, v: f32, a: f32) -> Color {
        Self::from_hsva(h, s, v, 1.)
    }

    pub fn from_hsva(h: f32, s: f32, v: f32, a: f32) -> Color {
        let h = h.clamp(0., 1.);
        let s = s.clamp(0., 1.);
        let v = v.clamp(0., 1.);
        let a = a.clamp(0., 1.);

        let i = (h / 6.) as u32;
        let ff = h / 6. - i as f32;
        let p = v * (1. - s);
        let q = v * (1. - (s * ff));
        let t = v * (1. - (s * (1. - ff)));

        let (r, g, b) = match i {
            0 => (v, t, p),
            1 => (q, v, p),
            2 => (p, v, t),
            3 => (p, q, v),
            4 => (t, p, v),
            _ => (v, p, q),
        };

        Self::from_rgba(r, g, b, a)
    }

    #[inline]
    pub fn red(&self) -> f32 {
        ((self.0 >> 24) & 0xff) as f32 / 255.
    }

    #[inline]
    pub fn green(&self) -> f32 {
        ((self.0 >> 16) & 0xff) as f32 / 255.
    }

    #[inline]
    pub fn blue(&self) -> f32 {
        ((self.0 >> 8) & 0xff) as f32 / 255.
    }

    #[inline]
    pub fn alpha(&self) -> f32 {
        (self.0 & 0xff) as f32 / 255.
    }
}

impl From<u32> for Color {
    fn from(value: u32) -> Self {
        Color(value)
    }
}

impl From<(f32, f32, f32)> for Color {
    fn from(value: (f32, f32, f32)) -> Self {
        Color::from_rgb(value.0, value.1, value.2)
    }
}

impl From<(f32, f32, f32, f32)> for Color {
    fn from(value: (f32, f32, f32, f32)) -> Self {
        Color::from_rgba(value.0, value.1, value.2, value.3)
    }
}

pub struct ColorCycle {
    colors: Vec<(String, Color)>,
}

impl ColorCycle {
    pub fn new(colors: &Vec<(&str, Color)>) -> Self {
        let mut vec = Vec::<(String, Color)>::new();

        for (name, color) in colors {
            vec.push((name.to_string(), *color));
        }

        Self {
            colors: vec
        }
    }
}

impl Index<usize> for ColorCycle {
    type Output = Color;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let len = self.colors.len();

        &self.colors[index % len].1
    }
}

impl ColorCycle {
    fn tableau() -> ColorCycle {
        Self::new(&vec![
            ("tab:blue", Color(0x1f77b4ff)),
            ("tab:orange", Color(0xff7f0eff)),
            ("tab:green", Color(0x2ca02cff)),
            ("tab:red", Color(0xd62728ff)),
            ("tab:purple", Color(0x9467bdff)),
            ("tab:brown", Color(0x8c564bff)),
            ("tab:pink", Color(0xe377c2ff)),
            ("tab:gray", Color(0x7f7f7fff)),
            ("tab:olive", Color(0xbcbd22ff)),
            ("tab:cyan", Color(0xf17becff)),
        ])
    }
}

impl Default for ColorCycle {
    fn default() -> Self {
        ColorCycle::tableau()
    }
}

// const TABLEAU : ColorCycle = ColorCycle::tableau();