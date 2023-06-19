use core::fmt;
use std::str::FromStr;

#[derive(Clone, Copy, PartialEq)]
pub struct Color(pub u32);

impl fmt::Debug for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Color({:8x})", self.0)
    }
}

impl Color {
    #[inline]
    pub fn black() -> Color {
        Color(0x000000ff)
    }

    #[inline]
    pub fn white() -> Color {
        Color(0xffffffff)
    }

    #[inline]
    pub fn none() -> Color {
        Color(0x0)
    }
    
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

    pub fn from_hsv(h: f32, s: f32, v: f32) -> Color {
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

    #[inline]
    pub fn get_rgba(&self) -> u32 {
        self.0
    }

    #[inline]
    pub fn to_srgb(color: f32) -> f32 {
        ((color + 0.055) / 1.055).powf(2.4)
    }

    #[inline]
    pub fn is_none(&self) -> bool {
        self.0 == 0
    }
}

impl From<u32> for Color {
    fn from(value: u32) -> Self {
        Color((value & 0xffffff) * 256 + 0xff)
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

impl From<&str> for Color {
    fn from(name: &str) -> Self {
        name.parse::<Color>().unwrap()

        /*
        if let Some(color) = super::color_data::lookup_color(name) {
            return color;
        }

        panic!("'{}' is an unknown color name", name);
        // TODO: parse '#', 'rgb(...)'

        //return Color::black();
        */
    }
}

impl FromStr for Color {
    type Err = ColorErr;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        if let Some(color) = super::color_data::lookup_color(name) {
            return Ok(color);
        }

        if name.starts_with("#") {
            let mut value = 0;
            for ch in name.as_bytes().iter().skip(1) {
                match ch {
                    b'0'..=b'9' => { value = 16 * value + *ch as u32; }
                    b'a'..=b'f' => { value = 16 * value + *ch as u32 - 'a' as u32 + 10; }
                    b'A'..=b'F' => { value = 16 * value + *ch as u32 - 'A' as u32 + 10; }
                    _ => {
                        return Err(ColorErr(format!("Invalid rgb color spec {:?}", name)));
                    }
                }
            }

            return Ok(Color::from(value))
        }

        return Err(ColorErr(format!("'{}' is an unknown color name", name)));
    }
}

#[derive(Clone, Debug)]
pub struct ColorErr(String);

