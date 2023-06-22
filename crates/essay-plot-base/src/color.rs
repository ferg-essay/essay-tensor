use core::fmt;
use std::str::FromStr;

use crate::color_data::lookup_color_name;

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
    pub fn to_rgba(&self) -> u32 {
        self.0
    }

    #[inline]
    pub fn srgb_to_lrgb(srgb: f32) -> f32 {
        if srgb > 0.04045 {
            ((srgb + 0.055) / 1.055).powf(2.4)
        } else {
            srgb / 12.92
        }
    }

    #[inline]
    pub fn lrgb_to_srgb(lrgb: f32) -> f32 {
        if lrgb > 0.0031308 {
            1.055 * lrgb.powf(1./2.4) - 0.055
        } else {
            lrgb * 12.92
        }
    }

    #[inline]
    pub fn to_lrgb(&self) -> [f32; 4] {
        [
            Self::srgb_to_lrgb(self.red()),
            Self::srgb_to_lrgb(self.green()),
            Self::srgb_to_lrgb(self.blue()),
            self.alpha(),
        ]
    }

    #[inline]
    pub fn from_lrgb(&self) -> [f32; 4] {
        [
            Self::srgb_to_lrgb(self.red()),
            Self::srgb_to_lrgb(self.green()),
            Self::srgb_to_lrgb(self.blue()),
            1.
        ]
    }

    /// CIE xyz
    #[inline]
    pub fn to_xyz(&self) -> [f32; 3] {
        let (r, g, b) = (self.red(), self.green(), self.blue());

        [
            0.412453 * r + 0.357580 * g + 0.180423 * b,
            0.212671 * r + 0.715160 * g + 0.072169 * b,
            0.019334 * r + 0.119193 * g + 0.950227 * b,
        ]
    }

    /// CIE xyz
    #[inline]
    pub fn from_xyz(x: f32, y: f32, z: f32) -> Color {
        Color::from_rgb(
            3.240479 * x - 1.537150 * y - 0.498535 * z,
            -0.969256 * x + 1.875992 * y + 0.041556 * z,
            0.055648 * x - 0.204043 * y + 1.057311 * z,
        )
    }

    /// CIE lab
    #[inline]
    pub fn to_lab(&self) -> [f32; 3] {
        let [x, y, z] = self.to_xyz();
        // D50
        let [xn, yn, zn] = Self::D65;

        fn f(x: f32) -> f32 {
            if x > 0.008856 {
                x.powf(1. / 3.)
            } else {
                7.787 * x + 16. / 116.
            }
        }

        [
            116. * f(y / yn) - 16.,
            500. * (f(x / xn) - f(y / yn)),
            200. * (f(y / yn) - f(z / zn)),
        ]
    }

    pub const D50 : [f32; 3] = [0.964212, 1.0, 0.825188];
    //pub const D50 : [f32; 3] = [96.4212, 100., 82.5188];
    pub const D65 : [f32; 3] = [0.950489, 1.0, 1.088840];

    #[inline]
    pub fn from_lab(l: f32, a: f32, b: f32) -> Color {
        // D65
        let [xn, yn, zn] = Self::D65;

        fn f(x: f32) -> f32 {
            let x3 = x * x * x;

            if x3 > 0.008856 {
                x3
            } else {
                (116. * x - 16.) / 903.3
            }
        }

        let f_y = (l + 16.) / 116.;

        Color::from_xyz(
            xn * f(f_y + a / 500.),
            yn * f(f_y),
            zn * f(f_y - b / 200.),
        )
    }

    /// Morland MSH in Diverging Color Maps for Scientific Visualization
    #[inline]
    pub fn to_msh(&self) -> [f32; 3] {
        let [l, a, b] = self.to_lab();

        lab_to_msh(l, a, b)
    }

    #[inline]
    pub fn from_msh(m: f32, s: f32, h: f32) -> Color {
        let [l, a, b] = msh_to_lab(m, s, h);

        Self::from_lab(l, a, b)
    }

    #[inline]
    pub fn is_none(&self) -> bool {
        self.0 == 0
    }

    pub fn closest_name(&self) -> String {
        lookup_color_name(self)
    }
}

// msh in Morland, Diverging Color Maps for Scientific Visualization
fn lab_to_msh(l: f32, a: f32, b: f32) -> [f32; 3] {
    let m = (l * l + a * a + b * b).sqrt();
    let a = if a != 0. { a } else { f32::EPSILON };
    [
        m,
        (l / m.max(f32::EPSILON)).acos(),
        b.atan2(a)
    ]
}

fn msh_to_lab(m: f32, s: f32, h: f32) -> [f32; 3] {
    [
        m * s.cos(),
        m * s.sin() * h.cos(),
        m * s.sin() * h.sin(),
    ]
}


impl From<u32> for Color {
    fn from(value: u32) -> Self {
        Color((value & 0xffffff) * 256 + 0xff)
    }
}

impl From<(u32, u32, u32)> for Color {
    fn from(rgb: (u32, u32, u32)) -> Self {
        Color(
            (rgb.0 & 0xff) << 24
            | (rgb.1 & 0xff) << 16
            | (rgb.2 & 0xff) << 8
            | 0xff
        )
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
                    b'0'..=b'9' => { value = 16 * value + *ch as u32 - b'0' as u32; }
                    b'a'..=b'f' => { value = 16 * value + *ch as u32 - b'a' as u32 + 10; }
                    b'A'..=b'F' => { value = 16 * value + *ch as u32 - b'A' as u32 + 10; }
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

#[cfg(test)]
mod test {
    use crate::Color;

    /// CIE xyz color
    #[test]
    fn test_xyz() {
        assert_eq!(Color(0x0000_00ff).to_xyz(), [0., 0., 0.]);
        assert_eq!(Color(0xffff_ffff).to_xyz(), [0.950456, 1.0, 1.088754]);

        assert_eq!(Color(0xff00_00ff).to_xyz(), [0.412453, 0.212671, 0.019334]);
        assert_eq!(Color(0x00ff_00ff).to_xyz(), [0.35758, 0.71516, 0.119193]);
        assert_eq!(Color(0x0000_ffff).to_xyz(), [0.180423, 0.072169, 0.950227]);

        assert_eq!(Color::from_xyz(0., 0., 0.), Color(0x0000_00ff));
        assert_eq!(Color::from_xyz(1., 0., 0.), Color(0xff00_0eff));
        assert_eq!(Color::from_xyz(0., 1., 0.), Color(0x00ff_00ff));
        assert_eq!(Color::from_xyz(0., 0., 1.), Color(0x000a_ffff));
        assert_eq!(Color::from_xyz(1., 1., 1.), Color(0xfff1_e7ff));

        assert_eq!(Color::from_xyz(0.1, 0.0, 0.0), Color(0x5200_01ff));
        assert_eq!(Color::from_xyz(0.0, 0.1, 0.0), Color(0x002f_00ff));
        assert_eq!(Color::from_xyz(0.0, 0.0, 0.1), Color(0x0001_1aff));
        assert_eq!(Color::from_xyz(0.1, 0.1, 0.1), Color(0x1e18_17ff));

        // full colors
        let [x, y, z] = Color(0xff00_00ff).to_xyz();
        assert_eq!(Color::from_xyz(x, y, z), Color(0xfe00_00ff));

        let [x, y, z] = Color(0x00ff_00ff).to_xyz();
        assert_eq!(Color::from_xyz(x, y, z), Color(0x00ff_00ff));

        let [x, y, z] = Color(0x0000_ffff).to_xyz();
        assert_eq!(Color::from_xyz(x, y, z), Color(0x0000_ffff));

        let [x, y, z] = Color(0xffff_ffff).to_xyz();
        assert_eq!(Color::from_xyz(x, y, z), Color(0xffff_feff));

        // low colors
        let [x, y, z] = Color(0x0400_00ff).to_xyz();
        assert_eq!(Color::from_xyz(x, y, z), Color(0x0300_00ff));

        let [x, y, z] = Color(0x0004_00ff).to_xyz();
        assert_eq!(Color::from_xyz(x, y, z), Color(0x0004_00ff));

        let [x, y, z] = Color(0x0000_04ff).to_xyz();
        assert_eq!(Color::from_xyz(x, y, z), Color(0x0000_04ff));

        let [x, y, z] = Color(0x0404_04ff).to_xyz();
        assert_eq!(Color::from_xyz(x, y, z), Color(0x0404_03ff));
    }

    /// CIE lab color
    #[test]
    fn test_lab() {
        assert_eq!(Color(0x0000_00ff).to_lab(), [0., 0., 0.]);
        assert_eq!(Color(0xffff_ffff).to_lab(), [100.0, -0.0057816505, 0.0052571297]);

        assert_eq!(Color(0xff00_00ff).to_lab(), [53.240585, 80.089806, 67.20291]);
        assert_eq!(Color(0x00ff_00ff).to_lab(), [87.7351, -86.185425, 83.18]);
        assert_eq!(Color(0x0000_ffff).to_lab(), [32.29567, 79.183685, -107.85673]);

        assert_eq!(Color::from_lab(53.24, 80.09, 67.20), Color(0xfe00_00ff));

        // full colors
        let [l, a, b] = Color(0xff00_00ff).to_lab();
        assert_eq!(Color::from_lab(l, a, b), Color(0xfe00_00ff));

        let [l, a, b] = Color(0x00ff_00ff).to_lab();
        assert_eq!(Color::from_lab(l, a, b), Color(0x00ff_00ff));

        let [l, a, b] = Color(0x0000_ffff).to_lab();
        assert_eq!(Color::from_lab(l, a, b), Color(0x0000_ffff));

        let [l, a, b] = Color(0xffff_ffff).to_lab();
        assert_eq!(Color::from_lab(l, a, b), Color(0xffff_feff));

        // low colors
        let [l, a, b] = Color(0x0400_00ff).to_lab();
        assert_eq!(Color::from_lab(l, a, b), Color(0x0300_00ff));

        let [l, a, b] = Color(0x0004_00ff).to_lab();
        assert_eq!(Color::from_lab(l, a, b), Color(0x0004_00ff));

        let [l, a, b] = Color(0x0000_04ff).to_lab();
        assert_eq!(Color::from_lab(l, a, b), Color(0x0000_04ff));

        let [l, a, b] = Color(0x0404_04ff).to_lab();
        assert_eq!(Color::from_lab(l, a, b), Color(0x0404_03ff));
    }

    /// MSH 
    #[test]
    fn test_msh() {
        assert_eq!(Color(0x0000_00ff).to_msh(), [0., 1.5707964, 0.]);
        //assert_eq!(Color(0xffff_ffff).to_msh(), [100.0, 0.0, -0.73791766]);

        //assert_eq!(Color(0xff00_00ff).to_lab(), [53.240585, 80.089806, 67.20291]);
        //assert_eq!(Color(0x00ff_00ff).to_msh(), [148.47319, 0.9386032, -0.7676548]);
        //assert_eq!(Color(0x0000_ffff).to_lab(), [32.29567, 79.183685, -107.85673]);

        //assert_eq!(Color::from_lab(53.24, 80.09, 67.20), Color(0xfe00_00ff));

        // full colors
        let [m, s, h] = Color(0xff00_00ff).to_msh();
        assert_eq!(Color::from_msh(m, s, h), Color(0xff00_00ff));

        let [m, s, h] = Color(0x00ff_00ff).to_msh();
        assert_eq!(Color::from_msh(m, s, h), Color(0x00ff_00ff));

        let [m, s, h] = Color(0x0000_ffff).to_msh();
        assert_eq!(Color::from_msh(m, s, h), Color(0x0000_ffff));

        let [m, s, h] = Color(0xffff_ffff).to_msh();
        assert_eq!(Color::from_msh(m, s, h), Color(0xfffe_ffff));

        // low colors
        let [m, s, h] = Color(0x0400_00ff).to_msh();
        assert_eq!(Color::from_msh(m, s, h), Color(0x0300_00ff));

        let [m, s, h] = Color(0x0004_00ff).to_msh();
        assert_eq!(Color::from_msh(m, s, h), Color(0x0004_00ff));

        let [m, s, h] = Color(0x0000_04ff).to_msh();
        assert_eq!(Color::from_msh(m, s, h), Color(0x0000_03ff));

        let [m, s, h] = Color(0x0404_04ff).to_msh();
        assert_eq!(Color::from_msh(m, s, h), Color(0x0403_04ff));

        // unsat colors
        let [m, s, h] = Color(0xfcff_ffff).to_msh();
        assert_eq!(Color::from_msh(m, s, h), Color(0xfbff_ffff));

        let [m, s, h] = Color(0xfffc_ffff).to_msh();
        assert_eq!(Color::from_msh(m, s, h), Color(0xfefc_feff));

        let [m, s, h] = Color(0xffff_fcff).to_msh();
        assert_eq!(Color::from_msh(m, s, h), Color(0xffff_fbff));
    }
}
