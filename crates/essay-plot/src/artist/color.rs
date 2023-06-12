use std::ops::Index;

use essay_plot_base::Color;


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
    pub fn tableau() -> ColorCycle {
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
