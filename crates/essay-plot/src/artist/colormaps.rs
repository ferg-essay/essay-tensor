use essay_plot_base::Color;

use super::ColorMap;

pub enum ColorMaps {
    Default,
    BlueOrange,
    BlackWhite,
}

impl From<ColorMaps> for ColorMap {
    fn from(value: ColorMaps) -> Self {
        match value {
            ColorMaps::BlueOrange | ColorMaps::Default => {
                // Top 1% options: vermillion, red, bright red
                ColorMap::from([
                    (0., "navy"),  // bottom 1% distinct
                    // cool, saturated blue to warm, unsaturated blue
                    (0.01, "royal blue"), (0.1, "blue"), (0.2, "azure"),
                    (0.5, "white"),
                    // cool, unsaturated orange to warm, saturated orange
                    (0.8, "amber"), (0.9, "orange"), (0.99, "orange red"), 
                    (1.0, "bright red") // top 1% distinct
                ])
            }

            ColorMaps::BlackWhite => {
                ColorMap::from([
                    (0., "black"),
                    (1., "white"),
                ])
            }
        }
    }
}