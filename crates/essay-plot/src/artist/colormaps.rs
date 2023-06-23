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
                // Top 1% options: vermillion, red, bright red, tomato red
                // Bottom 1% options: navy, dark navy, ultramarine blue, night blue
                // TODO: possibly use hsv instead of color names
                ColorMap::from([
                    (0., "cobalt blue"),  // bottom 1% distinct
                    // cool, saturated blue to warm, unsaturated blue
                    (0.01, "ultramarine blue"), (0.1, "blue"), (0.2, "azure"),
                    (0.5, "white"),
                    // cool, unsaturated orange to warm, saturated orange
                    (0.8, "amber"), (0.9, "orange"), (0.99, "tomato red"), 
                    (1.0, "red") // top 1% distinct
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