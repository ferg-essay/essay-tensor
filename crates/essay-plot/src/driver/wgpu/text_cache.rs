use core::fmt;
use std::collections::HashMap;

use wgpu_glyph::ab_glyph::{self, PxScale, Font};

use super::text_texture::TextTexture;

//let font = ab_glyph::FontArc::try_from_slice(include_bytes!(
//    "fonts/OpenSans-Medium.ttf"
//)).unwrap();

pub struct TextCache {
    width: u32,
    height: u32,

    data: Vec<u8>,

    font_map: HashMap<String, FontItem>,
    glyph_map: HashMap<GlyphId, TextRect>,
    is_modified: bool,
}

impl TextCache {
    pub fn new(width: u32, height: u32) -> Self {
        assert!(width % 256 == 0);

        let mut data = Vec::new();
        data.resize((width * height) as usize, 0);

        Self {
            width,
            height,
            data,
            font_map: HashMap::default(),
            glyph_map: HashMap::default(),
            is_modified: true,
        }
    }

    pub fn font(&mut self, name: &str, size: u16) -> ScaledFont {
        let len = self.font_map.len();

        let entry = self.font_map.entry(name.to_string())
            .or_insert_with(|| {
                let font = ab_glyph::FontArc::try_from_slice(include_bytes!(
                    "fonts/OpenSans-Medium.ttf"
                )).unwrap();

                FontItem {
                    id: FontId(len),
                    font,
                }
            }
        );

        ScaledFont {
            id: entry.id,
            font: entry.font.clone(),
            size: size,
        }
    }

    pub fn glyph(&mut self, font: &ScaledFont, glyph: char) -> TextRect {
        let glyph_id = GlyphId::new(font.id, font.size, glyph);

        if let Some(rect) = self.find_glyph(&glyph_id) {
            return rect;
        }

        let rect = self.add_glyph(&font.font, font.size, glyph);

        self.glyph_map.insert(glyph_id, rect.clone()); 

        rect
    }

    fn find_glyph(&mut self, glyph_id: &GlyphId) -> Option<TextRect> {
        match self.glyph_map.get(glyph_id) {
            Some(rect) => Some(rect.clone()),
            None => None,
        }
    }

    fn add_glyph(&mut self, font: &ab_glyph::FontArc, size: u16, glyph: char) -> TextRect {
        let scale = PxScale::from(size as f32);

        let glyph = font.glyph_id(glyph).with_scale(scale);

        let data = self.data.as_mut_slice();
        let w = self.width as u32;

        let xc = 0;
        let yc = 0;
        
        //println!("Size {:?}", font);
        //println!("Scale {:?}", font.pt_to_px_scale(12.0));
        let rect = match font.outline_glyph(glyph) {
            Some(og) => {
                let bounds = og.px_bounds();
            
                let dx = bounds.max.x - bounds.min.x;
                let dy = bounds.max.y - bounds.min.y;

                og.draw(|x, y, v| {
                    let v = (v * 255.).round().clamp(0., 255.) as u8;
                    data[(xc + x + w * (yc + y)) as usize] = v;
                });

                TextRect::new(
                    xc as u16, yc as u16,
                    xc as u16 + dx as u16, yc as u16 + dy as u16
                )
            },
            None => {
                TextRect::new(0, 0, 0, 0)
            }
        };

        rect
    }

    pub(crate) fn flush(
        &mut self, 
        queue: &wgpu::Queue, 
        texture: &TextTexture,
    ) {
        if self.is_modified {
            self.is_modified = false;

            texture.write_data(queue, &self.data);
        }
    }
}

struct FontItem {
    id: FontId,
    font: ab_glyph::FontArc,
}

pub struct ScaledFont {
    id: FontId,
    font: ab_glyph::FontArc,
    size: u16,
}

impl fmt::Debug for ScaledFont {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScaledFont")
            .field("id", &self.id)
            .field("font", &self.font)
            .field("size", &self.size)
            .finish()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FontId(usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GlyphId {
    font: FontId,
    size: u16,
    glyph: char,
}

impl GlyphId {
    fn new(font: FontId, size: u16, glyph: char) -> Self {
        Self {
            font,
            size,
            glyph,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TextRect {
    xmin: f32,
    ymin: f32,
    xmax: f32,
    ymax: f32,
}

impl TextRect {
    fn new(x: u16, y: u16, width: u16, height: u16) -> Self {
        Self {
            xmin: x as f32,
            ymin: y as f32,
            xmax: (x + width) as f32,
            ymax: (y + height) as f32,
        }
    }

    #[inline]
    pub(crate) fn is_none(&self) -> bool {
        self.xmin == self.xmax
    }

    #[inline]
    pub(crate) fn xmin(&self) -> f32 {
        self.xmin
    }

    #[inline]
    pub(crate) fn ymin(&self) -> f32 {
        self.ymin
    }

    #[inline]
    pub(crate) fn xmax(&self) -> f32 {
        self.xmax
    }

    #[inline]
    pub(crate) fn ymax(&self) -> f32 {
        self.ymax
    }

    #[inline]
    pub(crate) fn w(&self) -> f32 {
        self.xmax - self.xmin
    }

    #[inline]
    pub(crate) fn h(&self) -> f32 {
        self.ymax - self.ymin
    }
}