use core::fmt;
use std::collections::HashMap;

use wgpu_glyph::ab_glyph::{self, Font, PxScale};

use super::text_texture::TextTexture;

pub struct TextCache {
    width: u32,
    height: u32,

    data: Vec<u8>,

    x: u32,
    y: u32,

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
            x: 0,
            y: 0,
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
                    "../../assets/fonts/DejaVuSans.ttf"
                )).unwrap();

                FontItem {
                    id: FontId(len),
                    font,
                }
            }
        );

        ScaledFont::new(entry.id, entry.font.clone(), size)
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

    fn add_glyph(&mut self, font: &ab_glyph::FontArc, size: u16, ch: char) -> TextRect {
        //let scale = font.pt_to_px_scale(size as f32).unwrap();
        let scale = PxScale::from(size as f32);
        let glyph = font.glyph_id(ch).with_scale(scale);

        let data = self.data.as_mut_slice();
        let w = self.width as u32;

        let rect = match font.outline_glyph(glyph) {
            Some(og) => {
                let bounds = og.px_bounds();

                let dx = bounds.max.x - bounds.min.x;
                let dy = bounds.max.y - bounds.min.y;

                let xc = self.x;
                let yc = self.y; //  + dy as u32;

                og.draw(|x, y, v| {
                    let v = (v * 255.).round().clamp(0., 255.) as u8;
                    data[(xc + x + w * (yc + dy as u32 - y - 1)) as usize] = v;
                });

                let rect = TextRect::new(
                    bounds.min.x as i16,
                    bounds.min.y as i16,
                    bounds.max.x as i16,
                    bounds.max.y as i16,
                    og.glyph().position.y as i16,
                    xc as f32 / self.width as f32,
                    yc as f32 / self.height as f32,
                    (xc as f32 + dx) / self.width as f32,
                    (yc as f32 + dy) / self.height as f32,
                );

                let mut xc = xc + dx as u32 + 1;
                xc += (4 - xc % 4) % 4;

                self.x = xc;
                self.y = yc;

                rect
            },
            None => {
                TextRect::none()
            }
        };

        self.is_modified = true;

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
    descent: f32,
}

impl ScaledFont {
    fn new(
        id: FontId,
        font: ab_glyph::FontArc,
        size: u16
    ) -> Self {
        let height = font.ascent_unscaled() + font.descent_unscaled();
        let descent = size as f32 * font.descent_unscaled() / height;
        
        Self {
            id,
            font,
            size,
            descent,
        }
    }

    pub(crate) fn _height(&self) -> f32 {
        self.size as f32
    }

    pub(crate) fn descent(&self) -> f32 {
        self.descent
    }
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
    px_min: i16,
    py_min: i16,
    px_max: i16,
    py_max: i16,

    _desc: i16,

    tx_min: f32,
    ty_min: f32,
    tx_max: f32,
    ty_max: f32,
}

impl TextRect {
    fn new(
        px_min: i16,
        py_min: i16,
        px_max: i16,
        py_max: i16,

        desc: i16,

        tx_min: f32,
        ty_min: f32,
        tx_max: f32,
        ty_max: f32,
    ) -> Self {
        Self {
            px_min,
            py_min,
            px_max,
            py_max,

            _desc: desc,
    
            tx_min,
            ty_min,
            tx_max,
            ty_max,
        }
    }

    fn none() -> Self {
        Self {
            px_min: 0,
            py_min: 0,
            px_max: 0,
            py_max: 0,

            _desc: 0,
    
            tx_min: 0.,
            ty_min: 0.,
            tx_max: 0.,
            ty_max: 0.,
        }
    }

    #[inline]
    pub(crate) fn is_none(&self) -> bool {
        self.px_min == self.px_max
    }

    #[inline]
    pub(crate) fn _px_min(&self) -> i16 {
        self.px_min
    }

    #[inline]
    pub(crate) fn _py_min(&self) -> i16 {
        self.py_min
    }

    #[inline]
    pub(crate) fn _px_max(&self) -> i16 {
        self.px_max
    }

    #[inline]
    pub(crate) fn py_max(&self) -> i16 {
        self.py_max
    }

    #[inline]
    pub(crate) fn px_w(&self) -> u16 {
        (self.px_max - self.px_min) as u16
    }

    #[inline]
    pub(crate) fn px_h(&self) -> u16 {
        (self.py_max - self.py_min) as u16
    }

    #[inline]
    pub(crate) fn _desc(&self) -> u16 {
        self._desc as u16
    }

    #[inline]
    pub(crate) fn tx_min(&self) -> f32 {
        self.tx_min
    }

    #[inline]
    pub(crate) fn ty_min(&self) -> f32 {
        self.ty_min
    }

    #[inline]
    pub(crate) fn tx_max(&self) -> f32 {
        self.tx_max
    }

    #[inline]
    pub(crate) fn ty_max(&self) -> f32 {
        self.ty_max
    }
}