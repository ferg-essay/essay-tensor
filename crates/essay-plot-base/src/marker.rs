use crate::{Affine2d, Color, TextureId};

///
/// Offset and color information for repeated paths like markers.
/// 
/// Each instance has an affine transformation (offset, rotation, and scale),
/// and optional color and texture.
/// 
/// Note: because the scaling is of the rendered path, line widths are scaled
/// as well. Scaled markers with a consistent line width must be rendered
/// separately.
/// 
pub struct Instance {
    affine: Affine2d,
    color: Option<Color>,
    texture: Option<TextureId>,
}

impl Instance {
    #[inline]
    pub fn new(
        affine: Affine2d,
        color: Option<Color>,
        texture: Option<TextureId>,
    ) -> Self {
        Self {
            affine,
            color,
            texture,
        }
    }

    ///
    /// The affine transformation (offset, scale, rotation) of the instance.
    ///
    #[inline]
    pub fn affine(&self) -> &Affine2d {
        &self.affine
    }

    ///
    /// The color of the instance if it overrides the default.
    ///
    #[inline]
    pub fn color(&self) -> &Option<Color> {
        &self.color
    }

    ///
    /// The texture of the instance.
    ///
    #[inline]
    pub fn texture(&self) -> &Option<TextureId> {
        &self.texture
    }

}