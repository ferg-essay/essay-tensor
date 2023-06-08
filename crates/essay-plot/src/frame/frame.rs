use crate::artist::{Style, patch::{PatchTrait, self, DisplayPatch}};

use super::{Display, Bounds};

pub struct Frame {
    top: SpineTop,
    left: SpineY,
    right: SpineRight,
    bottom: SpineX,
}

impl Frame {
    pub fn new() -> Self {
        Self {
            top: SpineTop::new(),
            left: SpineY::new(),
            right: SpineRight::new(),
            bottom: SpineX::new(),
        }
    }
}

pub struct SpineTop {
    bounds: Bounds<Display>,
    border: Option<DisplayPatch>,
}

impl SpineTop {
    pub fn new() -> Self {
        Self {
            bounds: Bounds::<Display>::none(),
            border: None,
        }
    }
}

pub struct SpineRight {
}

impl SpineRight {
    pub fn new() -> Self {
        Self {
        }
    }
}

pub struct SpineY {
}

impl SpineY {
    pub fn new() -> Self {
        Self {
        }
    }
}

pub struct SpineX {
}

impl SpineX {
    pub fn new() -> Self {
        Self {
        }
    }
}

pub struct Spine {
    style: Style,
    patch: Box<dyn PatchTrait>,
}

impl Spine {
    pub fn new() -> Self {
        Self {
            style: Style::new(),
            patch: Box::new(patch::Line::new([0., 0.], [0., 0.])),
        }
    }
}