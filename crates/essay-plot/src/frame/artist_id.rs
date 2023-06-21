use super::FrameId;


#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArtistId {
    frame: FrameId,
    artist: ArtistEnum,
    index: usize,
}

impl ArtistId {
    #[inline]
    pub fn frame(&self) -> FrameId {
        self.frame
    }

    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }

    // TODO: eliminate need for this function
    pub(crate) fn new_data(frame: FrameId, index: usize) -> ArtistId {
        ArtistId {
            frame,
            artist: ArtistEnum::Data,
            index,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArtistEnum {
    None,

    Frame, // frame itself

    LeftFrame,
    RightFrame,
    TopFrame,
    BottomFrame,

    Data, // databox artists
}
