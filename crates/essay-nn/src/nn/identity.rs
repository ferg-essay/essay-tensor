pub struct Identity {
    len: usize,
}

impl Identity {
    pub fn new(len: usize) -> Self {
        Self {
            len: len,
        }
    }
}