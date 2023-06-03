#[derive(Debug)]
pub enum BackendErr {
    NotImplemented,
}

pub type Result<T, E = BackendErr> = std::result::Result<T, E>;

pub trait Backend {
    fn main_loop(&mut self) -> Result<()> {
        Err(BackendErr::NotImplemented)
    }
}