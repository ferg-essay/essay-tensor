use std::sync::{Arc, Mutex, OnceLock};

use crate::tensor::Type;


#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Dead(pub u32);

impl Drop for Dead {
    fn drop(&mut self) {
        Messages::send(&format!("Dead({:.08x})", self.0));
    }
}

impl Type for Dead {}

pub struct Messages {
    vec: Arc<Mutex<Vec<String>>>,
}

impl Messages {
    pub(crate) fn send(msg: &str) {
        Self::write(|vec| vec.push(String::from(msg)));
    }

    pub(crate) fn clear() {
        Self::take();
    }

    pub(crate) fn take() -> Vec<String> {
        Self::write(|vec| vec.drain(..).collect())
    }

    fn write<R>(f: impl FnOnce(&mut Vec<String>) -> R) -> R {
        let messages = DEADBEEF.get_or_init(|| Self {
            vec: Arc::new(Mutex::new(Vec::new())),
        });

        let mut vec = messages.vec.lock().unwrap();

        (f)(&mut vec)
    }
}

static DEADBEEF: OnceLock<Messages> = OnceLock::new();
