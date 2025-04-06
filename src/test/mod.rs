use std::{ops, sync::{Arc, Mutex, OnceLock}};

use num_traits::{One, Zero};

use crate::tensor::Type;

///
/// Minimal testing tensor type with only debugging/assertion traits
/// 
#[derive(PartialEq, Debug)]
pub struct T(pub usize);

impl Type for T {}

///
/// Minimal testing tensor type with Clone
/// 
#[derive(Clone, PartialEq, Debug)]
pub struct C(pub usize);

impl Type for C {}

///
/// Minimal testing tensor type with Zero
/// 
#[derive(Clone, PartialEq, Debug)]
pub struct Z(pub usize);

impl Type for Z {}

impl Zero for Z {
    fn zero() -> Self {
        Z(0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl ops::Add for Z {
    type Output = Z;

    fn add(self, rhs: Self) -> Self::Output {
        Z(self.0 + rhs.0)
    }
}

///
/// Minimal testing tensor type with One
/// 
#[derive(Clone, PartialEq, Debug)]
pub struct O(pub usize);

impl Type for O {}

impl One for O {
    fn one() -> Self {
        O(1)
    }

    fn is_one(&self) -> bool {
        self.0 == 1
    }
}

impl ops::Mul for O {
    type Output = O;

    fn mul(self, rhs: Self) -> Self::Output {
        O(self.0 * rhs.0)
    }
}

///
/// Minimal testing tensor type with Zero + One
/// 
#[derive(Clone, PartialEq, Debug)]
pub struct ZO(pub usize);

impl Type for ZO {}

impl Zero for ZO {
    fn zero() -> Self {
        ZO(0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }
    
    fn set_zero(&mut self) {
        *self = Zero::zero();
    }
}

impl One for ZO {
    fn one() -> Self {
        ZO(1)
    }

    fn is_one(&self) -> bool {
        self.0 == 1
    }
}

impl ops::Add for ZO {
    type Output = ZO;

    fn add(self, rhs: Self) -> Self::Output {
        ZO(self.0 + rhs.0)
    }
}

impl ops::Mul for ZO {
    type Output = ZO;

    fn mul(self, rhs: Self) -> Self::Output {
        ZO(self.0 * rhs.0)
    }
}

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
