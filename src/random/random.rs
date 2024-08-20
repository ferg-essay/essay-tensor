use std::{cell::RefCell, rc::Rc, sync::atomic::{AtomicU64, Ordering}};

use rand::RngCore;

// LCG/PCG random number generator from
// https://nullprogram.com/blog/2019/11/19/
pub struct Rand32(pub u64);

impl Rand32 {
    const A : u64 = 0x4822581B885237D5;
    const C : u64 = 0x6ED29E641A5F9CE5;
    const M : u32 = 0x8A53002D;

    /// New random generator with random seed
    pub fn new() -> Self {
        Self(next_u64())
    }

    #[inline]
    pub fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(Self::A).wrapping_add(Self::C);

        let mut r = (self.0 >> 32) as u32;
        r ^= r >> 16;

        r.wrapping_mul(Self::M)
    }

    #[inline]
    pub fn next_uniform(&mut self) -> f32 {
        self.next() as f32 / u32::MAX as f32
    }

    #[inline]
    pub fn next_normal(&mut self) -> f32 {
        let rng_a = self.next_uniform();
        let rng_b = self.next_uniform();

        // Box-Muller
        (-2. * rng_a.ln()).sqrt() * (std::f32::consts::TAU * rng_b).cos()
    }
}

impl From<&str> for Rand32 {
    fn from(value: &str) -> Self {
        let mut seed: u64 = 0;

        for v in value.as_bytes() {
            seed = seed.wrapping_mul(Self::A).wrapping_add(Self::C).wrapping_add(*v as u64);
        }

        Self(seed)
    }
}
pub struct Rand64(pub u128);

impl Rand64 {
    const A : u128 = 0xACAEAEFB4BF4A6A78B6E278B66409E7D;
    const C : u128 = 0x351402A3544A25016AEE39FBBADBB38D;
    const M : u64 = 0xA18E4C689403AE2D;

    /// New random generator with random seed
    pub fn new() -> Self {
        Self(next_u64() as u128)
    }

    #[inline]
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(Self::A).wrapping_add(Self::C);

        let mut r = (self.0 >> 32) as u64;
        r ^= r >> 32;

        r.wrapping_mul(Self::M)
    }

    #[inline]
    pub fn next_uniform(&mut self) -> f64 {
        self.next() as f64 / u32::MAX as f64
    }

    #[inline]
    pub fn next_normal(&mut self) -> f64 {
        let rng_a = self.next_uniform();
        let rng_b = self.next_uniform();

        // Box-Muller
        (-2. * rng_a.ln()).sqrt() * (std::f64::consts::TAU * rng_b).cos()
    }
}

impl From<&str> for Rand64 {
    fn from(value: &str) -> Self {
        let mut seed: u128 = 0;

        for v in value.as_bytes() {
            seed = seed.wrapping_mul(Self::A).wrapping_add(Self::C).wrapping_add(*v as u128);
        }

        Self(seed)
    }
}

pub fn random_seed(seed: u64) {
    SEED.store(seed, Ordering::Release);
}

fn next_u64() -> u64 {
    LOCAL_RNG.with(|x| { x.borrow_mut().next() })
}

thread_local! {
    static LOCAL_RNG: Rc<RefCell<Rand64>> = {
        let mut seed;

        loop {
            let old = SEED.load(Ordering::Acquire);

            if old == 0 {
                if cfg!(test) {
                    seed = 42
                } else {
                    seed = rand::thread_rng().next_u64()
                }
            } else {
                seed = Rand64(old as u128).next()
            }

            if SEED.compare_exchange(old, seed, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                break;
            }
        }

        Rc::new(RefCell::new(Rand64(seed as u128)))
    }
}
static SEED : AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
mod test {
    use super::Rand32;

    #[test]
    fn rand32_uniform() {
        let mut rand = Rand32::new();

        assert_eq!(0.31211653, rand.next_uniform());
        assert_eq!(0.27053323, rand.next_uniform());
        assert_eq!(0.25252375, rand.next_uniform());
        assert_eq!(0.7638205, rand.next_uniform());
        assert_eq!(0.50568986, rand.next_uniform());

        assert_eq!(0.6439561, rand.next_uniform());
        assert_eq!(0.61524963, rand.next_uniform());
        assert_eq!(0.30722934, rand.next_uniform());
        assert_eq!(0.41691482, rand.next_uniform());
        assert_eq!(0.75309724, rand.next_uniform());
    }

    #[test]
    fn rand32_normal() {
        let mut rand = Rand32::new();

        assert_eq!(-0.1963334, rand.next_normal());
        assert_eq!(0.14388694, rand.next_normal());
        assert_eq!(-0.72176474, rand.next_normal());
        assert_eq!(-0.34682474, rand.next_normal());
        assert_eq!(0.02574057, rand.next_normal());

        assert_eq!(-0.33432406, rand.next_normal());
        assert_eq!(-0.01841907, rand.next_normal());
        assert_eq!(1.6143173, rand.next_normal());
        assert_eq!(-0.9866579, rand.next_normal());
        assert_eq!(1.1143453, rand.next_normal());
    }

    #[test]
    fn rand32_u32_seed_0() {
        let mut rnd = Rand32(0);

        assert_eq!(2719371262, rnd.next());
        assert_eq!(3366669408, rnd.next());
        assert_eq!(1060708876, rnd.next());
        assert_eq!(3350485177, rnd.next());
        assert_eq!(0266455276, rnd.next());

        assert_eq!(3435573958, rnd.next());
        assert_eq!(1339492206, rnd.next());
        assert_eq!(1385110079, rnd.next());
        assert_eq!(0222350999, rnd.next());
        assert_eq!(0936719904, rnd.next());

        let mut rnd = Rand32(0);

        assert_eq!(2719371262, rnd.next());
        assert_eq!(3366669408, rnd.next());
        assert_eq!(1060708876, rnd.next());
        assert_eq!(3350485177, rnd.next());
        assert_eq!(0266455276, rnd.next());

        assert_eq!(3435573958, rnd.next());
        assert_eq!(1339492206, rnd.next());
        assert_eq!(1385110079, rnd.next());
        assert_eq!(0222350999, rnd.next());
        assert_eq!(0936719904, rnd.next());
    }

    #[test]
    fn rand32_u32_seeds() {
        let mut rnd = Rand32(0);
        assert_eq!(2719371262, rnd.next());

        let mut rnd = Rand32(1);
        assert_eq!(0067131503, rnd.next());

        let mut rnd = Rand32(2);
        assert_eq!(3750639004, rnd.next());

        let mut rnd = Rand32(10);
        assert_eq!(0613872473, rnd.next());

        let mut rnd = Rand32::from("");
        assert_eq!(2719371262, rnd.next());

        let mut rnd = Rand32::from("a");
        assert_eq!(2183894628, rnd.next());

        let mut rnd = Rand32::from(" ");
        assert_eq!(3200514678, rnd.next());

        let mut rnd = Rand32::from("aa");
        assert_eq!(3194888993, rnd.next());
    }

    #[test]
    fn rand32_uniform_seed_0() {
        let mut rnd = Rand32(0);

        assert_eq!(0.63315296, rnd.next_uniform());
        assert_eq!(0.7838638, rnd.next_uniform());
        assert_eq!(0.24696553, rnd.next_uniform());
        assert_eq!(0.78009564, rnd.next_uniform());
        assert_eq!(0.062038954, rnd.next_uniform());

        assert_eq!(0.7999069, rnd.next_uniform());
        assert_eq!(0.31187484, rnd.next_uniform());
        assert_eq!(0.32249606, rnd.next_uniform());
        assert_eq!(0.051770125, rnd.next_uniform());
        assert_eq!(0.21809709, rnd.next_uniform());
    }

    #[test]
    fn rand32_new() {
        let mut rnd = Rand32::new();

        assert_eq!(0.0040017962, rnd.next_uniform());
        assert_eq!(0.35252944, rnd.next_uniform());
        assert_eq!(0.31170943, rnd.next_uniform());

        let mut rnd = Rand32::new();

        assert_eq!(0.78586006, rnd.next_uniform());
        assert_eq!(0.81300724, rnd.next_uniform());
        assert_eq!(0.36018643, rnd.next_uniform());

        let mut rnd = Rand32::new();

        assert_eq!(0.79271746, rnd.next_uniform());
        assert_eq!(0.5769489, rnd.next_uniform());
        assert_eq!(0.053381264, rnd.next_uniform());
    }
}
