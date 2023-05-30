use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;


pub struct RandomSource {
    rng: Box<dyn RngCore>,
}

impl RandomSource {
    pub fn new(seed: Option<u64>) -> RandomSource {
        let rng: Box<dyn RngCore> = match seed {
            Some(seed) => Box::new(ChaCha8Rng::seed_from_u64(seed)),
            None => Box::new(rand::thread_rng())
        };

        Self {
            rng,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.rng.next_u64()
    }
}
