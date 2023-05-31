use std::time::Instant;

use dfdx::{tensor::{Cpu, Tensor, ZerosTensor}, shapes::Rank1};


fn main() {
    println!("Benchmarking dfdx");

    let dev: Cpu = Default::default();

    let a : Tensor<Rank1<65536>, f32, _> = dev.zeros();
    let b : Tensor<Rank1<65536>, f32, _> = dev.zeros();

    let a1 = a.clone();
    let b1 = b.clone();
    let start = Instant::now();
    let _c = a1 + b1;
    let time_one = start.elapsed();

    let start = Instant::now();
    for _k in 0..100 {
        let _c = a.clone() + b.clone();
    }
    let time_sum = start.elapsed();
    println!("one={time_one:?} sum={time_sum:?}");
}
