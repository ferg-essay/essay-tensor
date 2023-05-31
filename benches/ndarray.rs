use std::time::Instant;

use ndarray::Array1;


fn main() {
    println!("Benchmarking ndarray");

    let len = 65536;
    let a : Array1<f32> = Array1::<f32>::ones(len);
    let b : Array1<f32> = Array1::<f32>::ones(len);

    //let a1 = a.clone();
    //let b1 = b.clone();
    for _ in 0..16 {
        let start = Instant::now();
        let _c = &a + &b;
        let time_one = start.elapsed();

        let start = Instant::now();
        for _k in 0..100 {
            let _c = &a + &b;
        }
        let time_sum = start.elapsed();
        println!("one={time_one:?} sum={time_sum:?}");
    }
}
