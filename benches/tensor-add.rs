use std::time::Instant;

use essay_tensor::Tensor;

fn main() {
    println!("Benchmarking tensor add");

    for _ in 0..16 {
        let a = Tensor::zeros(&[32 * 1024]);
        let b = Tensor::ones(&[32 * 1024]);

        let start = Instant::now();
        let _c = &a + &b;
        let time_sum = start.elapsed();

        let start = Instant::now();
        let _c = &a * &b;
        let time_mul = start.elapsed();

        let a = Tensor::zeros(&[256, 256]);
        let b = Tensor::ones(&[256, 256]);

        let start = Instant::now();
        let _c = &a.matmul(&b);
        let time_matmul = start.elapsed();

        let a = Tensor::zeros(&[8192, 256]);
        let b = Tensor::ones(&[8192]);

        let start = Instant::now();
        let _c = &a.matvec(&b);
        let time_matvec = start.elapsed();

        println!("sum={time_sum:?}, mul={time_mul:?}, matmul={time_matmul:?} matvec={time_matvec:?}");
    }
}
