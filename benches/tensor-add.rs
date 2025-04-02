use std::time::Instant;

use essay_tensor::{Tensor, model::Function};

fn main() {
    println!("Benchmarking tensor add");

    for _ in 0..16 {

        let a = Tensor::<f32>::zeros(1);
        let b = Tensor::<f32>::ones(1);

        let start = Instant::now();
        //let _c = &a + &b;
        let _c = &a + &b;
        let time_null = start.elapsed();

        let len = 65536;

        let a = Tensor::zeros(len);
        let b = Tensor::ones(len);

        let k = 100;

        let start = Instant::now();
        for _ in 0..k {
            //let mut vec = Vec::<f32>::new();
            //vec.resize(len, 0.);
            //let vec_arc = Rc::new(vec);
            Tensor::<f32>::zeros([len]);
        }
        let _time_zeros = start.elapsed();

        let _add = Function::new(
            (a.clone(), b.clone()),
            |(x, y), _| x + y
        );

        let start = Instant::now();
        //let _c = &a + &b;
        let c = a.clone();
        for _ in 0..k {
            let _c = &c + &b;
            //let _c = _add.call((a.clone(), b.clone()));
        }
        let time_sum = start.elapsed();

        let mut v_a = Vec::<f32>::new();
        v_a.resize(len, 1.);
        let v_a = v_a;

        let mut v_b = Vec::<f32>::new();
        v_b.resize(len, 1.);
        let v_b = v_b;
        
        let start = Instant::now();
        //let _c = &a + &b;

        for _ in 0..k {
            let mut v_c = Vec::<f32>::new();
            v_c.resize(len, 1.);

            let sa = v_a.as_slice(); // ptr(); // .as_slice();
            let sb = v_b.as_slice(); // .as_slice();
            let sc = v_c.as_mut_slice(); // as_mut_slice();

            for i in 0..len {
                sc[i] = sa[i] + sb[i];
            }
            //let _c = _add.call((a.clone(), b.clone()));
        }
        let _t_sum = start.elapsed();

        let start = Instant::now();
        for _ in 0..k {
            let _c = &a * &b;
        }
        let time_mul = start.elapsed();

        let a = Tensor::zeros([256, 256]);
        let b = Tensor::ones([256, 256]);

        let start = Instant::now();
        for _ in 0.. k {
            let _c = &a.matmul(&b);
        }
        let time_matmul = start.elapsed();

        let a = Tensor::zeros([256, 8192]);
        let b = Tensor::ones([8, 8192]);
        //let a = Tensor::zeros([1, 1]);
        //let b = Tensor::ones([1, 1]);

        let start = Instant::now();

        for _ in 0..k {
            let _c = &a.matvec(&b);
        }
        //println!("Shape {:?}", _c.shape());
        let time_matvec = start.elapsed();

        //let a = Tensor::zeros([1024]);

        //let start = Instant::now();
        //let _c = &a.softmax();
        //let softmax_matvec = start.elapsed();

        //println!("null={time_null:?} zeros={time_zeros:?} sum={time_sum:?}, tsum={t_sum:?}");
        println!("null={time_null:?} sum={time_sum:?}, mul={time_mul:?}, matmul={time_matmul:?} matvec={time_matvec:?}");
    }
}
