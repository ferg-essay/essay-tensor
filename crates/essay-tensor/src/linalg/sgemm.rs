//
// C = alpha * A * B + beta * C
// A: m by k
// B: k by n
// C: m by n
// rsx - row stride of x
// csx = col stride of x
pub unsafe fn sgemm(
    m: usize, k: usize, n: usize,
    alpha: f32,
    a: *const f32, rsa: usize, csa: usize,
    b: *const f32, rsb: usize, csb: usize,
    beta: f32,
    c: *mut f32, rsc: usize, csc: usize
) {
    matrixmultiply::sgemm(
        m, k, n,
        alpha,
        a, rsa as isize, csa as isize,
        b, rsb as isize, csb as isize,
        beta,
        c, rsc as isize, csc as isize  
    )
    
    /*
    sgemm_naive(
        m, k, n,
        alpha,
        a, rsa as usize, csa as usize,
        b, rsb as usize, csb as usize,
        beta,
        c, rsc as usize, csc as usize,
    );
    */
}

#[allow(dead_code)]
unsafe fn sgemm_naive(
    m: usize, k: usize, n: usize,
    alpha: f32,
    a: *const f32, rsa: usize, csa: usize,
    b: *const f32, rsb: usize, csb: usize,
    beta: f32,
    c: *mut f32, rsc: usize, csc: usize
) {
    assert!(beta == 0.);

    macro_rules! a {
        ($i:expr, $j:expr) => (a.add(rsa * $i + csa * $j));
    }

    macro_rules! b {
        ($i:expr, $j:expr) => (b.add(rsb * $i + csb * $j));
    }

    macro_rules! c {
        ($i:expr, $j:expr) => (c.add(rsc * $i + csc * $j));
    }

    for m in 0..m {
        for n in 0..n {
            let mut v: f32 = 0.;
            
            for k in 0..k {
                v += *a![m, k] * *b![k, n];
            }

            *c![m, n] = alpha * v;
        }
    }
}
