
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
