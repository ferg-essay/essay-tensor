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

    for m in 0..m {
        for n in 0..n {
            let mut v: f32 = 0.;
            
            //let a = a.add(m * rsa);
            //let b = b.add(n * csb);

            // Unrolling improves >100%
            //
            //for k in 0..inner_len {
            //    let a = *a_start.add(k * a_stride[0]);
            //    let b = *b_start.add(k * b_stride[1]);
            //    v += a * b;
            //}
            let mut k = k;

            while k > 3 {
                k -= 4;

                let v0 = *a![m, k + 0] * *b![k + 0, n];
                let v1 = *a![m, k + 1] * *b![k + 1, n];
                let v2 = *a![m, k + 2] * *b![k + 2, n];
                let v3 = *a![m, k + 3] * *b![k + 3, n];

                v += v0 + v1 + v2 + v3;
            }

            for k in 0..=k {
                v += *a![m, k] * *b![k, n];
            }

            *c.add(rsc * m + csc * n) = alpha * v;
        }
    }
}

#[inline(always)]
unsafe fn at(ptr: *const f32, i: usize) -> f32 {
    *ptr.add(i)
}
