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
    #[cfg(not(feature="naive"))]
    matrixmultiply::sgemm(
        m, k, n,
        alpha,
        a, rsa as isize, csa as isize,
        b, rsb as isize, csb as isize,
        beta,
        c, rsc as isize, csc as isize  
    );
    
    #[cfg(feature="naive")]
    sgemm_naive(
        m, k, n,
        alpha,
        a, rsa as usize, csa as usize,
        b, rsb as usize, csb as usize,
        beta,
        c, rsc as usize, csc as usize,
    );
}
