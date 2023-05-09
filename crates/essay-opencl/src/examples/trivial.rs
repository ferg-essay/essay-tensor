use ocl::{self, ProQue};

const RESULTS_TO_PRINT: usize = 20;
const WORK_SIZE: usize = 1 << 20;
const COEFF: f32 = 5432.1;

static KERNEL_SRC: &'static str = r#"
    __kernel void add(
        __global float* buffer, float scalar
    ) {
        uint const idx = get_global_id(0);
        buffer[idx] += scalar;
    }
"#;

fn trivial() -> ocl::Result<()> {
    let pro_que = ProQue::builder().src(KERNEL_SRC).dims(1 << 20).build()?;
    let buffer = pro_que.create_buffer::<f32>()?;

    let kernel = pro_que
        .kernel_builder("add")
        .arg(&buffer)
        .arg(10.0f32)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    let mut vec = vec![0.0f32; buffer.len()];
    buffer.read(&mut vec).enq()?;

    println!("Value at index [{}] is now '{}'!", 2007, vec[2007]);

    unsafe {
        kernel.enq()?;
    }

    let mut vec = vec![0.0f32; buffer.len()];
    buffer.read(&mut vec).enq()?;
    println!("Value at index [{}] is now '{}'!", 2007, vec[2007]);

    Ok(())
}

#[test]
fn test() {
    trivial();
}
