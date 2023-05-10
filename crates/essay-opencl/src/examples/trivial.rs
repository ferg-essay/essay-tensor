use ocl::{self, ProQue, Platform, Device, Context, Program, Queue, flags, Buffer, Kernel};

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

fn trivial_exploded() -> ocl::Result<()> {
    let platform = Platform::default();
    let device = Device::first(platform)?;
    let context = Context::builder()
        .platform(platform)
        .devices(device.clone())
        .build()?;
    let program = Program::builder()
        .devices(device)
        .src(KERNEL_SRC)
        .build(&context)?;

    let queue = Queue::new(&context, device, None)?;
    let dims = 1 << 20;

    let buffer = Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_WRITE)
        .len(dims)
        .fill_val(0f32)
        .build()?;

    let kernel = Kernel::builder()
        .program(&program)
        .name("add")
        .queue(queue.clone())
        .global_work_size(dims)
        .arg(&buffer)
        .arg(&10.0f32)
        .build()?;
    unsafe {
        kernel
            .cmd()
            .queue(&queue)
            .global_work_offset(kernel.default_global_work_offset())
            .global_work_size(dims)
            .local_work_size(kernel.default_local_work_size())
            .enq()?;
    }
    
    let mut vec = vec![0.0f32; dims];
    buffer.cmd().queue(&queue).offset(0).read(&mut vec).enq()?;
    println!("Value [{}] is {}", 2007, vec[2007]);

    Ok(())
}

#[test]
fn test() {
    trivial_exploded();
}
