use std::{time::Instant, ptr};

use essay_tensor::{Tensor, tensor::TensorUninit};
use ocl::{Platform, Device, Program, Queue, Buffer, flags, Kernel, builders::KernelBuilder, MemMap};


static KERNEL_SRC: &'static str = r#"
    __kernel void add(
        __global float* buffer, float scalar
    ) {
        uint const idx = get_global_id(0);
        buffer[idx] += scalar;
    }
"#;

static BI_SRC: &'static str = r#"
    __kernel void add_bi(
        __global float* out,
        __global float* a,
        __global float* b
    ) {
        uint const idx = get_global_id(0);
        out[idx] = a[idx] + b[idx];
    }
"#;

pub struct OclContext {
    device: ocl::Device,
    context: ocl::Context,
    queue: ocl::Queue,
}

impl OclContext {
    pub fn new() -> Self {
        let platform = Platform::default();
        let device = Device::first(platform).unwrap();
        let context = ocl::Context::builder()
            .platform(platform)
            .devices(device.clone())
            .build().unwrap();

        let queue = Queue::new(&context, device, None).unwrap();

        Self {
            device,
            context,
            queue,
        }
    }

    pub fn program(&self, src: &str) -> ocl::Program {
        let program = Program::builder()
        .devices(self.device)
        .src(src)
        .build(&self.context).unwrap();

        program
    }

    pub fn buffer(&self, size: usize) -> ocl::Buffer::<f32> {
        let buffer = Buffer::<f32>::builder()
        .queue(self.queue.clone())
        .flags(flags::MEM_READ_WRITE)
        .len(size)
        //.fill_val(0f32)
        .build().unwrap();

        buffer
    }

    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    pub fn kernel(
        &self, 
        kernel: Kernel,
        args: Vec<ocl::Buffer<f32>>,
    ) -> OclKernel {
        OclKernel {
            kernel: kernel,
            buffers: args, 
        }
    }
}

pub struct OclKernel {
    kernel: ocl::Kernel,
    buffers: Vec<ocl::Buffer<f32>>,
}

impl OclKernel {
    pub fn buffer(&self, index: usize) -> Option<&ocl::Buffer<f32>> {
        self.buffers.get(index)
    }

    pub fn read(&self, context: &OclContext, index: usize, data: &mut [f32]) {
        self.buffers[index]
            .cmd()
            .queue(&context.queue)
            .offset(0)
            .read(data)
            .enq()
            .unwrap();
    }

    pub fn write(&self, context: &OclContext, index: usize, data: &[f32]) {
        self.buffers[index]
            .cmd()
            .queue(&context.queue)
            .offset(0)
            .write(data)
            .enq()
            .unwrap();
    }

    pub unsafe fn map(&self, context: &OclContext, index: usize) -> MemMap<f32> {
        self.buffers[index]
            .cmd()
            .queue(&context.queue)
            .offset(0)
            .map()
            .enq()
            .unwrap()
    }

    pub fn enq(&self) {
        unsafe {
            self.kernel
                .cmd()
                //.queue(&context.queue)
                .global_work_offset(self.kernel.default_global_work_offset())
                //.global_work_size(dims)
                //.global_work_size(128)
                .local_work_size(self.kernel.default_local_work_size())
                .enq().unwrap();
        }
    }
}

fn trivial_exploded(src: &str) {
    let context = OclContext::new();

    let program = context.program(src);

    let len = 64 * 1024;

    let a = Tensor::ones(&[len]);
    let b = Tensor::ones(&[len]);

    let dims = a.len() * 4;

    let a_buffer = context.buffer(dims);
    let b_buffer = context.buffer(dims);
    let c_buffer = context.buffer(dims);
    
    let kernel = Kernel::builder()
        .program(&program)
        .name("add_bi")
        .queue(context.queue.clone())
        .global_work_size(dims)
        .arg(&c_buffer)
        .arg(&a_buffer)
        .arg(&b_buffer)
        .build().unwrap();

    let kernel = context.kernel(kernel, vec![a_buffer, b_buffer, c_buffer]);

    //let start = Instant::now();

    unsafe {
        //let mut map_a = kernel.map(&context, 0);
        //let mut map_b = kernel.map(&context, 1);
        //let map_c = kernel.map(&context, 2);
    //}

    let s1 = Instant::now();
    let s1a = Instant::now();

    unsafe {
        kernel.write(&context, 0, a.data().as_slice());
        kernel.write(&context, 1, b.data().as_slice());
    }

    /*
    let ptr_a = map_a.as_mut_ptr();
    let ptr_at = a.data().as_ptr();

    for i in 0..a.len() {
        *ptr_a.add(i) = *ptr_at.add(i);
    }

    let ptr_b = map_b.as_mut_ptr();
    let ptr_bt = b.data().as_ptr();

    for i in 0..b.len() {
        *ptr_b.add(i) = *ptr_bt.add(i);
    }
    */

    //kernel.buffer(0).unwrap().

    let t1a = s1a.elapsed();

    let s2 = Instant::now();
    kernel.enq();
    let t2 = s2.elapsed();

    let out = unsafe {
        let mut c = TensorUninit::<f32>::new(len);
        kernel.read(&context, 2, c.as_slice_mut());
        c.init()
    };

    let t1 = s1.elapsed();
    let s3 = Instant::now();
    let _c = a + b;
    let t3 = s3.elapsed();

    println!("T1 {} {} {:?}", 1, out[0], t1);
    println!("T1a {} {} {:?}", 1, out[0], t1a);
    println!("T2 {} {} {:?}", 1, out[1], t2);
    println!("T3 {} {} {:?}", 1, out[1], t3);
    /*
    unsafe {
        kernel.enq();
    }
    */
}    
    //println!("Value [{}] is {}", 107, a[107]);
}

#[test]
fn test() {
    trivial_exploded(BI_SRC);
}
