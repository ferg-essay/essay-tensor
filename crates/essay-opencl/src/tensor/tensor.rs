use ocl::{Platform, Device, Program, Queue, Buffer, flags, Kernel, builders::KernelBuilder};


static KERNEL_SRC: &'static str = r#"
    __kernel void add(
        __global float* buffer, float scalar
    ) {
        uint const idx = get_global_id(0);
        buffer[idx] += scalar;
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

    let dims = 1 << 20;

    let buffer = context.buffer(dims);
    
    let kernel = Kernel::builder()
        .program(&program)
        .name("add")
        .queue(context.queue.clone())
        .global_work_size(dims)
        .arg(&buffer)
        .arg(&10.0f32)
        .build().unwrap();

    let kernel = context.kernel(kernel, vec![buffer]);

    kernel.write(&context, 0, &vec![100.0f32; 128]);

    kernel.enq();
    /*
    unsafe {
        kernel.enq();
    }
    */
    
    let mut vec = vec![0.0f32; dims];
    //buffer.cmd().queue(&context.queue).offset(0).read(&mut vec).enq()?;
    kernel.read(&context, 0, &mut vec);
    println!("Value [{}] is {}", 107, vec[107]);
    println!("Value [{}] is {}", 2007, vec[2007]);
}

#[test]
fn test() {
    trivial_exploded(KERNEL_SRC);
}
