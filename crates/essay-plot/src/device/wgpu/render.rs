use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    position: [f32; 2],
    tex_coord: [f32; 2],
    color: u32,
}

impl Vertex {
    const ATTRS: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2=> Uint32 ];

    pub(crate) fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }
}

pub struct VertexBuffer {
    stride: usize,
    vec: Vec<Vertex>,
    buffer: wgpu::Buffer,
}

impl VertexBuffer {
    pub(crate) fn new(len: usize, device: &wgpu::Device) -> Self {
        let mut vec = Vec::<Vertex>::new();
        vec.resize(len, Vertex { position: [0.0, 0.0], tex_coord: [0.0, 0.0], color: 0x00000000 });
        println!("Bufzi {:?}", vec.as_slice().len());

        path_render2(vec.as_mut_slice());

        let buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(vec.as_slice()),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );
    
        Self {
            stride: std::mem::size_of::<Vertex>(),
            vec,
            buffer,
        }
    }

    pub(crate) fn stride(&self) -> usize {
        self.stride
    }

    pub(crate) fn as_slice(&self) -> &[Vertex] {
        &self.vec
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [Vertex] {
        &mut self.vec
    }

    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub(crate) fn desc(&self) -> wgpu::VertexBufferLayout {
        Vertex::desc()
    }
}

pub fn path_render(vertices: &mut [Vertex]) -> usize {
    vertices[0] = ([0.0, 0.0], 0x0000ffff).into();
    vertices[1] = ([0.0, 1.0], 0x00ff00ff).into();
    vertices[2] = ([1.0, 0.0], 0x0000ffff).into();

    6
}

pub fn path_render2(vertices: &mut [Vertex]) -> usize {
    vertices[0] = ([-1.0, -1.0], 0xff00ffff).into();
    vertices[1] = ([-1.0, 0.0], 0x00ffffff).into();
    vertices[2] = ([ 0.0, -1.0], 0x007f00ff).into();

    vertices[3] = ([0.0, 0.0], [0.5, 0.0], 0xff00ffff).into();
    vertices[4] = ([0.0, -1.0], [1.0, 1.0], 0x00ffffff).into();
    vertices[5] = ([-1.0, 0.0], [0.0, 0.0], 0x00ff00ff).into();

    6
}

pub fn write_buffer(queue: &wgpu::Queue, vertices: &mut VertexBuffer, len: usize) {
    queue.write_buffer(vertices.buffer(), 0, bytemuck::cast_slice(vertices.as_slice()));
}

impl From<([f32; 2], u32)> for Vertex {
    fn from(value: ([f32; 2], u32)) -> Self {
        Vertex { position: value.0, tex_coord: [0., 0.], color: value.1 }
    }
}
impl From<([f32; 2], [f32; 2], u32)> for Vertex {
    fn from(value: ([f32; 2], [f32; 2], u32)) -> Self {
        Vertex { position: value.0, tex_coord: value.1, color: value.2 }
    }
}