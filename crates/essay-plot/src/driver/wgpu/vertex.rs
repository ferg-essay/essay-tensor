use wgpu::util::DeviceExt;


#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    position: [f32; 2],
    uv_coord: [f32; 2],
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

    offset: usize,
}

impl VertexBuffer {
    pub(crate) fn new(len: usize, device: &wgpu::Device) -> Self {
        let mut vec = Vec::<Vertex>::new();
        vec.resize(len, Vertex { position: [0.0, 0.0], uv_coord: [0.0, 0.0], color: 0x00000000 });

        let buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(vec.as_slice()),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );
    
        Self {
            stride: std::mem::size_of::<Vertex>(),
            vec,
            buffer,
            offset: 0,
        }
    }

    pub(crate) fn stride(&self) -> usize {
        self.stride
    }

    pub(crate) fn as_slice(&self) -> &[Vertex] {
        &self.vec
    }

    pub(crate) fn _as_mut_slice(&mut self) -> &mut [Vertex] {
        &mut self.vec
    }

    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub(crate) fn desc(&self) -> wgpu::VertexBufferLayout {
        Vertex::desc()
    }

    pub(crate) fn clear(&mut self) {
        self.offset = 0
    }

    pub(crate) fn offset(&self) -> usize {
        self.offset
    }

    pub(crate) fn push(&mut self, x: f32, y: f32, color: u32) {
        self.vec[self.offset] = ([x, y], color).into();
        self.offset += 1;
    }

    pub(crate) fn push_tex(&mut self, x: f32, y: f32, u: f32, v: f32, color: u32) {
        self.vec[self.offset] = ([x, y], [u, v], color).into();
        self.offset += 1;
    }
}

impl From<([f32; 2], u32)> for Vertex {
    fn from(value: ([f32; 2], u32)) -> Self {
        Vertex { position: value.0, uv_coord: [0., 0.], color: value.1 }
    }
}
impl From<([f32; 2], [f32; 2], u32)> for Vertex {
    fn from(value: ([f32; 2], [f32; 2], u32)) -> Self {
        Vertex { position: value.0, uv_coord: value.1, color: value.2 }
    }
}
