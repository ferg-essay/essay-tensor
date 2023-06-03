use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    position: [f32; 2],
    color: u32,
}

impl Vertex {
    const ATTRS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x2, 1=> Uint32 ];

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
        vec.resize(len, Vertex { position: [0.0, 0.5], color: 0x00000000 });
        println!("Bufzi {:?}", vec.as_slice().len());

        path_render2(vec.as_mut_slice());

        let buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(vec.as_slice()),
                usage: wgpu::BufferUsages::VERTEX,
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

    pub(crate) fn as_slice(&mut self) -> &mut [Vertex] {
        &mut self.vec
    }

    pub(crate) fn buffer(&mut self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub(crate) fn desc(&self) -> wgpu::VertexBufferLayout {
        Vertex::desc()
    }
}

pub fn path_render(vertices: &mut [Vertex]) -> usize {
    vertices[0] = ([0.0, 0.0], 0xff00ffff).into();
    vertices[1] = ([0.0, 1.0], 0xffffffff).into();
    vertices[2] = ([1.0, 0.0], 0xff00ffff).into();

    3
}

pub fn path_render2(vertices: &mut [Vertex]) -> usize {
    vertices[0] = ([0.0, 0.0], 0xff00ffff).into();
    vertices[1] = ([0.0, -1.0], 0x00ffffff).into();
    vertices[2] = ([-1.0, 0.0], 0x00ff00ff).into();

    3
}

impl From<([f32; 2], u32)> for Vertex {
    fn from(value: ([f32; 2], u32)) -> Self {
        Vertex { position: value.0, color: value.1 }
    }
}