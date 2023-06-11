use wgpu::util::DeviceExt;

use crate::{graph::{Affine2d, Point}, artist::Color};

pub struct Shape2dRender {
    vertex_stride: usize,
    vertex_vec: Vec<Shape2dVertex>,
    vertex_buffer: wgpu::Buffer,
    vertex_offset: usize,

    style_stride: usize,
    style_vec: Vec<Shape2dStyle>,
    style_buffer: wgpu::Buffer,
    style_offset: usize,

    shape_items: Vec<Shape2dItem>,

    pipeline: wgpu::RenderPipeline,
}

impl Shape2dRender {
    pub(crate) fn new(
        device: &wgpu::Device, 
        format: wgpu::TextureFormat,
    ) -> Self {
        let len = 2048;

        let mut vertex_vec = Vec::<Shape2dVertex>::new();
        vertex_vec.resize(len, Shape2dVertex { position: [0.0, 0.0] });

        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(vertex_vec.as_slice()),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );

        let mut style_vec = Vec::<Shape2dStyle>::new();
        style_vec.resize(len, Shape2dStyle { 
            affine_0: [0.0, 0.0, 0.0, 0.0], 
            affine_1: [0.0, 0.0, 0.0, 0.0], 
            color: [0.0, 0.0, 0.0, 0.0],
        });

        let style_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(style_vec.as_slice()),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );

        let shader = device.create_shader_module(wgpu::include_wgsl!("shape2d.wgsl"));

        let pipeline = create_shape2d_pipeline(
            device, 
            &shader,
            "vs_shape",
            "fs_shape",
            format,
            Shape2dVertex::desc(),
            Shape2dStyle::desc(),
        );
    
        Self {
            vertex_stride: std::mem::size_of::<Shape2dVertex>(),
            vertex_vec,
            vertex_buffer,
            vertex_offset: 0,

            style_stride: std::mem::size_of::<Shape2dStyle>(),
            style_vec,
            style_buffer,
            style_offset: 0,
            // style_bind_group,

            shape_items: Vec::new(),
            pipeline,
        }
    }

    pub fn start_shape(&mut self) {
        let start = self.vertex_offset;

        self.shape_items.push(Shape2dItem {
            v_start: start,
            v_end: usize::MAX,
            s_start: self.style_offset,
            s_end: usize::MAX,
        });
    }

    pub(crate) fn draw_line(&mut self, 
        p0: &Point,
        p1: &Point,
        lw_x: f32, lw_y: f32,
    ) {
        let Point(x0, y0) = p0;
        let Point(x1, y1) = p1;

        let dx = x1 - x0;
        let dy = y1 - y0;

        let len = dx.hypot(dy).max(f32::EPSILON);

        // normal to the line
        let nx = dy * lw_x / len;
        let ny = - dx * lw_y / len;

        self.vertex(x0 - nx, y0 - ny);
        self.vertex(x0 + nx, y0 + ny);
        self.vertex(x1 + nx, y1 + ny);

        self.vertex(x1 + nx, y1 + ny);
        self.vertex(x1 - nx, y1 - ny);
        self.vertex(x0 - nx, y0 - ny);
    }

    pub fn draw_style(
        &mut self, 
        color: Color,
        affine: Affine2d,
    ) {
        let end = self.vertex_offset;

        let len = self.shape_items.len();
        let item = &mut self.shape_items[len - 1];
        item.v_end = end;

        self.style_vec[self.style_offset] = Shape2dStyle::new(&affine, color.get_srgba());
        self.style_offset += 1;

        item.s_end = self.style_offset;
    }

    pub fn flush(
        &mut self, 
        queue: &wgpu::Queue, 
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        if self.shape_items.len() == 0 {
            return;
        }

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                }
            })],
            depth_stencil_attachment: None,
        });

        queue.write_buffer(
            &mut self.vertex_buffer, 
            0,
            bytemuck::cast_slice(self.vertex_vec.as_slice())
        );

        queue.write_buffer(
            &mut self.style_buffer, 
            0,
            bytemuck::cast_slice(self.style_vec.as_slice())
        );

        rpass.set_pipeline(&self.pipeline);

        for item in self.shape_items.drain(..) {
            let stride = self.vertex_stride;
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(
                (stride * item.v_start) as u64..(stride * item.v_end) as u64
            ));
            let stride = self.style_stride;
            rpass.set_vertex_buffer(1, self.style_buffer.slice(
                (stride * item.s_start) as u64..(stride * item.s_end) as u64
            ));

            if item.v_start < item.v_end {
                rpass.draw(
                    0..(item.v_end - item.v_start) as u32,
                    0..(item.s_end - item.s_start) as u32,
                );
            }
        }

        self.vertex_offset = 0;
    }

    fn vertex(&mut self, x: f32, y: f32) {
        let vertex = Shape2dVertex { position: [x, y] };

        self.vertex_vec[self.vertex_offset] = vertex;
        self.vertex_offset += 1;
    }
}

pub struct Shape2dItem {
    v_start: usize,
    v_end: usize,

    s_start: usize,
    s_end: usize,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Shape2dVertex {
    position: [f32; 2],
}

impl Shape2dVertex {
    const ATTRS: [wgpu::VertexAttribute; 1] =
        wgpu::vertex_attr_array![0 => Float32x2 ];

    pub(crate) fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Shape2dVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }
}

pub struct Shape2dBuffer {
    stride: usize,
    vec: Vec<Shape2dVertex>,
    buffer: wgpu::Buffer,

    offset: usize,
}

impl Shape2dBuffer {
    pub(crate) fn new(len: usize, device: &wgpu::Device) -> Self {
        let mut vec = Vec::<Shape2dVertex>::new();
        vec.resize(len, Shape2dVertex { position: [0.0, 0.0], });

        let buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(vec.as_slice()),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );
    
        Self {
            stride: std::mem::size_of::<Shape2dVertex>(),
            vec,
            buffer,
            offset: 0,
        }
    }

    pub(crate) fn stride(&self) -> usize {
        self.stride
    }

    pub(crate) fn as_slice(&self) -> &[Shape2dVertex] {
        &self.vec
    }

    pub(crate) fn _as_mut_slice(&mut self) -> &mut [Shape2dVertex] {
        &mut self.vec
    }

    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub(crate) fn desc(&self) -> wgpu::VertexBufferLayout {
        Shape2dVertex::desc()
    }

    pub(crate) fn clear(&mut self) {
        self.offset = 0
    }

    pub(crate) fn offset(&self) -> usize {
        self.offset
    }

    pub(crate) fn push(&mut self, x: f32, y: f32) {
        self.vec[self.offset] = Shape2dVertex { position: [x, y] };
        self.offset += 1;
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Shape2dStyle {
    affine_0: [f32; 4],
    affine_1: [f32; 4],
    color: [f32; 4],
}

impl Shape2dStyle {
    const ATTRS: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![
            1 => Float32x4, 
            2 => Float32x4,
            3 => Float32x4
        ];

    pub(crate) fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Shape2dVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRS,
        }
    }

    fn empty() -> Self {
        Self::new(&Affine2d::eye(), 0x000000ff)
    }

    fn new(affine: &Affine2d, color: u32) -> Self {
        let mat = affine.mat();

        Self {
            affine_0: [mat[0], mat[1], 0., mat[2]],
            affine_1: [mat[3], mat[4], 0., mat[5]],
            color: [
                ((color >> 24) & 0xff) as f32 / 255.,
                ((color >> 16) & 0xff) as f32 / 255.,
                ((color >> 8) & 0xff) as f32 / 255.,
                ((color) & 0xff) as f32 / 255.,
            ],
        }
    }

    fn create_buffer(this: Self, device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("shape2d style"),
                contents: bytemuck::cast_slice(&[this]),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        )
    }
}

fn create_shape2d_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    vertex_entry: &str,
    fragment_entry: &str,
    format: wgpu::TextureFormat,
    vertex_layout: wgpu::VertexBufferLayout,
    style_layout: wgpu::VertexBufferLayout,
) -> wgpu::RenderPipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[
        ],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: vertex_entry,
            buffers: &[
                vertex_layout,
                style_layout,
            ],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: fragment_entry,
            targets: &[
                Some(wgpu::ColorTargetState {
                    format,

                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add
                        },

                        alpha: wgpu::BlendComponent::OVER
                    }),

                    write_mask: wgpu::ColorWrites::ALL,
                })
            ],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    })
}
