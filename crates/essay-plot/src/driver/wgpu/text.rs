use wgpu::util::DeviceExt;
use wgpu_glyph::ab_glyph::{self, Font, PxScale};

use crate::frame::Affine2d;

use super::{text_texture::TextTexture, text_cache::TextCache};

pub struct TextRender {
    texture: TextTexture,
    text_cache: TextCache,

    vertex_stride: usize,
    vertex_vec: Vec<TextVertex>,
    vertex_buffer: wgpu::Buffer,

    vertex_offset: usize,

    style_buffer: wgpu::Buffer,
    style_bind_group: wgpu::BindGroup,

    text_items: Vec<TextItem>,

    pipeline: wgpu::RenderPipeline,
}

impl TextRender {
    pub(crate) fn new(
        device: &wgpu::Device, 
        format: wgpu::TextureFormat,
        width: u32, 
        height: u32
    ) -> Self {
        let len = 2048;

        let mut vec = Vec::<TextVertex>::new();
        vec.resize(len, TextVertex { position: [0.0, 0.0], tex_coord: [0.0, 0.0] });

        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(vec.as_slice()),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );

        let texture = TextTexture::new(device, width, height);

        let text_shader = device.create_shader_module(wgpu::include_wgsl!("text.wgsl"));

        let style_buffer = TextStyle::create_buffer(TextStyle::empty(), device);

        let (style_layout, style_bind_group) = create_uniform_bind_group(device, &style_buffer);

        let pipeline = create_text_pipeline(
            device, 
            &text_shader,
            "vs_text",
            "fs_text",
            format,
            TextVertex::desc(),
            style_layout,
            &texture,
        );
    
        Self {
            texture: TextTexture::new(device, width, height),
            text_cache: TextCache::new(width, height),

            vertex_stride: std::mem::size_of::<TextVertex>(),
            vertex_vec: vec,
            vertex_buffer,
            vertex_offset: 0,

            style_buffer,
            style_bind_group,

            text_items: Vec::new(),
            pipeline,
        }
    }

    pub fn draw(
        &mut self, 
        text: &str, 
        font_name: &str, 
        size: f32
    ) {
        let font = self.text_cache.font(font_name, size as u16);
        let mut x = 0.;
        let y = 0.;

        let start = self.vertex_offset;

        let r = self.text_cache.glyph(&font, 'H');

        if r.is_none() {
            println!("Empty glyph");
            return;
        }

        self.vertex(x, y, r.xmin(), r.ymin());
        self.vertex(x + r.w(), y, r.xmax(), r.ymin());
        self.vertex(x + r.w(), y + r.h(), r.xmax(), r.ymax());

        self.vertex(x + r.w(), y + r.h(), r.xmax(), r.ymax());
        self.vertex(x, y + r.h(), r.xmin(), r.ymax());
        self.vertex(x, y, r.xmin(), r.ymin());

        x += r.w();

        let end = self.vertex_offset;
        let affine = Affine2d::eye().scale(1600.0f32.recip(), 800.0f32.recip());
        self.text_items.push(TextItem {
            style: TextStyle::new(&affine, 0xff8000ff),
            start,
            end
        })
    }

    pub fn flush(
        &mut self, 
        queue: &wgpu::Queue, 
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        self.text_cache.flush(queue, &self.texture);

        if self.text_items.len() == 0 {
            return;
        }

        println!("Flush");
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

        for item in self.text_items.drain(..) {
            rpass.set_pipeline(&self.pipeline);
            let len = item.end - item.start;
            let stride = self.vertex_stride;
            let size = len * self.vertex_stride;
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(
                (stride * item.start) as u64..(stride * item.end) as u64
            ));
            rpass.set_bind_group(0, self.texture.bind_group(), &[]);
            rpass.set_bind_group(1, &self.style_bind_group, &[]);
            rpass.draw(0..len as u32, 0..1);
            println!("Item {}, {}", item.start, item.end);
        }

        /*
        let vertex_len = self.vertex.offset();
        if vertex_len > 0 {
            write_buffer(queue, &mut self.vertex, vertex_len);

            rpass.set_pipeline(&self.vertex_pipeline);

            let size = (vertex_len * self.vertex.stride()) as u64;
            rpass.set_vertex_buffer(0, self.vertex.buffer().slice(..size));
    
            rpass.draw(0..vertex_len as u32, 0..1);
        }
        */
    }

    fn vertex(&mut self, x: f32, y: f32, u: f32, v: f32) {
        let vertex = TextVertex::new(x, y, u, v);

        self.vertex_vec[self.vertex_offset] = vertex;
        self.vertex_offset += 1;
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TextVertex {
    position: [f32; 2],
    tex_coord: [f32; 2],
}

impl TextVertex {
    const ATTRS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2 ];

    pub(crate) fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<TextVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }

    fn new(x: f32, y: f32, u: f32, v: f32) -> Self {
        Self {
            position: [x, y],
            tex_coord: [u, v],
        }
    }
}

pub struct TextItem {
    style: TextStyle,
    start: usize,
    end: usize,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TextStyle {
    affine_0: [f32; 3],
    _padding: u32,
    affine_1: [f32; 3],
    _padding2: u32,
    color: [f32; 4],
}

impl TextStyle {
    const ATTRS: [wgpu::VertexAttribute; 5] =
        wgpu::vertex_attr_array![
            0 => Float32x3, 
            1 => Uint32,
            2 => Float32x3,
            3 => Uint32,
            4 => Float32x4
        ];

    pub(crate) fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<TextVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }

    fn empty() -> Self {
        Self::new(&Affine2d::eye(), 0x000000ff)
    }

    fn new(affine: &Affine2d, color: u32) -> Self {
        let mat = affine.mat();
        Self {
            affine_0: [mat[0], mat[1], mat[2]],
            affine_1: [mat[3], mat[4], mat[5]],
            color: [
                ((color >> 24) & 0xff) as f32 / 255.,
                ((color >> 16) & 0xff) as f32 / 255.,
                ((color >> 8) & 0xff) as f32 / 255.,
                ((color) & 0xff) as f32 / 255.,
            ],
            _padding: 0,
            _padding2: 0,
        }
    }

    fn create_buffer(this: Self, device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("text style"),
                contents: bytemuck::cast_slice(&[this]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        )
    }
}

fn create_text_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    // pipeline_layout: &wgpu::PipelineLayout,
    vertex_entry: &str,
    fragment_entry: &str,
    format: wgpu::TextureFormat,
    vertex_layout: wgpu::VertexBufferLayout,
    style_layout: wgpu::BindGroupLayout,
    texture: &TextTexture,
) -> wgpu::RenderPipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[
            texture.layout(),
            &style_layout,
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

fn create_uniform_bind_group(
    device: &wgpu::Device, 
    buffer: &wgpu::Buffer
) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
        label: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
        label: None
    });

    (layout, bind_group)
}
