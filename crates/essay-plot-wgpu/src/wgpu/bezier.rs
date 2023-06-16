use essay_plot_base::{Point, Color, Affine2d};
use wgpu::util::DeviceExt;

pub struct BezierRender {
    vertex_stride: usize,
    vertex_vec: Vec<BezierVertex>,
    vertex_buffer: wgpu::Buffer,
    vertex_offset: usize,

    style_stride: usize,
    style_vec: Vec<BezierStyle>,
    style_buffer: wgpu::Buffer,
    style_offset: usize,

    shape_items: Vec<BezierItem>,

    pipeline: wgpu::RenderPipeline,
}

impl BezierRender {
    pub(crate) fn new(
        device: &wgpu::Device, 
        format: wgpu::TextureFormat,
    ) -> Self {
        let len = 2048;

        let mut vertex_vec = Vec::<BezierVertex>::new();
        vertex_vec.resize(len, BezierVertex { 
            position: [0.0, 0.0],
            uv: [0., 0.]
        });

        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(vertex_vec.as_slice()),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );

        let mut style_vec = Vec::<BezierStyle>::new();
        style_vec.resize(len, BezierStyle { 
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

        let pipeline = create_bezier_pipeline(
            device, 
            format,
        );
    
        Self {
            vertex_stride: std::mem::size_of::<BezierVertex>(),
            vertex_vec,
            vertex_buffer,
            vertex_offset: 0,

            style_stride: std::mem::size_of::<BezierStyle>(),
            style_vec,
            style_buffer,
            style_offset: 0,
            // style_bind_group,

            shape_items: Vec::new(),
            pipeline,
        }
    }

    pub fn clear(&mut self) {
        self.shape_items.drain(..);
        self.vertex_offset = 0;
        self.style_offset = 0;
    }

    pub fn start_shape(&mut self) {
        let start = self.vertex_offset;

        self.shape_items.push(BezierItem {
            v_start: start,
            v_end: usize::MAX,
            s_start: self.style_offset,
            s_end: usize::MAX,
        });
    }

    pub(crate) fn draw_line(
        &mut self, 
        p0: &Point,
        p1: &Point,
        lw: f32,
    ) {
        let Point(x0, y0) = p0;
        let Point(x1, y1) = p1;

        let dx = p1.x() - p0.x();
        let dy = p1.y() - p0.y();

        let len = dx.hypot(dy).max(f32::EPSILON);

        let dx = dx / len;
        let dy = dy / len;

        // normal to the line
        let nx = dy * lw;
        let ny = - dx * lw;

        // TODO: incorrect extend
        let dx2 = dx * lw; // for extend
        let dy2 = dy * lw;

        self.vertex(x0 - nx - dx2, y0 - ny - dy2);
        self.vertex(x0 + nx - dx2, y0 + ny - dy2);
        self.vertex(x1 + nx + dx2, y1 + ny + dy2);

        self.vertex(x1 + nx + dx2, y1 + ny + dy2);
        self.vertex(x1 - nx + dx2, y1 - ny + dy2);
        self.vertex(x0 - nx - dx2, y0 - ny - dy2);
    }

    pub(crate) fn draw_bezier_line(
        &mut self, 
        p0: &Point,
        p1: &Point,
        p2: &Point,
        lw: f32,
    ) {
        //self.vertex_buffer.push(p0.x(), p0.y(), 0x000000ff);
        //self.vertex_buffer.push(p1.x(), p1.y(), 0x000000ff);
        //self.vertex_buffer.push(p2.x(), p2.y(), 0x0000000ff);

        //self.bezier_vertex.push_tex(p0.x(), p0.y(), -1.0,0.0, color);
        //self.bezier_vertex.push_tex(p1.x(), p1.y(), 0.0, 2.0, color);
        //self.bezier_vertex.push_tex(p2.x(), p2.y(), 1.0, 0.0, color);

        self.vertex_bezier(p0.x(), p0.y(), -1.0,1.0);
        self.vertex_bezier(p1.x(), p1.y(), 0.0, -1.0);
        self.vertex_bezier(p2.x(), p2.y(), 1.0, 1.0);

        //self.bezier_rev_vertex.push_tex(p0.x(), p0.y(), -1.0,1.0, color);
        //self.bezier_rev_vertex.push_tex(p1.x(), p1.y(), 0.0, -1.0, color);
        //self.bezier_rev_vertex.push_tex(p2.x(), p2.y(), 1.0, 1.0, color);
    }

    pub(crate) fn draw_bezier_fill(
        &mut self, 
        p0: &Point,
        p1: &Point,
        p2: &Point,
    ) {
        self.vertex_bezier(p0.x(), p0.y(), -1.0,1.0);
        self.vertex_bezier(p1.x(), p1.y(), 0.0, -1.0);
        self.vertex_bezier(p2.x(), p2.y(), 1.0, 1.0);
    }

    pub fn draw_style(
        &mut self, 
        color: Color,
        affine: &Affine2d,
    ) {
        let end = self.vertex_offset;

        let len = self.shape_items.len();
        let item = &mut self.shape_items[len - 1];
        item.v_end = end;

        self.style_vec[self.style_offset] = BezierStyle::new(affine, color);
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
            if item.v_start < item.v_end && item.s_start < item.s_end {
                let stride = self.vertex_stride;
                rpass.set_vertex_buffer(0, self.vertex_buffer.slice(
                    (stride * item.v_start) as u64..(stride * item.v_end) as u64
                ));

                let stride = self.style_stride;
                rpass.set_vertex_buffer(1, self.style_buffer.slice(
                    (stride * item.s_start) as u64..(stride * item.s_end) as u64
                ));

                rpass.draw(
                    0..(item.v_end - item.v_start) as u32,
                    0..(item.s_end - item.s_start) as u32,
                );
            }
        }

        self.vertex_offset = 0;
    }

    fn vertex(&mut self, x: f32, y: f32) {
        let (u, v) = (0., 0.);
        let vertex = BezierVertex { position: [x, y], uv: [u, v] };

        self.vertex_vec[self.vertex_offset] = vertex;
        self.vertex_offset += 1;
    }

    fn vertex_bezier(&mut self, x: f32, y: f32, u: f32, v: f32) {
        let vertex = BezierVertex { position: [x, y], uv: [u, v] };

        self.vertex_vec[self.vertex_offset] = vertex;
        self.vertex_offset += 1;
    }
}

pub struct BezierItem {
    v_start: usize,
    v_end: usize,

    s_start: usize,
    s_end: usize,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BezierVertex {
    position: [f32; 2],
    uv: [f32; 2],
}

impl BezierVertex {
    const ATTRS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2 ];

    pub(crate) fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<BezierVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BezierStyle {
    affine_0: [f32; 4],
    affine_1: [f32; 4],
    color: [f32; 4],
}

impl BezierStyle {
    const ATTRS: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![
            2 => Float32x4, 
            3 => Float32x4,
            4 => Float32x4
        ];

    pub(crate) fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<BezierStyle>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRS,
        }
    }

    fn new(affine: &Affine2d, color: Color) -> Self {
        let mat = affine.mat();

        Self {
            affine_0: [mat[0], mat[1], 0., mat[2]],
            affine_1: [mat[3], mat[4], 0., mat[5]],
            color: [
                Color::to_srgb(color.red()),
                Color::to_srgb(color.green()),
                Color::to_srgb(color.blue()),
                color.alpha(),
            ],
        }
    }
}

fn create_bezier_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("bezier.wgsl"));

    let vertex_entry = "vs_bezier";
    let fragment_entry = "fs_bezier";

    let vertex_layout = BezierVertex::desc();
    let style_layout = BezierStyle::desc();

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
