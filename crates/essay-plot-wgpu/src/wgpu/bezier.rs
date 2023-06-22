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
            uv: [0., 0., 0.]
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
            v_end: start,
            s_start: self.style_offset,
            s_end: self.style_offset,
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
        b0: &Point,
        b1: &Point,
        b2: &Point,
        lw2: f32,
    ) {
        let dx = b2.x() - b0.x();
        let dy = b2.y() - b0.y();

        let len = dx.hypot(dy).max(f32::EPSILON);

        let min_bezier = 3.0;

        if len <= min_bezier {
            self.draw_line(b0, b2, lw2);
            return;
        }

        let dx = dx / len;
        let dy = dy / len;

        // normal to the line
        let mut nx = dy * lw2;
        let mut ny = dx * lw2;

        let ccw = 
            (b1.x() - b0.x()) * (b2.y() - b0.y())
            - (b2.x() - b0.x()) * (b1.y() - b0.y());

        let min_bezier_area = 1.;
        if ccw.abs() < min_bezier_area {
            self.draw_line(b0, b2, lw2);
            return;
        } 
        if ccw < 0. {
            (nx, ny) = (-nx, -ny);
        }

        // outer bezier's points
        let p0 = Point(b0.x() + nx, b0.y() - ny);
        let p1 = Point(b1.x() + nx, b1.y() - ny);
        let p2 = Point(b2.x() + nx, b2.y() - ny);

        // inner bezier's points
        let q0 = Point(b0.x() - nx, b0.y() + ny);
        let q1 = Point(b1.x() - nx, b1.y() + ny);
        let q2 = Point(b2.x() - nx, b2.y() + ny);

        // if inner bezier point is inside line's rectangle
        // (0 < AM dot AB < AB dot AB) && (0 < AM dot AD < AD dot AD)

        let Point(a_x, a_y) = q2;
        let Point(b_x, b_y) = q0;
        let Point(d_x, d_y) = p2;
        
        let Point(m_x, m_y) = q1;

        let am_ab = (a_x - m_x) * (a_x - b_x) + (a_y - m_y) * (a_y - b_y);
        let ab_ab = (a_x - b_x).powi(2) + (a_y - b_y).powi(2);
        let am_ad = (a_x - m_x) * (a_x - d_x) + (a_y - m_y) * (a_y - d_y);
        let ad_ad = (a_x - d_x).powi(2) + (a_y - d_y).powi(2);

        if 0. <= am_ab && am_ab <= ab_ab && 0. <= am_ad && am_ad <= ad_ad {
            // inner bezier's control point inside line's rectangle
            self.vertex(q0.x(), q0.y()); // inner q0
            self.vertex(p0.x(), p0.y()); // outer p0
            self.vertex(q1.x(), q1.y()); // inner q1

            self.vertex(p0.x(), p0.y()); // outer p0
            self.vertex(p2.x(), p2.y()); // outer p2
            self.vertex(q1.x(), q1.y()); // inner q1

            self.vertex(p2.x(), p2.y()); // outer p2
            self.vertex(q2.x(), q2.y()); // inner q2
            self.vertex(q1.x(), q1.y()); // inner q1

            // outer bezier
            self.vertex_bezier(p0.x(), p0.y(), -1.0,  1.0, 0.);
            self.vertex_bezier(p1.x(), p1.y(),  0.0, -1.0, 0.);
            self.vertex_bezier(p2.x(), p2.y(),  1.0,  1.0, 0.);
            // inner bezier
            self.vertex_bezier(q0.x(), q0.y(), -1.0, 1.0, 1.);
            self.vertex_bezier(q1.x(), q1.y(), 0.0, 1.0, -1.);
            self.vertex_bezier(q2.x(), q2.y(), 1.0, 1.0, 1.);
        } else {
            // intersection of outer base p0 to p2 with inner q0 to q1
            let mp = intersection(p0, p2, q0, q1);

            if mp != p0 { // non-zero
                self.vertex(q0.x(), q0.y());
                self.vertex(p0.x(), p0.y());
                self.vertex(mp.x(), mp.y());
            }

            // intersection of outer base p0 to p2 with inner q1 to q2
            let mp = intersection(p0, p2, q1, q2);

            if mp != p0 { // non-zero
                self.vertex(q2.x(), q2.y());
                self.vertex(p2.x(), p2.y());
                self.vertex(mp.x(), mp.y());
            }

            // height of p1 from p0 to p2 line 
            let v_height = vertex_height(p0, p1, p2);
            // linewidth in uv coordinates
            let v_factor = lw2 / v_height;

            // outer bezier
            self.vertex_bezier(p0.x(), p0.y(), -1.0, 1., 1.0 - v_factor);
            self.vertex_bezier(p1.x(), p1.y(), 0.0, -1., -1.0 - v_factor);
            self.vertex_bezier(p2.x(), p2.y(), 1.0, 1., 1.0 - v_factor);
            // inner bezier
            self.vertex_bezier(q0.x(), q0.y(), -1.0, 1. + v_factor, 1.);
            self.vertex_bezier(q1.x(), q1.y(), 0.0, -1. + v_factor, -1.);
            self.vertex_bezier(q2.x(), q2.y(), 1.0, 1. + v_factor, 1.);
        }
    }

        //self.vertex_buffer.push(p0.x(), p0.y(), 0x000000ff);
        //self.vertex_buffer.push(p1.x(), p1.y(), 0x000000ff);
        //self.vertex_buffer.push(p2.x(), p2.y(), 0x0000000ff);

        //self.bezier_vertex.push_tex(p0.x(), p0.y(), -1.0,0.0, color);
        //self.bezier_vertex.push_tex(p1.x(), p1.y(), 0.0, 2.0, color);
        //self.bezier_vertex.push_tex(p2.x(), p2.y(), 1.0, 0.0, color);

        //self.vertex_bezier(p0.x(), p0.y(), -1.0,1.0);
        //self.vertex_bezier(p1.x(), p1.y(), 0.0, -1.0);
        //self.vertex_bezier(p2.x(), p2.y(), 1.0, 1.0);

        //self.bezier_rev_vertex.push_tex(p0.x(), p0.y(), -1.0,1.0, color);
        //self.bezier_rev_vertex.push_tex(p1.x(), p1.y(), 0.0, -1.0, color);
        //self.bezier_rev_vertex.push_tex(p2.x(), p2.y(), 1.0, 1.0, color);

    pub(crate) fn draw_bezier_fill(
        &mut self, 
        p0: &Point,
        p1: &Point,
        p2: &Point,
    ) {
        self.vertex_bezier(p0.x(), p0.y(), -1.0,1., 0.);
        self.vertex_bezier(p1.x(), p1.y(), 0.0, -1., 0.);
        self.vertex_bezier(p2.x(), p2.y(), 1.0, 1., 0.);
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
        let vertex = BezierVertex { position: [x, y], uv: [0., 0., 0.] };

        self.vertex_vec[self.vertex_offset] = vertex;
        self.vertex_offset += 1;
    }

    fn vertex_bezier(&mut self, x: f32, y: f32, u: f32, v: f32, w: f32) {
        let vertex = BezierVertex { position: [x, y], uv: [u, v, w] };

        self.vertex_vec[self.vertex_offset] = vertex;
        self.vertex_offset += 1;
    }
}

fn intersection(p0: Point, p1: Point, q0: Point, q1: Point) -> Point {
    let det = (p0.x() - p1.x()) * (q0.y() - q1.y())
        - (p0.y() - p1.y()) * (q0.x() - q1.x());

    if det.abs() <= f32::EPSILON {
        return p0; // p0 is marker for coincident or parallel lines
    }

    let p_xy = p0.x() * p1.y() - p0.y() * p1.x();
    let q_xy = q0.x() * q1.y() - q0.y() * q1.x();

    let x = (p_xy * (q0.x() - q1.x()) - (p0.x() - p1.x()) * q_xy) / det;
    let y = (p_xy * (q0.y() - q1.y()) - (p0.y() - p1.y()) * q_xy) / det;

    Point(x, y)
}

fn vertex_height(p0: Point, p1: Point, p2: Point) -> f32 {
    let a = p0.dist(&p1);
    let b = p1.dist(&p2);
    let c = p2.dist(&p0);
    // Heron's formula
    let s = 0.5 * (a + b + c);
    let area = (s * (s - a) * (s - b) * (s - c)).sqrt();

    0.5 * area / c
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
    uv: [f32; 3],
}

impl BezierVertex {
    const ATTRS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x3 ];

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
                Color::srgb_to_lrgb(color.red()),
                Color::srgb_to_lrgb(color.green()),
                Color::srgb_to_lrgb(color.blue()),
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
