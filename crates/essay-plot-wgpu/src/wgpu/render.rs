use essay_plot_base::{Canvas, Affine2d, Point, Bounds, Path, StyleOpt, Color, PathCode, driver::{RenderErr, Renderer, FigureApi}, TextStyle, Coord, WidthAlign, HeightAlign, JoinStyle, CapStyle};
use essay_tensor::Tensor;

use super::{vertex::VertexBuffer, text::{TextRender}, shape2d::Shape2dRender, tesselate::tesselate, triangle2d::GridMesh2dRender, bezier::BezierRender};

pub struct FigureRenderer {
    canvas: Canvas,

    shape2d_render: Shape2dRender,
    text_render: TextRender,
    triangle_render: GridMesh2dRender,
    bezier_render: BezierRender,

    to_gpu: Affine2d,

    is_request_redraw: bool,
}

impl<'a> FigureRenderer {
    pub(crate) fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> Self {
    
        let shape2d_render = Shape2dRender::new(device, format);
        let text_render = TextRender::new(device, format, 512, 512);
        let triangle_render = GridMesh2dRender::new(device, format);
        let bezier_render = BezierRender::new(device, format);
        
        Self {
            canvas: Canvas::new((), 1.),

            shape2d_render,
            text_render,
            triangle_render,
            bezier_render,

            to_gpu: Affine2d::eye(),

            is_request_redraw: false,
        }
    }

    pub(crate) fn is_request_redraw(&self) -> bool {
        self.is_request_redraw
    }

    pub(crate) fn clear(&mut self) {
        self.is_request_redraw = false;

        self.bezier_render.clear();
        self.text_render.clear();
        self.shape2d_render.clear();
        self.triangle_render.clear();
    }

    pub(crate) fn set_canvas_bounds(&mut self, width: u32, height: u32) {
        self.canvas.set_bounds([width as f32, height as f32]);

        let pos_gpu = Bounds::<Canvas>::new(
            Point(-1., -1.),
            Point(1., 1.)
        );

        self.to_gpu = self.canvas.bounds().affine_to(&pos_gpu);
    }

    pub(crate) fn set_scale_factor(&mut self, scale_factor: f32) {
        self.canvas.set_scale_factor(scale_factor);
    }

    fn draw_closed_path(
        &mut self, 
        path: &Path<Canvas>, 
        _clip: &Bounds<Canvas>,
    ) {
        self.shape2d_render.start_shape();
        self.bezier_render.start_shape();

        let mut points = Vec::<Point>::new();
        let mut prev = Point(0., 0.);
        for code in path.codes() {
            let last = code.tail();

            points.push(last);

            if let PathCode::Bezier2(p1, p2) = code {
                self.bezier_render.draw_bezier_fill(&last, p1, p2);
            }

            prev = last;
        }

        let triangles = tesselate(points);

        for triangle in &triangles {
            self.shape2d_render.draw_triangle(&triangle[0], &triangle[1], &triangle[2]);
        }
    }

    fn draw_lines(
        &mut self, 
        path: &Path<Canvas>, 
        style: &dyn StyleOpt, 
        _clip: &Bounds<Canvas>,
    ) {
        let linewidth  = match style.get_linewidth() {
            Some(linewidth) => *linewidth,
            None => 0.5,
        };

        let joinstyle  = match style.get_joinstyle() {
            Some(joinstyle) => joinstyle.clone(),
            None => JoinStyle::Bevel,
        };

        let capstyle  = match style.get_capstyle() {
            Some(capstyle) => capstyle.clone(),
            None => CapStyle::Projecting,
        };
        
        let lw2 = self.to_px(0.5 * linewidth); // / self.canvas.width();

        self.shape2d_render.start_shape();
        self.bezier_render.start_shape();

        let mut p0 = Point(0.0f32, 0.0f32);
        let mut p_move = p0;
        let mut p_first = p0;
        let mut p_last = p0;

        for code in path.codes() {
            let p_next = match code {
                PathCode::MoveTo(p) => {
                    self.cap_line(p_last, p0, lw2, &capstyle);
                    
                    p0 = *p;
                    p_move = p0;
                    p_first = p0;
                    p_last = p0;
                    p0
                }
                PathCode::LineTo(p1) => {
                    self.shape2d_render.draw_line(&p0, p1, lw2);
                    // TODO: clip
                    *p1
                }
                PathCode::Bezier2(p1, p2) => {
                    //self.draw_bezier(p0, *p1, *p2, rgba);
                    //self.bezier_render.draw_line(&p0, p2, lw, lw);
                    //self.bezier_render.draw_line(&p0, p2, lw);
                    self.bezier_render.draw_bezier_line(&p0, p1, p2, lw2);

                    *p2
                }
                PathCode::Bezier3(_, _, _) => {
                    panic!("Bezier3 should already be split into Bezier2");
                }
                PathCode::ClosePoly(p1) => {
                    //self.draw_line(p0.x(), p0.y(), p1.x(), p1.y(), lw_x, lw_y, rgba);
                    self.shape2d_render.draw_line(&p0, p1, lw2);
                    self.shape2d_render.draw_line(p1, &p_move, lw2);

                    self.join_lines(p0, *p1, p_move, lw2, &joinstyle);
                    self.join_lines(*p1, p_move, p_first, lw2, &joinstyle);

                    *p1
                }
            };

            self.join_lines(p_last, p0, p_next, lw2, &joinstyle);

            if p_first == p_move {
                p_first = p_next;
                self.cap_line(p_next, p_move, lw2, &capstyle);
            }
            p_last = p0;
            p0 = p_next;
        }
        self.cap_line(p_last, p0, lw2, &capstyle);
    }

    fn join_lines(
        &mut self, 
        b0: Point, 
        b1: Point, 
        b2: Point, 
        lw2: f32, 
        join_style: &JoinStyle
    ) {
        let min_join = 1.;

        if b0 == b1 || b1 == b2 || lw2 < min_join {
            // small lines can ignore joining.
            return;
        }

        let (nx, ny) = line_normal(b0, b1, lw2);

        // outside edge
        let p0 = Point(b0.x() + nx, b0.y() - ny);
        let p1 = Point(b1.x() + nx, b1.y() - ny);

        let (nx, ny) = line_normal(b1, b2, lw2);

        // outside edge
        let q1 = Point(b1.x() + nx, b1.y() - ny);
        let q2 = Point(b2.x() + nx, b2.y() - ny);

        // add bevel triangle
        self.shape2d_render.draw_triangle(&p1, &q1, &b1);

        match join_style {
            JoinStyle::Bevel => {},
            JoinStyle::Miter => {
                // TODO: clamp intersections of too-long length
                let mp = line_intersection(p0, p1, q1, q2);

                if mp != p0 { // non-parallel
                    self.shape2d_render.draw_triangle(&p1, &mp, &q1);
                }
            },
            JoinStyle::Round => {
                let mp = line_intersection(p0, p1, q1, q2);

                if mp != p0 { // non-parallel
                    self.bezier_render.draw_bezier_fill(&p1, &mp, &q1);
                }
            }
        }
    }

    fn cap_line(
        &mut self, 
        b0: Point, 
        b1: Point,
        lw2: f32, 
        cap_style: &CapStyle
    ) {
        if b0 == b1 || cap_style == &CapStyle::Butt {
            // small lines can ignore joining.
            return;
        }

        let (nx, ny) = line_normal(b0, b1, lw2);
        let (dx, dy) = (ny, nx);

        // outside edge
        let p0 = Point(b1.x() + nx, b1.y() - ny);
        // extended edge
        let p1 = Point(b1.x() + nx + dx, b1.y() - ny + dy);

        // inside edge
        let q0 = Point(b1.x() - nx, b1.y() + ny);
        // extended edge
        let q1 = Point(b1.x() - nx + dx, b1.y() + ny + dy);

        let mp = Point(b1.x() + dx, b1.y() + dy);

        match cap_style {
            CapStyle::Round => {
                self.shape2d_render.draw_triangle(&p0, &mp, &q0);
                self.bezier_render.draw_bezier_fill(&p0, &p1, &mp);
                self.bezier_render.draw_bezier_fill(&mp, &q1, &q0);
            }
            CapStyle::Projecting => {
                self.shape2d_render.draw_triangle(&p0, &p1, &q1);
                self.shape2d_render.draw_triangle(&q1, &q0, &p0);
            },
            CapStyle::Butt => {
                panic!()
            }
        }
        // add bevel triangle
        /*
        match join_style {
            JoinStyle::Bevel => {},
            JoinStyle::Mitre => {
                let mp = line_intersection(p0, p1, q1, q2);

                if mp != p0 { // non-parallel
                    self.shape2d_render.draw_triangle(&p1, &mp, &q1);
                }
            },
            JoinStyle::Rounded => {
                let mp = line_intersection(p0, p1, q1, q2);

                if mp != p0 { // non-parallel
                    self.bezier_render.draw_bezier_fill(&p1, &mp, &q1);
                }
            }
        }
        */
    }
}

pub(crate) fn line_normal(
    p0: Point, 
    p1: Point, 
    lw2: f32, 
) -> (f32, f32) {
    let dx = p1.x() - p0.x();
    let dy = p1.y() - p0.y();

    let len = dx.hypot(dy).max(f32::EPSILON);

    let dx = dx / len;
    let dy = dy / len;

    // normal to the line
    let nx = dy * lw2;
    let ny = dx * lw2;

    (nx, ny)
}

pub(crate) fn line_intersection(
    p0: Point, 
    p1: Point, 
    q0: Point, 
    q1: Point
) -> Point {
    let mut det = (p0.x() - p1.x()) * (q0.y() - q1.y())
        - (p0.y() - p1.y()) * (q0.x() - q1.x());

    if det.abs() <= f32::EPSILON {
        return p0; // p0 is marker for coincident or parallel lines
    } else if det.abs() < 0.2 {
        // clamp long extensions for miter
        det = 0.2 * det.signum();
    }


    let p_xy = p0.x() * p1.y() - p0.y() * p1.x();
    let q_xy = q0.x() * q1.y() - q0.y() * q1.x();

    let x = (p_xy * (q0.x() - q1.x()) - (p0.x() - p1.x()) * q_xy) / det;
    let y = (p_xy * (q0.y() - q1.y()) - (p0.y() - p1.y()) * q_xy) / det;

    Point(x, y)
}

impl Renderer for FigureRenderer {
    ///
    /// Returns the boundary of the canvas, usually in pixels or points.
    ///
    fn get_canvas(&self) -> &Canvas {
        &self.canvas
    }

    fn to_px(&self, size: f32) -> f32 {
        self.canvas.to_px(size)
    }

    fn draw_path(
        &mut self, 
        style: &dyn StyleOpt, 
        path: &Path<Canvas>, 
        to_device: &Affine2d,
        clip: &Bounds<Canvas>,
    ) -> Result<(), RenderErr> {
        // let to_unit = self.to_gpu.matmul(to_device);

        let path = transform_path(path);

        let facecolor = match style.get_facecolor() {
            Some(color) => *color,
            None => Color(0x000000ff)
        };

        let edgecolor = match style.get_edgecolor() {
            Some(color) => *color,
            None => Color(0x000000ff)
        };

        if path.is_closed_path() && ! facecolor.is_none() {
            self.draw_closed_path(&path, clip);

            self.shape2d_render.draw_style(facecolor, &self.to_gpu);
            self.bezier_render.draw_style(facecolor, &self.to_gpu);

            if facecolor != edgecolor {
                self.draw_lines(&path, style, clip);

                self.shape2d_render.draw_style(edgecolor, &self.to_gpu);
                self.bezier_render.draw_style(edgecolor, &self.to_gpu);
            }
        } else {
            self.draw_lines(&path, style, clip);

            self.shape2d_render.draw_style(edgecolor, &self.to_gpu);
            self.bezier_render.draw_style(edgecolor, &self.to_gpu);
        }


        return Ok(());
    }

    fn draw_markers(
        &mut self, 
        path: &Path<Canvas>, 
        xy: &Tensor,
        style: &dyn StyleOpt, 
        clip: &Bounds<Canvas>,
    ) -> Result<(), RenderErr> {
        let path = transform_path(path);

        let color = match style.get_facecolor() {
            Some(color) => *color,
            None => Color(0x000000ff)
        };

        if path.is_closed_path() {
            self.draw_closed_path(&path, clip);
        } else {
            self.draw_lines(&path, style, clip);
        }

        for xy in xy.iter_slice() {
            let offset = Affine2d::eye().translate(xy[0], xy[1]);

            self.shape2d_render.draw_style(color, &self.to_gpu.matmul(&offset));
        }

        Ok(())
    }

    fn draw_text(
        &mut self,
        xy: Point, // location in Canvas coordinates
        text: &str,
        angle: f32,
        style: &dyn StyleOpt, 
        text_style: &TextStyle,
       // prop, - font properties
        // affine: &Affine2d,
        //angle: f32, // rotation
        clip: &Bounds<Canvas>,
    ) -> Result<(), RenderErr> {

        let color = match style.get_facecolor() {
            Some(color) => *color,
            None => Color(0x000000ff),
        };

        let size = match &text_style.get_size() {
            Some(size) => *size,
            None => 12.,
        };

        let size = self.to_px(size);

        let halign = match text_style.get_width_align() {
            Some(align) => align.clone(),
            None => WidthAlign::Center,
        };

        let valign = match text_style.get_height_align() {
            Some(align) => align.clone(),
            None => HeightAlign::Bottom,
        };

        self.text_render.draw(
            text,
            "sans-serif", 
            size,
            xy, 
            Point(self.canvas.width(), self.canvas.height()),
            color,
            angle,
            halign,
            valign,
        );
 
        /*
        self.glyph.queue(Section {
            screen_position: (xy.0, self.pos_canvas.height() - xy.1),
            bounds: (self.pos_canvas.width(), self.pos_canvas.height()),
            text: vec![Text::new(s)
                .with_color([color.red(), color.green(), color.blue(), color.alpha()])
                .with_scale(64.)
                ],
            ..Section::default()
        });
        */

        Ok(())
    }

    fn draw_triangles(
        &mut self,
        vertices: Tensor<f32>,  // Nx2 x,y in canvas coordinates
        rgba: Tensor<u32>,    // N in rgba
        triangles: Tensor<u32>, // Mx3 vertex indices
    ) -> Result<(), RenderErr> {
        assert!(vertices.rank() == 2, 
            "vertices must be 2d (rank2) shape={:?}",
            vertices.shape().as_slice());
        assert!(vertices.cols() == 2, 
            "vertices must be rows of 2 columns (x, y) shape={:?}",
            vertices.shape().as_slice());
        assert!(rgba.rank() == 1,
            "colors must be a 1D vector shape={:?}",
            rgba.shape().as_slice());
        assert!(vertices.rows() == rgba.cols(), 
            "number of vertices and colors must match. vertices={:?} colors={:?}",
            vertices.shape().as_slice(), rgba.shape().as_slice());
        assert!(triangles.cols() == 3, 
            "triangle indices must have 3 vertices (3 columns) shape={:?}",
            triangles.shape().as_slice());

        self.triangle_render.start_triangles();

        for (xy, color) in vertices.iter_slice().zip(rgba.iter()) {
            self.triangle_render.draw_vertex(xy[0], xy[1], *color);
        }

        for tri in triangles.iter_slice() {
            self.triangle_render.draw_triangle(tri[0], tri[1], tri[2]);
        }

        self.triangle_render.draw_style(&self.to_gpu);

        Ok(())
    }

    fn request_redraw(&mut self, bounds: &Bounds<Canvas>) {
        self.is_request_redraw = true;
    }
}

// transform and normalize path
fn transform_path(path: &Path<Canvas>) -> Path<Canvas> {
    let mut codes = Vec::<PathCode>::new();

    let mut p0 = Point(0.0f32, 0.0f32);

    // TODO: clip
    for code in path.codes() {
        p0 = match code {
            PathCode::MoveTo(p0) => {
                codes.push(PathCode::MoveTo(*p0));

                *p0
            }
            PathCode::LineTo(p1) => {
                codes.push(PathCode::LineTo(*p1));

                *p1
            }
            PathCode::Bezier2(p1, p2) => {
                codes.push(PathCode::Bezier2(*p1, *p2));

                *p2
            }
            PathCode::Bezier3(p1, p2, p3) => {
                let p1 = *p1;
                let p2 = *p2;
                let p3 = *p3;

                // Truong, et. al. 2020
                // Quadratic Approximation of Cubic Curves
                // 
                // Note: if more accuracy is needed, the cubic can also be 
                // split into two cubics before converting to quadratics

                // q0_1 = b0 + 0.75 (b1 - b0)
                let q0_1 = Point(
                    p0.x() + 0.75 * (p1.x() - p0.x()),
                    p0.y() + 0.75 * (p1.y() - p0.y()),
                );

                // q1_1 = b3 + 0.75 (b2 - b3)
                let q1_1 = Point(
                    p3.x() + 0.75 * (p2.x() - p3.x()),
                    p3.y() + 0.75 * (p2.y() - p3.y()),
                );

                // q0_2 = q1_0 = 0.5 * (q0_1 + q1_1)
                let q0_2 = Point(
                    0.5 * (q0_1.x() + q1_1.x()),
                    0.5 * (q0_1.y() + q1_1.y()),
                );

                codes.push(PathCode::Bezier2(q0_1, q0_2));
                codes.push(PathCode::Bezier2(q1_1, p3));

                p3
            }
            PathCode::ClosePoly(p1) => {
                codes.push(PathCode::ClosePoly(*p1));

                *p1
            }
        }
    }

    Path::<Canvas>::new(codes)
}

struct Gpu {}
impl Coord for Gpu {}

// For BezierQuadratic to BezierCubic, see Truong, et. al. 2020 Quadratic 
// Approximation of Cubic Curves

fn create_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    // pipeline_layout: &wgpu::PipelineLayout,
    vertex_entry: &str,
    fragment_entry: &str,
    texture_format: wgpu::TextureFormat,
    vertex_layout: wgpu::VertexBufferLayout,
) -> wgpu::RenderPipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: vertex_entry,
            buffers: &[
                vertex_layout
            ],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: fragment_entry,
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: texture_format,

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

impl FigureRenderer {
    pub(crate) fn draw(
        &mut self,
        figure: &mut Box<dyn FigureApi>,
        bounds: (u32, u32),
        scale_factor: f32,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let (width, height) = bounds;

        self.clear();
        // self.staging_belt.recall();

        self.set_canvas_bounds(width, height);
        self.set_scale_factor(scale_factor);
        let draw_bounds = self.canvas.bounds().clone();
        figure.draw(self, &draw_bounds);

        {
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
            // clip 
            // rpass.set_viewport(0., 0., 1., 1., 0.0, 1.0);
            
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

            /*
            let bezier_len = self.bezier_vertex.offset();
            if bezier_len > 0 {
                write_buffer(queue, &mut self.bezier_vertex, bezier_len);
                rpass.set_pipeline(&self.bezier_pipeline);

                let size = (bezier_len * self.bezier_vertex.stride()) as u64;
                rpass.set_vertex_buffer(0, self.bezier_vertex.buffer().slice(..size));
        
                rpass.draw(0..bezier_len as u32, 0..1);
            }

            let bezier_rev_len = self.bezier_rev_vertex.offset();
            if bezier_rev_len > 0 {
                let size = (bezier_rev_len * self.bezier_rev_vertex.stride()) as u64;
                write_buffer(queue, &mut self.bezier_rev_vertex, bezier_rev_len);
                rpass.set_pipeline(&self.bezier_rev_pipeline);

                rpass.set_vertex_buffer(0, self.bezier_rev_vertex.buffer().slice(..size));
        
                rpass.draw(0..bezier_rev_len as u32, 0..1);
            }
            */
        }

        // let (width, height) = (self.canvas.width(), self.canvas.height());

        self.shape2d_render.flush(queue, view, encoder);
        self.bezier_render.flush(queue, view, encoder);
        self.triangle_render.flush(queue, view, encoder);
        self.text_render.flush(queue, view, encoder);

        /*
        self.glyph.queue(Section {
            screen_position: (100., 100.),
            bounds: (width, height),
            text: vec![Text::new("Hello")
                .with_color([0.0, 0., 0., 1.])
                .with_scale(40.)],
            ..Section::default()
        });

        self.glyph.draw_queued(
            device,
            &mut self.staging_belt,
            encoder,
            &view,
            width as u32,
            height as u32,
        ).unwrap();
        */
        // self.staging_belt.finish();
        //self.staging_belt.recall();
    }
}

pub fn write_buffer(queue: &wgpu::Queue, vertices: &mut VertexBuffer, _size: usize) {
    //queue.write_buffer(vertices.buffer(), 0, bytemuck::cast_slice(vertices.as_slice()));
    queue.write_buffer(vertices.buffer(), 0, bytemuck::cast_slice(vertices.as_slice()));
}
