use wgpu::util::DeviceExt;
use wgpu_glyph::{ab_glyph::{self, FontArc}, GlyphBrushBuilder, GlyphBrush, Section, Text};

use crate::{
    driver::{Renderer, Device, GraphicsContext, renderer::RenderErr, wgpu::tesselate}, 
    figure::{Bounds, Point, Affine2d, Data, CoordMarker, Figure, FigureInner}, 
    artist::{Path, PathCode}
};

use super::vertex::VertexBuffer;

pub struct PlotRenderer<'a> {
    surface: &'a wgpu::Surface,
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    //staging_belt: &'a mut wgpu::util::StagingBelt,

    render_pipeline: &'a wgpu::RenderPipeline,
    vertex_buffer: &'a mut VertexBuffer,
    bezier_pipeline: &'a wgpu::RenderPipeline,
    bezier_vertex: &'a mut VertexBuffer,
    bezier_rev_pipeline: &'a wgpu::RenderPipeline,
    bezier_rev_vertex: &'a mut VertexBuffer,

    glyph: GlyphBrush<()>,

    pos_device: Bounds<Device>,
    to_unit: Affine2d,
}

impl<'a> PlotRenderer<'a> {
    pub(crate) fn new(
        surface: &'a wgpu::Surface,
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        //staging_belt: &'a mut wgpu::util::StagingBelt,

        render_format: wgpu::TextureFormat,

        render_pipeline: &'a wgpu::RenderPipeline,
        vertex_buffer: &'a mut VertexBuffer,
        bezier_pipeline: &'a wgpu::RenderPipeline,
        bezier_vertex: &'a mut VertexBuffer,
        bezier_rev_pipeline: &'a wgpu::RenderPipeline,
        bezier_rev_vertex: &'a mut VertexBuffer,
    ) -> Self {
        let font_palatino = ab_glyph::FontArc::try_from_slice(include_bytes!(
            "/System/Library/Fonts/Palatino.ttc"
        )).unwrap();

        let glyph_brush = GlyphBrushBuilder::using_font(font_palatino)
            .build(&device, render_format);
        
        Self {
            surface,
            device,
            queue,
            //staging_belt,
            render_pipeline,
            vertex_buffer,
            bezier_pipeline,
            bezier_vertex,
            bezier_rev_pipeline,
            bezier_rev_vertex,

            glyph: glyph_brush,

            pos_device: Bounds::<Device>::unit(),
            to_unit: Affine2d::eye(),
        }
    }

    pub(crate) fn set_canvas_bounds(&mut self, width: u32, height: u32) {
        self.pos_device = Bounds::<Device>::new(
            Point(0., 0.), 
            Point(width as f32, height as f32)
        );

        let pos_unit = Bounds::<Device>::new(
            Point(-1., -1.),
            Point(1., 1.)
        );

        self.to_unit = self.pos_device.affine_to(&pos_unit);
    }
    /*
    fn render(&mut self) {
        render(
            self.surface, 
            self.device,
            self.queue,
            self.render_pipeline,
            &mut self.vertex_buffer,
        )
    }
    */

    pub(crate) fn clear(&mut self) {
        self.vertex_buffer.clear();
        self.bezier_vertex.clear();
    }

    fn draw_line(&mut self, 
        x0: f32, y0: f32, 
        x1: f32, y1: f32,
        thickness: f32,
        color: u32
    ) {
        let dx = x1 - x0;
        let dy = y1 - y0;

        let len = dx.hypot(dy).max(f32::EPSILON);

        // normal to the line
        let nx = dy * (thickness / len);
        let ny = - dx * (thickness / len);

        self.vertex_buffer.push(x0 - nx, y0 - ny, color);
        self.vertex_buffer.push(x0 + nx, y0 + ny, color);
        self.vertex_buffer.push(x1 + nx, y1 + ny, color);

        self.vertex_buffer.push(x1 + nx, y1 + ny, color);
        self.vertex_buffer.push(x1 - nx, y1 - ny, color);
        self.vertex_buffer.push(x0 - nx, y0 - ny, color);
    }

    fn draw_bezier(&mut self, 
        p0: Point,
        p1: Point,
        p2: Point,
        color: u32
    ) {
        //self.vertex_buffer.push(p0.x(), p0.y(), 0x000000ff);
        //self.vertex_buffer.push(p1.x(), p1.y(), 0x000000ff);
        //self.vertex_buffer.push(p2.x(), p2.y(), 0x0000000ff);

        //self.bezier_vertex.push_tex(p0.x(), p0.y(), -1.0,0.0, color);
        //self.bezier_vertex.push_tex(p1.x(), p1.y(), 0.0, 2.0, color);
        //self.bezier_vertex.push_tex(p2.x(), p2.y(), 1.0, 0.0, color);

        self.bezier_vertex.push_tex(p0.x(), p0.y(), -1.0,1.0, color);
        self.bezier_vertex.push_tex(p1.x(), p1.y(), 0.0, -1.0, color);
        self.bezier_vertex.push_tex(p2.x(), p2.y(), 1.0, 1.0, color);

        //self.bezier_rev_vertex.push_tex(p0.x(), p0.y(), -1.0,1.0, color);
        //self.bezier_rev_vertex.push_tex(p1.x(), p1.y(), 0.0, -1.0, color);
        //self.bezier_rev_vertex.push_tex(p2.x(), p2.y(), 1.0, 1.0, color);
    }

    fn draw_closed_path(
        &mut self, 
        gc: &GraphicsContext, 
        path: &Path<Unit>, 
        to_device: &Affine2d,
        _clip: &Bounds<Device>,
    ) {
        // let to_unit = self.to_unit.matmul(to_device);
        let rgba = gc.get_color().get_rgba();

        let mut points = Vec::<Point>::new();
        let mut prev = Point(0., 0.);
        for code in path.codes() {
            let last = code.tail();

            points.push(last);

            if let PathCode::Bezier2(p1, p2) = code {
                self.draw_bezier(prev, *p1, *p2, rgba);
            }

            prev = last;
        }

        let triangles = tesselate::tesselate(points);

        for triangle in &triangles {
            self.vertex_buffer.push(triangle[0].x(), triangle[0].y(), rgba);
            self.vertex_buffer.push(triangle[1].x(), triangle[1].y(), rgba);
            self.vertex_buffer.push(triangle[2].x(), triangle[2].y(), rgba);
        }
    }
}

impl Renderer for PlotRenderer<'_> {
    ///
    /// Returns the boundary of the canvas, usually in pixels or points.
    ///
    fn get_canvas_bounds(&self) -> Bounds<Device> {
        self.pos_device.clone()
    }

    fn new_gc(&mut self) -> GraphicsContext {
        GraphicsContext::default()
    }

    fn draw_path(
        &mut self, 
        gc: &GraphicsContext, 
        path: &Path<Data>, 
        to_device: &Affine2d,
        _clip: &Bounds<Device>,
    ) -> Result<(), RenderErr> {
        let to_unit = self.to_unit.matmul(to_device);

        let path = transform_path(path, &to_unit);

        if path.is_closed_path() {
            self.draw_closed_path(gc, &path, to_device, _clip);
            return Ok(());
        }

        // TODO: thickness in points
        let thickness  = 0.5 * gc.get_linewidth() / self.pos_device.height();
        let color = gc.get_color().get_rgba();

        let mut p0 = Point(0.0f32, 0.0f32);
        for code in path.codes() {
            p0 = match code {
                PathCode::MoveTo(p0) => {
                    *p0
                }
                PathCode::LineTo(p1) => {
                    self.draw_line(p0.x(), p0.y(), p1.x(), p1.y(), thickness, color);
                    // TODO: clip
                    *p1
                }
                PathCode::Bezier2(p1, p2) => {
                    self.draw_bezier(p0, *p1, *p2, color);

                    *p2
                }
                PathCode::Bezier3(_, _, _) => {
                    panic!("Bezier3 should already be split into Bezier2");
                }
                PathCode::ClosePoly(p1) => {
                    self.draw_line(p0.x(), p0.y(), p1.x(), p1.y(), thickness, color);

                    *p1
                }
            }
        }

        Ok(())
    }
}

// transform and normalize path
fn transform_path(path: &Path<Data>, to_unit: &Affine2d) -> Path<Unit> {
    let mut codes = Vec::<PathCode>::new();

    let mut p0 = Point(0.0f32, 0.0f32);

    // TODO: clip
    for code in path.codes() {
        p0 = match code {
            PathCode::MoveTo(p0) => {
                let p0 = to_unit.transform_point(*p0);

                codes.push(PathCode::MoveTo(p0));

                p0
            }
            PathCode::LineTo(p1) => {
                let p1 = to_unit.transform_point(*p1);

                codes.push(PathCode::MoveTo(p1));

                p1
            }
            PathCode::Bezier2(p1, p2) => {
                let p1 = to_unit.transform_point(*p1);
                let p2 = to_unit.transform_point(*p2);

                codes.push(PathCode::Bezier2(p1, p2));

                p2
            }
            PathCode::Bezier3(p1, p2, p3) => {
                let p1 = to_unit.transform_point(*p1);
                let p2 = to_unit.transform_point(*p2);
                let p3 = to_unit.transform_point(*p3);

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
                let p1 = to_unit.transform_point(*p1);

                codes.push(PathCode::ClosePoly(p1));

                p1
            }
        }
    }

    Path::<Unit>::new(codes)
}

struct Unit {}
impl CoordMarker for Unit {}

// For BezierQuadratic to BezierCubic, see Truong, et. al. 2020 Quadratic 
// Approximation of Cubic Curves
//surface: &wgpu::Surface,
//device: &wgpu::Device,
//queue: &wgpu::Queue,
//render_pipeline: &wgpu::RenderPipeline,
//vertex_buffer: &mut VertexBuffer,

impl PlotRenderer<'_> {
    pub(crate) fn draw(
        &mut self,
        figure: &mut FigureInner,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        self.clear();
        figure.draw(self);

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 1.0,
                            g: 1.0,
                            b: 1.0,
                            a: 1.0,
                        }),
                        store: true,
                    }
                })],
                depth_stencil_attachment: None,
            });
            // clip 
            // rpass.set_viewport(0., 0., 1., 1., 0.0, 1.0);
            

            let vertex_len = self.vertex_buffer.offset();
            if vertex_len > 0 {
                write_buffer(self.queue, self.vertex_buffer, vertex_len);

                rpass.set_pipeline(&self.render_pipeline);

                let size = (vertex_len * self.vertex_buffer.stride()) as u64;
                rpass.set_vertex_buffer(0, self.vertex_buffer.buffer().slice(..size));
        
                rpass.draw(0..vertex_len as u32, 0..1);
            }

            let bezier_len = self.bezier_vertex.offset();
            if bezier_len > 0 {
                write_buffer(self.queue, self.bezier_vertex, bezier_len);
                rpass.set_pipeline(&self.bezier_pipeline);

                let size = (bezier_len * self.bezier_vertex.stride()) as u64;
                rpass.set_vertex_buffer(0, self.bezier_vertex.buffer().slice(..size));
        
                rpass.draw(0..bezier_len as u32, 0..1);
            }

            let bezier_rev_len = self.bezier_rev_vertex.offset();
            if bezier_rev_len > 0 {
                let size = (bezier_rev_len * self.bezier_rev_vertex.stride()) as u64;
                write_buffer(self.queue, self.bezier_rev_vertex, bezier_rev_len);
                rpass.set_pipeline(&self.bezier_rev_pipeline);

                rpass.set_vertex_buffer(0, self.bezier_rev_vertex.buffer().slice(..size));
        
                rpass.draw(0..bezier_rev_len as u32, 0..1);
            }
        }
        /*
        self.glyph.queue(Section {
            screen_position: (100., 100.),
            bounds: (1600., 800.),
            text: vec![Text::new("Hello")
                .with_color([0.0, 0., 0., 1.])
                .with_scale(40.)],
            ..Section::default()
        });

        self.glyph.draw_queued(
            &self.device,
            self.staging_belt,
            &mut encoder,
            &view,
            1600,
            800,
        ).unwrap();
        */
    }
}

pub fn write_buffer(queue: &wgpu::Queue, vertices: &mut VertexBuffer, size: usize) {
    //queue.write_buffer(vertices.buffer(), 0, bytemuck::cast_slice(vertices.as_slice()));
    queue.write_buffer(vertices.buffer(), 0, bytemuck::cast_slice(vertices.as_slice()));
}
