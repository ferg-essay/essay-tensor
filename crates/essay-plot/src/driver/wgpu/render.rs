use wgpu::util::DeviceExt;

use crate::{
    driver::{Renderer, Device, GraphicsContext, renderer::RenderErr, wgpu::tesselate}, 
    figure::{Bounds, Point, Affine2d, Data}, artist::{Path, PathCode}
};

use super::vertex::VertexBuffer;

pub struct WgpuRenderer<'a> {
    surface: &'a wgpu::Surface,
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,

    render_pipeline: &'a wgpu::RenderPipeline,
    vertex_buffer: &'a mut VertexBuffer,
    bezier_pipeline: &'a wgpu::RenderPipeline,
    bezier_vertex: &'a mut VertexBuffer,

    pos_device: Bounds<Device>,
    to_unit: Affine2d,
}

impl<'a> WgpuRenderer<'a> {
    pub(crate) fn new(
        surface: &'a wgpu::Surface,
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        render_pipeline: &'a wgpu::RenderPipeline,
        vertex_buffer: &'a mut VertexBuffer,
        bezier_pipeline: &'a wgpu::RenderPipeline,
        bezier_vertex: &'a mut VertexBuffer,
    ) -> Self {
        Self {
            surface,
            device,
            queue,
            render_pipeline,
            vertex_buffer,
            bezier_pipeline,
            bezier_vertex,

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
        println!("Set-Canvas {:?} {:?}", self.pos_device, pos_unit);
        println!("  To unit {:?}", self.to_unit);
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
        println!("VB");
    }

    fn draw_bezier(&mut self, 
        p0: Point,
        p1: Point,
        p2: Point,
        color: u32
    ) {
        self.bezier_vertex.push_tex(p0.x(), p0.y(), -1.0,0.0, color);
        self.bezier_vertex.push_tex(p1.x(), p1.y(), 0.0, 2.0, color);
        self.bezier_vertex.push_tex(p2.x(), p2.y(), 1.0, 0.0, color);
        //self.vertex_buffer.push(p0.x(), p0.y(), color);
        //self.vertex_buffer.push(p1.x(), p1.y(), color);
        //self.vertex_buffer.push(p2.x(), p2.y(), color);
        println!("DB");
    }

    fn draw_closed_path(
        &mut self, 
        gc: &GraphicsContext, 
        path: &Path<Data>, 
        to_device: &Affine2d,
        _clip: &Bounds<Device>,
    ) {
        let to_unit = self.to_unit.matmul(to_device);
        println!("ClosedPath");

        let mut points = Vec::<Point>::new();

        for code in path.codes() {
            points.push(to_unit.transform_point(code.tail()));
        }
        /*
        let xy = path.points();

        for point in xy.iter_slice() {
            let point = to_unit.transform_point(Point(point[0], point[1]));

            points.push(point);
        }
        */

        let triangles = tesselate::tesselate(points);
        let rgba = gc.get_color().get_rgba();

        for triangle in &triangles {
            self.vertex_buffer.push(triangle[0].x(), triangle[0].y(), rgba);
            self.vertex_buffer.push(triangle[1].x(), triangle[1].y(), rgba);
            self.vertex_buffer.push(triangle[2].x(), triangle[2].y(), rgba);
        }
    }
}

impl Renderer for WgpuRenderer<'_> {
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
        if path.is_closed_path() {
            self.draw_closed_path(gc, path, to_device, _clip);
            return Ok(());
        }
        let mut p0 = Point(0.0f32, 0.0f32);

        // TODO: thickness in points
        let thickness  = 0.5 * gc.get_linewidth() / self.pos_device.height();
        let color = gc.get_color().get_rgba();

        let to_unit = self.to_unit.matmul(to_device);

        let path = path.transform::<Device>(&to_unit);

        for code in path.codes() {
            p0 = match code {
                PathCode::MoveTo(p0) => {
                    *p0
                }
                PathCode::LineTo(p1) => {
                    self.draw_line(p0.x(), p0.y(), p1.x(), p1.y(), thickness, color);
                    println!("Line {:?} {:?}", p0, p1);
                    // TODO: clip
                    *p1
                }
                PathCode::Bezier2(p1, p2) => {
                    self.draw_bezier(p0, *p1, *p2, color);
                    println!("Bezier2 {:?} {:?} {:?}", p0, p1, p2);

                    *p2
                }
                PathCode::Bezier3(p1, p2, p3) => {
                    println!("Bezier3 {:?} {:?} {:?} {:?}", p0, p1, p2, p3);

                    *p3
                }
                PathCode::ClosePoly(p1) => todo!(),
            }
        }

        Ok(())
    }
}

// For BezierQuadratic to BezierCubic, see Truong, et. al. 2020 Quadratic 
// Approximation of Cubic Curves
//surface: &wgpu::Surface,
//device: &wgpu::Device,
//queue: &wgpu::Queue,
//render_pipeline: &wgpu::RenderPipeline,
//vertex_buffer: &mut VertexBuffer,

impl WgpuRenderer<'_> {
    pub(crate) fn render(&mut self) {
        let frame = self.surface
            .get_current_texture()
            .expect("Failed to get next swap chain texture");

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // self.vertex_buffer.clear();
        //path_render(self.vertex_buffer);

        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
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

                println!("Vertex_len {}", vertex_len);
                let size = (vertex_len * self.vertex_buffer.stride()) as u64;
                rpass.set_vertex_buffer(0, self.vertex_buffer.buffer().slice(..size));
        
                rpass.draw(0..vertex_len as u32, 0..1);
            }

            let bezier_len = self.bezier_vertex.offset();
            if bezier_len > 0 {
                write_buffer(self.queue, self.bezier_vertex, bezier_len);
                println!("Bezier_len {}", bezier_len);
                rpass.set_pipeline(&self.bezier_pipeline);

                let size = (bezier_len * self.bezier_vertex.stride()) as u64;
                rpass.set_vertex_buffer(0, self.bezier_vertex.buffer().slice(..size));
        
                rpass.draw(0..bezier_len as u32, 0..1);
            }
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
    }
}

pub fn write_buffer(queue: &wgpu::Queue, vertices: &mut VertexBuffer, len: usize) {
    queue.write_buffer(vertices.buffer(), 0, bytemuck::cast_slice(vertices.as_slice()));
}
