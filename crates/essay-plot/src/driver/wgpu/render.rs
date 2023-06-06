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
    ) -> Self {
        Self {
            surface,
            device,
            queue,
            render_pipeline,
            vertex_buffer,

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

    fn draw_closed_path(
        &mut self, 
        gc: &GraphicsContext, 
        path: &Path<Data>, 
        to_device: &Affine2d,
        _clip: &Bounds<Device>,
    ) {
        let to_unit = self.to_unit.matmul(to_device);

        let mut points = Vec::<Point>::new();

        let xy = path.points();

        for point in xy.iter_slice() {
            let point = to_unit.transform_point(Point(point[0], point[1]));

            points.push(point);
        }

        let triangles = tesselate::tesselate(points);

        println!("Triangles {:?}", triangles);

        for triangle in &triangles {
            self.vertex_buffer.push(triangle[0].x(), triangle[0].y(), 0xc00000ff);
            self.vertex_buffer.push(triangle[1].x(), triangle[1].y(), 0xc0c000ff);
            self.vertex_buffer.push(triangle[2].x(), triangle[2].y(), 0xc00000ff);
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
        let color = gc.get_rgba();

        let to_unit = self.to_unit.matmul(to_device);

        for (point, code) in path.points().iter_slice().zip(path.codes().iter()) {
            let p1 = to_unit.transform_point(Point(point[0], point[1]));
            //println!("Point {:?} {:?} {},{}", point, code, x, y);

            match code {
                PathCode::MoveTo => {
                    p0 = p1;
                }
                PathCode::LineTo => {
                    self.draw_line(p0.x(), p0.y(), p1.x(), p1.y(), thickness, color);

                    // TODO: clip
                    p0 = p1;
                }
                PathCode::BezierQuadratic => todo!(),
                PathCode::BezierCubic => todo!(),
                PathCode::ClosePoly => todo!(),
            }
        }

        Ok(())
    }
}

pub fn write_buffer(queue: &wgpu::Queue, vertices: &mut VertexBuffer, len: usize) {
    queue.write_buffer(vertices.buffer(), 0, bytemuck::cast_slice(vertices.as_slice()));
}
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

                println!("Render {}", vertex_len);
                let size = (vertex_len * self.vertex_buffer.stride()) as u64;
                rpass.set_vertex_buffer(0, self.vertex_buffer.buffer().slice(..size));
        
                rpass.draw(0..vertex_len as u32, 0..1);
            }
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
    }
}
