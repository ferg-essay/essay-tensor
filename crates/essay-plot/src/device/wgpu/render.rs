use wgpu::util::DeviceExt;

use crate::{device::{Renderer, Device, GraphicsContext, renderer::RenderErr}, axes::{Bounds, Point, Affine2d, Data}, artist::{Path, PathCode}};

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
            }
        }

        Ok(())
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    position: [f32; 2],
    tex_coord: [f32; 2],
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
        vec.resize(len, Vertex { position: [0.0, 0.0], tex_coord: [0.0, 0.0], color: 0x00000000 });

        // path_render2(vec.as_mut_slice());

        let buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
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

    pub(crate) fn as_mut_slice(&mut self) -> &mut [Vertex] {
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

    fn push(&mut self, x: f32, y: f32, color: u32) {
        self.vec[self.offset] = ([x, y], color).into();
        self.offset += 1;
    }
}

pub fn path_render(vertices: &mut VertexBuffer) {
    vertices.push(0.0, 0.0, 0x0000ffff);
    vertices.push(0.0, 1.0, 0x0000ffff);
    vertices.push(1.0, 0.0, 0x0000ffff);
}

pub fn path_render2(vertices: &mut VertexBuffer) {
    vertices.push(-1.0, -1.0, 0x0000ffff);
    vertices.push(-1.0, 0.0, 0x0000ffff);
    vertices.push(0.0, -1.0, 0x0000ffff);
    /*
    vertices[3] = ([0.0, 0.0], [0.5, 0.0], 0xff00ffff).into();
    vertices[4] = ([0.0, -1.0], [1.0, 1.0], 0x00ffffff).into();
    vertices[5] = ([-1.0, 0.0], [0.0, 0.0], 0x00ff00ff).into();
    */

    vertices.push(0.0, 0.0, 0xff0000ff);
    vertices.push(0.0, -1.0, 0xff0000ff);
    vertices.push(-1.0, 0.0, 0xff0000ff);
}

pub fn write_buffer(queue: &wgpu::Queue, vertices: &mut VertexBuffer, len: usize) {
    queue.write_buffer(vertices.buffer(), 0, bytemuck::cast_slice(vertices.as_slice()));
}

impl From<([f32; 2], u32)> for Vertex {
    fn from(value: ([f32; 2], u32)) -> Self {
        Vertex { position: value.0, tex_coord: [0., 0.], color: value.1 }
    }
}
impl From<([f32; 2], [f32; 2], u32)> for Vertex {
    fn from(value: ([f32; 2], [f32; 2], u32)) -> Self {
        Vertex { position: value.0, tex_coord: value.1, color: value.2 }
    }
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
            write_buffer(self.queue, self.vertex_buffer, vertex_len);

            rpass.set_pipeline(&self.render_pipeline);

        //let vertex_len = 3;
            println!("Render {}", vertex_len);
            let size = (vertex_len * self.vertex_buffer.stride()) as u64;
            rpass.set_vertex_buffer(0, self.vertex_buffer.buffer().slice(..size));
        // rpass.draw(0..self.num_vertices, 0..1);
            rpass.draw(0..vertex_len as u32, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
    }
}
