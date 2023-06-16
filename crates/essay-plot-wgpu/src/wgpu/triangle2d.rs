use essay_plot_base::{Affine2d};
use wgpu::util::DeviceExt;

pub struct GridMesh2dRender {
    vertex_stride: usize,
    vertex_vec: Vec<GridMesh2dVertex>,
    vertex_buffer: wgpu::Buffer,
    vertex_offset: usize,

    index_stride: usize,
    index_vec: Vec<u32>,
    index_buffer: wgpu::Buffer,
    index_offset: usize,

    style_stride: usize,
    style_vec: Vec<GridMesh2dStyle>,
    style_buffer: wgpu::Buffer,
    style_offset: usize,

    mesh_items: Vec<GridMesh2dItem>,

    pipeline: wgpu::RenderPipeline,
}

impl GridMesh2dRender {
    pub(crate) fn new(
        device: &wgpu::Device, 
        format: wgpu::TextureFormat,
    ) -> Self {
        let len = 2048;

        let mut vertex_vec = Vec::<GridMesh2dVertex>::new();
        vertex_vec.resize(len, GridMesh2dVertex { 
            position: [0.0, 0.0], 
            color: 0 
        });

        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(vertex_vec.as_slice()),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );

        let mut index_vec = Vec::<u32>::new();
        index_vec.resize(len, 0);

        let index_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(index_vec.as_slice()),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            }
        );

        let mut style_vec = Vec::<GridMesh2dStyle>::new();
        style_vec.resize(len, GridMesh2dStyle { 
            affine_0: [0.0, 0.0, 0.0, 0.0], 
            affine_1: [0.0, 0.0, 0.0, 0.0], 
        });

        let style_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(style_vec.as_slice()),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );

        let pipeline = create_gridmesh2d_pipeline(
            device, 
            format,
        );
    
        Self {
            vertex_stride: std::mem::size_of::<GridMesh2dVertex>(),
            vertex_vec,
            vertex_buffer,
            vertex_offset: 0,

            index_stride: std::mem::size_of::<u32>(),
            index_vec,
            index_buffer,
            index_offset: 0,

            style_stride: std::mem::size_of::<GridMesh2dStyle>(),
            style_vec,
            style_buffer,
            style_offset: 0,
            // style_bind_group,

            mesh_items: Vec::new(),
            pipeline,
        }
    }

    pub fn clear(&mut self) {
        self.vertex_offset = 0;
        self.index_offset = 0;
        self.style_offset = 0;
        self.mesh_items.drain(..);
    }

    pub fn start_triangles(&mut self) {
        self.mesh_items.push(GridMesh2dItem {
            v_start: self.vertex_offset,
            v_end: usize::MAX,
            i_start: self.index_offset,
            i_end: usize::MAX,
            s_start: self.style_offset,
            s_end: usize::MAX,
        });
    }

    pub fn draw_vertex(&mut self, x: f32, y: f32, rgba: u32) {
        let vertex = GridMesh2dVertex { 
            position: [x, y],
            color: rgba,
        };
        
        // TODO: autoextend on overflow
        self.vertex_vec[self.vertex_offset] = vertex;
        self.vertex_offset += 1;
    }

    pub fn draw_triangle(&mut self, v0: u32, v1: u32, v2: u32) {
        let item = &self.mesh_items[self.mesh_items.len() - 1];

        let v_start = item.v_start;
        let offset = self.index_offset;

        // TODO: autoextend on overflow
        assert!((v_start + v0 as usize) < self.vertex_offset);
        self.index_vec[offset] = v0;
        assert!((v_start + v1 as usize) < self.vertex_offset);
        self.index_vec[offset + 1] = v1;
        assert!((v_start + v2 as usize) < self.vertex_offset);
        self.index_vec[offset + 2] = v2;

        self.index_offset += 3;
    }

    pub fn draw_style(
        &mut self, 
        affine: &Affine2d,
    ) {
        let v_end = self.vertex_offset;
        let i_end = self.index_offset;

        let len = self.mesh_items.len();
        let item = &mut self.mesh_items[len - 1];

        item.v_end = v_end;
        item.i_end = i_end;

        self.style_vec[self.style_offset] = GridMesh2dStyle::new(affine);
        self.style_offset += 1;

        item.s_end = self.style_offset;
    }

    pub fn flush(
        &mut self, 
        queue: &wgpu::Queue, 
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        if self.mesh_items.len() == 0 {
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
            &mut self.index_buffer, 
            0,
            bytemuck::cast_slice(self.index_vec.as_slice())
        );

        queue.write_buffer(
            &mut self.style_buffer, 
            0,
            bytemuck::cast_slice(self.style_vec.as_slice())
        );

        rpass.set_pipeline(&self.pipeline);

        for item in self.mesh_items.drain(..) {
            let stride = self.vertex_stride;
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(
                (stride * item.v_start) as u64..(stride * item.v_end) as u64
            ));
            let stride = self.style_stride;
            rpass.set_vertex_buffer(1, self.style_buffer.slice(
                (stride * item.s_start) as u64..(stride * item.s_end) as u64
            ));
            let stride = self.index_stride;
            rpass.set_index_buffer(self.index_buffer.slice(
                (stride * item.i_start) as u64..(stride * item.i_end) as u64
            ), wgpu::IndexFormat::Uint32
            );

            if item.v_start < item.v_end {
                rpass.draw_indexed(
                    0..(item.i_end - item.i_start) as u32,
                    0,
                    0..(item.s_end - item.s_start) as u32,
                );
            }
        }
    }
}

pub struct GridMesh2dItem {
    v_start: usize,
    v_end: usize,

    i_start: usize,
    i_end: usize,

    s_start: usize,
    s_end: usize,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GridMesh2dVertex {
    position: [f32; 2],
    color: u32,
}

impl GridMesh2dVertex {
    const ATTRS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Uint32 ];

    pub(crate) fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GridMesh2dVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }

}
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GridMesh2dStyle {
    affine_0: [f32; 4],
    affine_1: [f32; 4],
}

impl GridMesh2dStyle {
    const ATTRS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![
            2 => Float32x4, 
            3 => Float32x4,
        ];

    pub(crate) fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GridMesh2dStyle>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRS,
        }
    }

    fn new(affine: &Affine2d) -> Self {
        let mat = affine.mat();

        Self {
            affine_0: [mat[0], mat[1], 0., mat[2]],
            affine_1: [mat[3], mat[4], 0., mat[5]],
        }
    }
}

fn create_gridmesh2d_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("triangle2d.wgsl"));

    let vertex_entry = "vs_triangle";
    let fragment_entry = "fs_triangle";

    let vertex_layout = GridMesh2dVertex::desc();
    let style_layout = GridMesh2dStyle::desc();

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
