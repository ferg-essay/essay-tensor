use winit::{
    event::{Event, WindowEvent},    
    event_loop::{EventLoop, ControlFlow}, 
    window::Window,
};

use crate::{figure::FigureInner};

use super::{render::{FigureRenderer}, vertex::VertexBuffer};

async fn init_wgpu_args(window: &Window) -> EventLoopArgs {
    // event_loop: EventLoop<()>, window: Window) -> EventLoopArgs {
    let size = window.inner_size();

    let instance = wgpu::Instance::default();

    let surface = unsafe { instance.create_surface(&window) }.unwrap();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        })
        .await
        .expect("Failed to find adapter");

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
            },
            None,
        )
        .await
        .expect("Failed to create device");

    //let shader = device.create_shader_module(wgpu::include_wgsl!("shader2.wgsl"));

    //let vertex_buffer = VertexBuffer::new(1024, &device);

    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0];
    /*
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });

    let render_pipeline = create_pipeline(
        &device,
        &shader,
        &pipeline_layout,
        "vs_main",
        "fs_main",
        swapchain_format,
        vertex_buffer.desc(),
    );

    let bezier_shader = device.create_shader_module(wgpu::include_wgsl!("bezier.wgsl"));

    let bezier_vertex = VertexBuffer::new(1024, &device);

    let bezier_pipeline = create_pipeline(
        &device,
        &bezier_shader,
        &pipeline_layout,
        "vs_bezier",
        "fs_bezier",
        swapchain_format,
        bezier_vertex.desc(),
    );

    let bezier_rev_vertex = VertexBuffer::new(1024, &device);

    let bezier_rev_pipeline = create_pipeline(
        &device,
        &bezier_shader,
        &pipeline_layout,
        "vs_bezier",
        "fs_bezier_rev",
        swapchain_format,
        bezier_rev_vertex.desc(),
    );
    */

    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: swapchain_format,
        width: size.width,
        height: size.height,
        //present_mode: wgpu::PresentMode::Fifo,
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: swapchain_capabilities.alpha_modes[0],
        view_formats: vec![],
    };

    surface.configure(&device, &config);

    EventLoopArgs {
        instance,
        adapter,
        device,
        config,
        surface,
        //shader,
        queue,
        /*
        pipeline_layout,
        render_pipeline,
        vertex_buffer,
        bezier_pipeline,
        bezier_vertex,
        bezier_rev_pipeline,
        bezier_rev_vertex,
        */
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    pipeline_layout: &wgpu::PipelineLayout,
    vertex_entry: &str,
    fragment_entry: &str,
    texture_format: wgpu::TextureFormat,
    vertex_layout: wgpu::VertexBufferLayout,
) -> wgpu::RenderPipeline {
    //let swapchain_format = wgpu::TextureFormat::Rgba16Float;

    // let swapchain_capabilities = surface.get_capabilities(&adapter);
    // let swapchain_format = swapchain_capabilities.formats[0];

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

struct EventLoopArgs {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    surface: wgpu::Surface,
    //shader: wgpu::ShaderModule,
    /*
    pipeline_layout: wgpu::PipelineLayout,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: VertexBuffer,
    bezier_pipeline: wgpu::RenderPipeline,
    bezier_vertex: VertexBuffer,
    bezier_rev_pipeline: wgpu::RenderPipeline,
    bezier_rev_vertex: VertexBuffer,
    */
}

fn run_event_loop(
    event_loop: EventLoop<()>, 
    window: Window, 
    args: EventLoopArgs,
    figure: FigureInner,
) {
    let EventLoopArgs {
        instance,
        adapter,
        mut config,
        device,
        surface,
        queue,
        //shader,
        /*
        pipeline_layout,
        render_pipeline,
        mut vertex_buffer,
        bezier_pipeline,
        mut bezier_vertex,
        bezier_rev_pipeline,
        mut bezier_rev_vertex,
        */
    } = args;

    let mut figure = figure;


    //let mut staging_belt = wgpu::util::StagingBelt::new(1024);

    let mut figure_renderer = FigureRenderer::new(
        &device,
        config.format,
        /*
        &render_pipeline,
        &mut vertex_buffer,
        &bezier_pipeline,
        &mut bezier_vertex,
        &bezier_rev_pipeline,
        &mut bezier_rev_vertex,
        */
    );

    event_loop.run(move |event, _, control_flow| {
        let _ = (&instance, &adapter, &figure);

        let mut main_renderer = MainRenderer::new(
            &surface,
            // &config,
            &device,
            &queue,
            config.format,
        );
    
        *control_flow = ControlFlow::Wait;
        match event {
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                config.width = size.width;
                config.height = size.height;
                surface.configure(&device, &config);
                figure_renderer.set_canvas_bounds(config.width, config.height);
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                //figure_renderer.clear();
                //figure.draw(&mut figure_renderer);
                main_renderer.render(|device, queue, view, encoder| {
                    figure_renderer.draw(&mut figure, device, queue, view, encoder);
                });
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            _ => {}
        }
    });
}

struct MainRenderer<'a> {
    surface: &'a wgpu::Surface,
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    render_format: wgpu::TextureFormat,
}

impl<'a> MainRenderer<'a> {
    pub(crate) fn new(
        surface: &'a wgpu::Surface,
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        render_format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            surface,
            device,
            queue,
            render_format,
        }
    }

    pub(crate) fn render(
        &mut self, 
        view_renderer: impl FnOnce(
            &wgpu::Device,
            &wgpu::Queue, 
            &wgpu::TextureView, 
            &mut wgpu::CommandEncoder
        )
    ) {
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
            // clip 
            // rpass.set_viewport(0., 0., 1., 1., 0.0, 1.0);
            
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
        }

        view_renderer(&self.device, &self.queue, &view, &mut encoder);

        //self.staging_belt.finish();
        self.queue.submit(Some(encoder.finish()));
        frame.present();
        //self.staging_belt.recall();
    }
}

pub trait ViewRenderer {
    fn render(
        &mut self,
        device: &wgpu::Device, 
        queue: &wgpu::Queue, 
        view: &wgpu::TextureView, 
        encoder: &wgpu::CommandEncoder
    );
}

pub(crate) fn main_loop(figure: FigureInner) {
    let event_loop = EventLoop::new();
    let window = winit::window::Window::new(&event_loop).unwrap();

    env_logger::init();
    let wgpu_args = pollster::block_on(init_wgpu_args(&window));

    run_event_loop(event_loop, window, wgpu_args, figure);
}