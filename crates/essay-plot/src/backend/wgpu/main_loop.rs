use std::borrow::Cow;

use winit::{
    event::{Event, WindowEvent},    
    event_loop::{EventLoop, ControlFlow}, 
    window::Window,
};

use super::render::{VertexBuffer, Vertex, path_render, write_buffer};

pub struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
}

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
    /*
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });
    */
    let shader = device.create_shader_module(wgpu::include_wgsl!("shader2.wgsl"));

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });

    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0];
    //let swapchain_format = wgpu::TextureFormat::Rgba16Float;

    let vertex_buffer = VertexBuffer::new(1024, &device);

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[
                vertex_buffer.desc()
            ],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: swapchain_format,

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
    });

    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: swapchain_format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo,
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
        shader,
        queue,
        render_pipeline,
        pipeline_layout,
        vertex_buffer,
    }
}

struct EventLoopArgs {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    config: wgpu::SurfaceConfiguration,
    surface: wgpu::Surface,
    shader: wgpu::ShaderModule,
    queue: wgpu::Queue,
    render_pipeline: wgpu::RenderPipeline,
    pipeline_layout: wgpu::PipelineLayout,
    vertex_buffer: VertexBuffer,
}

fn run_event_loop(event_loop: EventLoop<()>, window: Window, args: EventLoopArgs) {
    let EventLoopArgs {
        instance,
        adapter,
        mut config,
        device,
        surface,
        shader,
        queue,
        render_pipeline,
        pipeline_layout,
        mut vertex_buffer,
    } = args;

    event_loop.run(move |event, _, control_flow| {
        let _ = (&instance, &adapter, &shader, &pipeline_layout);

        *control_flow = ControlFlow::Wait;
        match event {
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                config.width = size.width;
                config.height = size.height;
                surface.configure(&device, &config);
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                render(&surface, &device, &queue, &render_pipeline, &mut vertex_buffer);
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            _ => {}
        }
    });
}

fn render(
    surface: &wgpu::Surface,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    render_pipeline: &wgpu::RenderPipeline,
    vertex_buffer: &mut VertexBuffer,
) {
    let frame = surface
        .get_current_texture()
        .expect("Failed to get next swap chain texture");

    let view = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let vertex_len = path_render(vertex_buffer.as_mut_slice());

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
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

        write_buffer(queue, vertex_buffer, vertex_len);

        rpass.set_pipeline(&render_pipeline);

        //let vertex_len = 3;
        let size = (vertex_len * vertex_buffer.stride()) as u64;
        rpass.set_vertex_buffer(0, vertex_buffer.buffer().slice(..size));
        
        // rpass.draw(0..self.num_vertices, 0..1);
        rpass.draw(0..vertex_len as u32, 0..1);
    }

    queue.submit(Some(encoder.finish()));
    frame.present();
}


pub(crate) fn main_loop() {
    println!("Clip Main");
    let event_loop = EventLoop::new();
    let window = winit::window::Window::new(&event_loop).unwrap();

    env_logger::init();
    let data = pollster::block_on(init_wgpu_args(&window));

    run_event_loop(event_loop, window, data);
}