use winit::{
    event::{Event, WindowEvent},    
    event_loop::{EventLoop, ControlFlow}, 
    window::Window,
};

use crate::figure::FigureInner;

use super::{render::{FigureRenderer}};

async fn init_wgpu_args(window: &Window) -> EventLoopArgs {
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

    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let texture_format = swapchain_capabilities.formats[0];

    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: texture_format,
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
        queue,
    }
}

struct EventLoopArgs {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    surface: wgpu::Surface,
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
    } = args;

    let mut figure = figure;


    //let mut staging_belt = wgpu::util::StagingBelt::new(1024);

    let mut figure_renderer = FigureRenderer::new(
        &device,
        config.format,
    );

    event_loop.run(move |event, _, control_flow| {
        let _ = (&instance, &adapter, &figure);

        let mut main_renderer = MainRenderer::new(
            &surface,
            &device,
            &queue,
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
                // figure_renderer.set_canvas_bounds(config.width, config.height);
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                //figure_renderer.clear();
                //figure.draw(&mut figure_renderer);
                main_renderer.render(|device, queue, view, encoder| {
                    figure_renderer.draw(
                        &mut figure,
                        (config.width, config.height),
                        device, 
                        queue, 
                        view, 
                        encoder);
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
}

impl<'a> MainRenderer<'a> {
    pub(crate) fn new(
        surface: &'a wgpu::Surface,
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
    ) -> Self {
        Self {
            surface,
            device,
            queue,
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
            
            let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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