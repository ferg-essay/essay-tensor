use std::time::Instant;

use essay_plot_base::{driver::FigureApi, Point, CanvasEvent};
use winit::{
    event::{Event, WindowEvent, ElementState, MouseButton },    
    event_loop::{EventLoop, ControlFlow}, 
    window::Window,
};

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
        present_mode: wgpu::PresentMode::Fifo,
        //present_mode: wgpu::PresentMode::AutoVsync,
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

struct MouseState {
    left: ElementState,
    left_press_start: Point,
    left_press_last: Point,
    left_press_time: Instant,

    right: ElementState,
    right_press_start: Point,
    right_press_time: Instant,
}

impl MouseState {
    fn new() -> Self {
        Self {
            left: ElementState::Released,
            left_press_start: Point(0., 0.),
            left_press_last: Point(0., 0.),
            left_press_time: Instant::now(),

            right: ElementState::Released,
            right_press_start: Point(0., 0.),
            right_press_time: Instant::now(),
        }
    }
}

struct CursorState {
    position: Point,
}

impl CursorState {
    fn new() -> Self {
        Self {
            position: Point(0., 0.),
        }
    }
}

fn run_event_loop(
    event_loop: EventLoop<()>, 
    window: Window, 
    args: EventLoopArgs,
    figure: Box<dyn FigureApi>,
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

    let pan_min = 20.;
    let zoom_min = 20.;

    // TODO: is double clicking no longer recommended?
    let dbl_click = 500; // time in millis

    let mut cursor = CursorState::new();
    let mut mouse = MouseState::new();

    event_loop.run(move |event, _, control_flow| {
        let _ = (&instance, &adapter, &figure);

        let mut main_renderer = MainRenderer::new(
            &surface,
            &device,
            &queue,
        );

        let mut is_draw = false;
    
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
            Event::WindowEvent {
                event: WindowEvent::MouseInput {
                    state,
                    button,
                    ..
                },
                ..
            } => {
                match button {
                    MouseButton::Left => {
                        mouse.left = state;

                        if state == ElementState::Pressed {
                            figure.event(
                                &mut figure_renderer,
                                &CanvasEvent::MouseLeftPress(cursor.position),
                            );
                            let now = Instant::now();

                            if now.duration_since(mouse.left_press_time).as_millis() < dbl_click {
                                figure.event(
                                    &mut figure_renderer,
                                    &CanvasEvent::ResetView(cursor.position),
                                )
                            }

                            mouse.left_press_start = cursor.position;
                            mouse.left_press_last = cursor.position;
                            mouse.left_press_time = now;
                        }
                    },
                    MouseButton::Right => {
                        mouse.right = state;

                        match state {
                            ElementState::Pressed => {
                                figure.event(
                                    &mut figure_renderer,
                                    &CanvasEvent::MouseRightPress(cursor.position),
                                );

                                mouse.right_press_start = cursor.position;
                                mouse.right_press_time = Instant::now();
                            }
                            ElementState::Released => {
                                figure.event(
                                    &mut figure_renderer,
                                    &CanvasEvent::MouseRightRelease(cursor.position),
                                );

                                if zoom_min <= mouse.right_press_start.norm(&cursor.position) {
                                    figure.event(
                                        &mut figure_renderer,
                                        &CanvasEvent::ZoomBounds(
                                            mouse.right_press_start, 
                                            cursor.position
                                        )
                                    );
                                }
                            }
                        }
                    },
                    _ => {}
                }
            }
            Event::WindowEvent {
                event: WindowEvent::CursorMoved {
                    position,
                    ..
                },
                ..
            } => {
                cursor.position = Point(position.x as f32, config.height as f32 - position.y as f32);

                if mouse.left == ElementState::Pressed 
                    && pan_min <= mouse.left_press_start.norm(&cursor.position) {
                    figure.event(
                        &mut figure_renderer,
                        &CanvasEvent::Pan(
                            mouse.left_press_start, 
                            mouse.left_press_last, 
                            cursor.position
                        ),
                    );

                    mouse.left_press_last = cursor.position;
                }
                if mouse.right == ElementState::Pressed
                    && pan_min <= mouse.left_press_start.norm(&cursor.position) {
                        figure.event(
                            &mut figure_renderer,
                            &CanvasEvent::MouseRightDrag(mouse.left_press_start, cursor.position),
                    );
                }
            }
            Event::RedrawRequested(_) => {
                //figure_renderer.clear();
                //figure.draw(&mut figure_renderer);
                is_draw = true;
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            _ => {}
        }

        if is_draw || figure_renderer.is_request_redraw() {
            main_renderer.render(|device, queue, view, encoder| {
                figure_renderer.draw(
                    &mut figure,
                    (config.width, config.height),
                    window.scale_factor() as f32,
                    device, 
                    queue, 
                    view, 
                    encoder);
            });
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

pub(crate) fn main_loop(figure: Box<dyn FigureApi>) {
    let event_loop = EventLoop::new();
    let window = winit::window::Window::new(&event_loop).unwrap();

    env_logger::init();
    let wgpu_args = pollster::block_on(init_wgpu_args(&window));

    run_event_loop(event_loop, window, wgpu_args, figure);
}