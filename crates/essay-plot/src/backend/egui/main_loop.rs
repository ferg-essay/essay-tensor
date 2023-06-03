use eframe::{egui};
use egui::Ui;

pub struct MainLoop {
    title: String,
    width: u32,
    height: u32,

    container: Box<UiContainer>,
}

pub trait UiRender {
    fn render(&self, ui: &mut egui::Ui);
}

pub trait UiBuilder {
    fn build(&mut self) -> Box<dyn UiRender>;
}

//
// # MainLoop
//

impl MainLoop {
    pub fn new() -> MainLoop {
        Self {
            title: String::from("Title"),
            width: 800,
            height: 600,

            container: ui_container(),
        }
    }

    pub fn run(
        self,
        render: impl FnMut(&mut Ui) + 'static
    ) -> Result<(), String> {
        let options = eframe::NativeOptions {
            initial_window_size: Some(egui::vec2(self.width as f32, self.height as f32)),
            ..Default::default()
        };

        let inner = MainLoopInner {
            render: Box::new(render),
        };

        eframe::run_native(
            &self.title,
            options,
            Box::new(|_cc| Box::new(inner)),
        ).unwrap();

        Ok(())
    }

    pub fn add(&mut self, ui_item: Box<dyn UiBuilder>) {
        self.container.add(ui_item);
    }
}

struct MainLoopInner {
    render: Box<dyn FnMut(&mut Ui)>,
}

impl eframe::App for MainLoopInner {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            (self.render)(ui);
            /*
            ui.heading("Hello GUI");
            ui.horizontal(|ui| {
                let name_label = ui.label("Name: ");
                ui.text_edit_singleline(&mut self.name)
                    .labelled_by(name_label.id);
            });
            ui.label(format!("Hello '{}'", self.name))
            */
        });
    }
}

//
// # UiContainer
//

pub fn ui_container() -> Box<UiContainer> {
    Box::new(UiContainer {
        items: Vec::new(),
    })
}

pub struct UiContainer {
    items: Vec<Box<dyn UiBuilder + 'static>>,
}

impl UiContainer {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
        }
    }

    pub fn add(&mut self, item: Box<dyn UiBuilder>) {
        self.items.push(item);
    }
}

impl UiBuilder for UiContainer {
    fn build(&mut self) -> Box<dyn UiRender> {
        let mut items: Vec<Box<dyn UiRender>> = Vec::new();

        for mut item in &mut self.items.drain(..) {
            items.push(item.build());
        }

        Box::new(UiContainerRender {
            items: items,
        })
    }
}

struct UiContainerRender {
    items: Vec<Box<dyn UiRender>>,
}

impl UiRender for UiContainerRender {
    fn render(&self, ui: &mut egui::Ui) {
        for item in &self.items {
            item.render(ui);
        }
    }
}

//
// # UiLabel
//

pub fn ui_label(msg: &str) -> Box<dyn UiBuilder> {
    Box::new(UiWrapperBuilder::new(Box::new(UiLabel { msg: String::from(msg)})))
}

struct UiLabel {
    msg: String,
}

impl UiRender for UiLabel {
    fn render(&self, ui: &mut egui::Ui) {
        ui.label(&self.msg);
    }
}

//
// # UiWrapperBuilder
//

pub struct UiWrapperBuilder {
    item: Option<Box<dyn UiRender>>,
}

impl UiWrapperBuilder {
    pub fn new(item: Box<dyn UiRender>) -> Self {
        Self {
            item: Some(item),
        }
    }
}

impl UiBuilder for UiWrapperBuilder {
    fn build(&mut self) -> Box<dyn UiRender> {
        self.item.take().unwrap()
    }
}
