use essay_tensor::prelude::*;

use egui::{plot, epaint::Hsva};

use crate::backend::{Backend, BackendErr};

use super::main_loop;

pub struct EguiBackend {
}

impl EguiBackend {
    pub fn new() -> Self {
        Self {
        }
    }
}

impl Backend for EguiBackend {
    fn main_loop(&mut self) -> Result<(), BackendErr> {
        let main_loop = main_loop::MainLoop::new();

        main_loop.run(move |ui| {
        }).unwrap();

        Ok(())
    }
}

pub struct Plot {
    x: Tensor,
    y: Tensor,
    //plot: plotly::Plot,
    //layout: plotly::Layout,
    //x_axis: Option<Axis>,
    //y_axis: Option<Axis>,
}

impl Plot {
    pub fn new(x: impl Into<Tensor>, y: impl Into<Tensor>) -> Self {
        Self {
            x: x.into(),
            y: y.into(),
            //plot: plotly::Plot::new(),
            //layout: Default::default(),
            //x_axis: Default::default(),
            //y_axis: Default::default(),
        }
    }

    fn points(&self) -> Vec::<[f64; 2]> {
        (0..self.x.len()).map(|i| {
            [self.x[i] as f64, self.y[i] as f64]
        }).collect()
    }

    pub fn plot(
        &mut self, 
        x: impl Into<Tensor>, 
        y: impl Into<Tensor>, 
        opt: impl PlotOpt
    ) -> &mut Self {
        let x : Tensor = x.into();
        let y : Tensor = y.into();

        let opt = opt.into();

        let x = Vec::from(x.as_slice());
        let y = Vec::from(y.as_slice());

        /*
        let mut trace = plotly::Scatter::new(x, y);

        if let Some(name) = &opt.name {
            trace = trace.name(name);
        }

        trace = trace.marker(opt.marker);

        self.plot.add_trace(trace);
        */

        self
    }

    pub fn scatter(
        &mut self, 
        x: impl Into<Tensor>, 
        y: impl Into<Tensor>, 
        opt: impl PlotOpt
    ) -> &mut Self {
        let x : Tensor = x.into();
        let y : Tensor = y.into();

        let opt = opt.into();

        let x = Vec::from(x.as_slice());
        let y = Vec::from(y.as_slice());

        /*
        let mut trace = plotly::Scatter::new(x, y);

        trace = trace.mode(Mode::Markers);

        if let Some(name) = &opt.name {
            trace = trace.name(name);
        }

        trace = trace.marker(opt.marker);

        self.plot.add_trace(trace);
        */

        self
    }

        /*
    fn marker(trace: Box<plotly::Scatter<f32, f32>>, format: &str) -> Box<plotly::Scatter<f32, f32>> {
        let mut marker = Marker::new();
        
        match format {
            "." => marker = marker.symbol(MarkerSymbol::CircleDot),
            "o" => marker = marker.symbol(MarkerSymbol::CircleOpen),
            "*" => marker = marker.symbol(MarkerSymbol::AsteriskOpen),
            _ => todo!(),
        }

        trace.marker(marker)
    }
        */

    pub fn set_title(&mut self, title: &str) -> &mut Self {
        //self.layout = self.layout.clone().title(title.into());

        self
    }

    pub fn set_xlabel(&mut self, label: &str) -> &mut Self {
        /*
        let axis = match &self.x_axis {
            Some(axis) => axis.clone(),
            None => Axis::new(),
        };
        
        let axis = axis.title(label.into());

        self.x_axis = Some(axis);
        */

        self
    }

    pub fn set_ylabel(&mut self, label: &str) -> &mut Self {
        /*
        let axis = match &self.y_axis {
            Some(axis) => axis.clone(),
            None => Axis::new(),
        };
        
        let axis = axis.title(label.into());

        self.y_axis = Some(axis);
        */

        self
    }

    pub fn show(&mut self) -> &mut Self {
        //self.plot.show();
        self.main_loop();

        self
    }

    fn main_loop(&mut self) {
        let main_loop = main_loop::MainLoop::new();
        main_loop.run(move |ui| {

        }).unwrap()
    }

    fn draw(&self, ui: &mut egui::Ui) {
        todo!()
    }
}

pub trait PlotOpt {
    // fn marker(self, marker: &str) -> PlotArg;

    fn label(self, name: &str) -> PlotArg;

    fn into(self) -> PlotArg;
}

#[derive(Default)]
pub struct PlotArg {
    name: Option<String>,
    //marker: Marker,
}

impl PlotOpt for PlotArg {
    /*
    fn marker(self, name: &str) -> Self {
        let marker = match name {
            "." => self.marker.symbol(MarkerSymbol::CircleDot),
            "o" => self.marker.symbol(MarkerSymbol::CircleOpen),
            "*" => self.marker.symbol(MarkerSymbol::AsteriskOpen),
            _ => todo!(),
        };

        Self { marker, ..self }
    }
    */

    fn label(self, name: &str) -> Self {
        Self { name: Some(name.to_owned()), ..self }
    }

    fn into(self) -> Self {
        self
    }
}

impl PlotOpt for () {
    /*
    fn marker(self, format: &str) -> PlotArg {
        PlotArg::default().marker(format)
    }
    */

    fn label(self, name: &str) -> PlotArg {
        PlotArg::default().label(name)
    }

    fn into(self) -> PlotArg {
        PlotArg::default()
    }
}
