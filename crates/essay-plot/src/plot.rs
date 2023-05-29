use essay_tensor::prelude::*;
use plotly::{common::{Mode, MarkerSymbol, Marker}, Layout, layout::Axis};

extern crate plotly;

pub struct Plot {
    plot: plotly::Plot,
    layout: plotly::Layout,
    x_axis: Option<Axis>,
    y_axis: Option<Axis>,
}

impl Plot {
    pub fn new() -> Self {
        Self {
            plot: plotly::Plot::new(),
            layout: Default::default(),
            x_axis: Default::default(),
            y_axis: Default::default(),
        }
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

        let mut trace = plotly::Scatter::new(x, y);

        if let Some(name) = &opt.name {
            trace = trace.name(name);
        }

        trace = trace.marker(opt.marker);

        self.plot.add_trace(trace);

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

        let mut trace = plotly::Scatter::new(x, y);

        trace = trace.mode(Mode::Markers);

        if let Some(name) = &opt.name {
            trace = trace.name(name);
        }

        trace = trace.marker(opt.marker);

        self.plot.add_trace(trace);

        self
    }

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

    pub fn set_title(&mut self, title: &str) -> &mut Self {
        self.layout = self.layout.clone().title(title.into());

        self
    }

    pub fn set_xlabel(&mut self, label: &str) -> &mut Self {
        let axis = match &self.x_axis {
            Some(axis) => axis.clone(),
            None => Axis::new(),
        };
        
        let axis = axis.title(label.into());

        self.x_axis = Some(axis);

        self
    }

    pub fn set_ylabel(&mut self, label: &str) -> &mut Self {
        let axis = match &self.y_axis {
            Some(axis) => axis.clone(),
            None => Axis::new(),
        };
        
        let axis = axis.title(label.into());

        self.y_axis = Some(axis);

        self
    }

    pub fn show(&mut self) -> &mut Self {
        let mut layout = self.layout.clone();

        if let Some(x_axis) = &self.x_axis {
            layout = layout.x_axis(x_axis.clone());
        }

        self.plot.set_layout(layout);

        self.plot.show();

        self
    }
}

pub trait PlotOpt {
    fn marker(self, marker: &str) -> PlotArg;

    fn label(self, name: &str) -> PlotArg;

    fn into(self) -> PlotArg;
}

#[derive(Default)]
pub struct PlotArg {
    name: Option<String>,
    marker: Marker,
}

impl PlotOpt for PlotArg {
    fn marker(self, name: &str) -> Self {
        let marker = match name {
            "." => self.marker.symbol(MarkerSymbol::CircleDot),
            "o" => self.marker.symbol(MarkerSymbol::CircleOpen),
            "*" => self.marker.symbol(MarkerSymbol::AsteriskOpen),
            _ => todo!(),
        };

        Self { marker, ..self }
    }

    fn label(self, name: &str) -> Self {
        Self { name: Some(name.to_owned()), ..self }
    }

    fn into(self) -> Self {
        self
    }
}

impl PlotOpt for () {
    fn marker(self, format: &str) -> PlotArg {
        PlotArg::default().marker(format)
    }

    fn label(self, name: &str) -> PlotArg {
        PlotArg::default().label(name)
    }

    fn into(self) -> PlotArg {
        PlotArg::default()
    }
}


#[cfg(test)]
mod test {
    /*
    extern crate plotly;
    use plotly::common::Mode;
    use plotly::{Plot, Scatter};

    fn line_and_scatter_plot() {
        let trace1 = Scatter::new(vec![1, 2, 3, 4], vec![10, 15, 13, 17])
            .name("trace1")
            .mode(Mode::Lines);

        let mut plot = Plot::new();
        plot.add_trace(trace1);
        plot.show();
    }
    */

    use crate::prelude::*;
    use essay_tensor::prelude::*;

    #[test]
    fn test() {
        println!("Hello");

        let mut plot = Plot::new();

        let x = tf32!([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
        let y = &x * &x + tf32!(1.);

        plot.plot(&x, &y, ().marker("o").label("My Item"));
        plot.scatter(&x, &x * &x * &x, ().marker(".").label("My Item 3"));
        plot.set_title("My Title");
        plot.set_xlabel("My x-axis");

        plot.show();
    }
}