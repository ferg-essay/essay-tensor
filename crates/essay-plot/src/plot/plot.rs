use essay_tensor::Tensor;

use crate::prelude::Figure;

pub fn plot(
    x: impl Into<Tensor>, 
    y: impl Into<Tensor>, 
    opt: impl Into<PlotOpt>
) -> Figure {
    todo!()
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
        opt: impl Into<PlotOpt>
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
        opt: impl Into<PlotOpt>
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
        todo!()
    }
}

#[derive(Default)]
pub struct PlotOpt {
}

impl PlotOpt {
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

    /*
    fn label(self, name: &str) -> Self {
        Self { name: Some(name.to_owned()), ..self }
    }

    fn into(self) -> Self {
        self
    }
    */    
}

impl From<()> for PlotOpt {
    fn from(_value: ()) -> Self {
        PlotOpt::default()
    }
}
