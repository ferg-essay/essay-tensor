pub trait TickFormatter {
    fn format(&self, value: f32) -> String;
}

pub enum Formatter {
    Plain,
}

impl TickFormatter for Formatter {
    fn format(&self, value: f32) -> String {
        match self {
            Formatter::Plain => {
                format!("{:.2}", value)
            }
        }
    }
}

