use crate::prelude::Shape;



#[derive(Default)]
pub struct AxisOpt {
    axis: Option<isize>,
}

impl AxisOpt {
    pub fn axis(self, axis: isize) -> Self {
        Self { axis: Some(axis), ..self }
    }

    pub fn get_axis(self) -> Option<isize> {
        self.axis
    }

    pub(crate) fn _axis_with_shape(&self, shape: &Shape) -> usize {
        match self.axis {
            Some(axis) => {
                (axis + shape.rank() as isize) as usize % shape.rank()
            },
            None => 0
        }
    }
}

pub fn axis_from_rank(axis: &Option<isize>, rank: usize) -> usize {
    match axis {
        Some(axis) => (axis + rank as isize) as usize % rank,
        None => 0,
    }
}

pub struct Axis;

impl Axis {
    pub fn axis(axis: isize) -> AxisOpt {
        AxisOpt::default().axis(axis)
    }
}

impl From<Axis> for AxisOpt {
    fn from(_value: Axis) -> Self {
        AxisOpt::default()
    }
}

impl From<()> for AxisOpt {
    fn from(_value: ()) -> Self {
        AxisOpt::default()
    }
}

impl From<isize> for AxisOpt {
    fn from(axis: isize) -> Self {
        AxisOpt::default().axis(axis)
    }
}
