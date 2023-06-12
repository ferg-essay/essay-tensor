use std::ops::Index;

use essay_plot_base::Point;

#[derive(Clone, Debug, PartialEq)]
pub struct Triangle(Point, Point, Point);

pub fn tesselate(points: Vec<Point>) -> Vec<Triangle> {
    let mut points = points;
    let mut triangles = Vec::<Triangle>::new();
    
    let mut index = 0;
    let mut index_start = index;
    while points.len() >= 3 {
        let len = points.len();

        let p0 = points[index];
        let p1 = points[(index + 1) % len];
        let p2 = points[(index + 2) % len];

        let triangle = Triangle(p0, p1, p2);

        if triangle.is_inside(&points) {
            triangles.push(triangle);

            points.remove((index + 1) % len);

            index_start = index;
        }

        index = (index + 1) % points.len();
        assert_ne!(index, index_start);
    }

    triangles
}

impl Triangle {
    fn is_inside(&self, polygon: &Vec<Point>) -> bool {
        let center = Point(
            (self.0.x() + self.1.x() + self.2.x()) / 3.,
            (self.0.y() + self.1.y() + self.2.y()) / 3.,
        );

        let mut n_crosses = 0;
        for i in 0..polygon.len() - 1 {
            let p0 = polygon[i];
            let p1 = polygon[i + 1];
            if center.is_below(&p0, &p1) {
                n_crosses += 1;
            }
        }

        let p0 = polygon[polygon.len() - 1];
        let p1 = polygon[0];

        if center.is_below(&p0, &p1) {
            n_crosses += 1;
        }

        n_crosses % 2 == 1
    }
}

impl Index<usize> for Triangle {
    type Output = Point;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.0,
            1 => &self.1,
            2 => &self.2,
            _ => panic!("Invalid index")
        }
    }
}