use core::fmt;

use essay_plot_base::Point;
use essay_tensor::{Tensor, tensor::TensorVec};

use super::triangulate::Triangulation;

struct TriDelaunay {
    vertices: Vec<Point>, // [n, 2]
    valid_len: usize,

    edges: Vec<Edge>,
    free_edges: Vec<EdgeId>,
}

impl TriDelaunay {
    fn new(points: &Tensor) -> TriDelaunay {
        assert!(points.rank() == 2);
        assert!(points.cols() == 2);
    
        let vertices = initial_points(&points);

        let len = vertices.len();
        let v0 = VertId(len - 3);
        let v1 = VertId(len - 2);
        let v2 = VertId(len - 1);
    
        // edge 0 is outside marker
        let mut edges = Vec::new();
        edges.push(Edge::new(VertId::none(), VertId::none()));

        // TODO: add field for valid vs super triangle cutoff
        let mut tri = Self {
            vertices,
            valid_len: len - 3,
            edges,
            free_edges: Vec::new(),
        };

        // ghost edges first to make them easily identifiable
        let d0 = tri.alloc_edge(v1, v0);
        let d1 = tri.alloc_edge(v2, v1);
        let d2 = tri.alloc_edge(v0, v2);

        let e0 = tri.alloc_edge(v0, v1);
        let e1 = tri.alloc_edge(v1, v2);
        let e2 = tri.alloc_edge(v2, v0);

        tri.update_edge_dual(e0, e1, e2, d0);
        tri.update_edge_dual(d0, d2, d1, e0);

        tri.update_edge_dual(e1, e2, e0, d1);
        tri.update_edge_dual(d1, d0, d2, e1);

        tri.update_edge_dual(e2, e0, e1, d2);
        tri.update_edge_dual(d2, d1, d0, e2);

        tri
    }

    fn build(&mut self) {
        let mut edge = EdgeId(4);

        for i in 0..self.vertices.len() - 3 {
            edge = self.add_vertex(VertId(i), edge);
        }
    }

    fn to_triangulation(&self) -> Triangulation {
        let mut xy_vec = TensorVec::<[f32; 2]>::new();

        let orig_xy = self.vertices.len() - 3;
        // let orig_xy = self.vertices.len();
        for i in 0..orig_xy {
            let point = self.vertices[i];

            xy_vec.push([point.0, point.1]);
        }

        Triangulation::new(xy_vec.into_tensor(), self.triangles())
    }

    fn add_vertex(&mut self, p: VertId, start_edge: EdgeId) -> EdgeId {
        let edge = self.find_enclosing_triangle(p, start_edge);

        let edge = self.create_enclosing_polygon(p, edge);

        self.add_edges(p, edge);

        edge
    }

    ///
    /// Walk triangles toward the new point until finding one that contains
    /// the point.
    /// 
    fn find_enclosing_triangle(&self, v: VertId, start_edge: EdgeId) -> EdgeId {
        let mut edge_id = start_edge;
        loop {
            if self.in_triangle(v, edge_id) {
                return edge_id;
            }

            // walk triangle toward point
            let p = self.vertices[v.index()];

            let [a, b, c] = self.triangle_points(edge_id);

            if edge_sign(p, a, b) < 0. {
                edge_id = self.edges[edge_id.index()].dual;
            } else if edge_sign(p, b, c) < 0. {
                let fwd_id = self.edges[edge_id.index()].forward;
                edge_id = self.edges[fwd_id.index()].dual;
            } else if edge_sign(p, c, a) < 0. {
                let rev_id = self.edges[edge_id.index()].reverse;
                edge_id = self.edges[rev_id.index()].dual;
            } else {
                todo!("Point {:?} (a={:?}, b={:?}, c={:?})", p, a, b, c);
            }
        }
    }

    ///
    /// Create enclosing polygon by removing edges where both triangle
    /// circles contain the new point.
    /// 
    /// The starting edge's triangle is known to contains the new point.
    /// 
    fn create_enclosing_polygon(&mut self, p: VertId, edge_id: EdgeId) -> EdgeId {
        let is_internal = self.is_internal(edge_id);

        let edge = self.edges[edge_id.index()].clone();

        self.check_dual(p, edge.forward, is_internal);
        self.check_dual(p, edge.reverse, is_internal);
        self.check_dual(p, edge_id, is_internal)
    }

    ///
    /// Recursively check the dual triangle of an edge. Since the edge's
    /// triangle circle is known to contain the point, the edge will be 
    /// removed if the dual's circle also contains the point.
    ///
    fn check_dual(&mut self, v: VertId, edge_id: EdgeId, is_internal: bool) -> EdgeId {
        let dual_id = self.edges[edge_id.index()].dual;

        // if point is not in dual circle, return original edge
        if self.in_triangle_circle(v, dual_id) < 0. {
            return edge_id;
        }

        // don't remove an external edge
        if is_internal && ! self.is_internal(dual_id) {
            return edge_id;
        }

        let dual = self.edges[dual_id.index()].clone();

        self.remove_edge(edge_id);

        // recursively check new triangle's duals
        self.check_dual(v, dual.reverse, is_internal);
        self.check_dual(v, dual.forward, is_internal)
    }

    fn remove_edge(&mut self, edge_id: EdgeId) {
        // remove edge
        let edge = self.edges[edge_id.index()].clone();
        let dual_id = edge.dual;
        let dual = self.edges[dual_id.index()].clone();

        self.edges[edge_id.index()].verts[0] = VertId::none();
        self.edges[dual_id.index()].verts[0] = VertId::none();

        self.edges[edge.forward.index()].reverse = dual.reverse;
        self.edges[edge.reverse.index()].forward = dual.forward;

        self.edges[dual.forward.index()].reverse = edge.reverse;
        self.edges[dual.reverse.index()].forward = edge.forward;

        // TODO: check degenerate cases where the triangle collapses to a line

        self.free_edge(dual_id);
        self.free_edge(edge_id);
    }

    ///
    /// add edges/triangles for a new vertex after the old ones are cleared.
    ///
    fn add_edges(&mut self, p: VertId, start_id: EdgeId) {
        // bounding edges

        let mut old_id = start_id;
        let old = self.edge(old_id).clone();

        let edge_0 = self.alloc_edge(p, old.verts[0]);
        let first_0 = edge_0;
        let edge_1 = self.alloc_edge(old.verts[1], p);

        self.update_edge(old_id, edge_1, edge_0);

        self.update_edge_dual(edge_0, old_id, edge_1, EdgeId(0));
        // self.update_edge_dual(prev_1, prev_0, prev_old, edge_0);

        let mut prev_0 = edge_0;
        let mut prev_1 = edge_1;
        let mut prev_old = old_id;

        old_id = old.forward;

        loop {
            // old edge to be updated
            let old = self.edge(old_id).clone();

            let edge_0 = self.alloc_edge(p, old.verts[0]);
            let edge_1 = self.alloc_edge(old.verts[1], p);

            // update old edge
            self.update_edge(old_id, edge_1, edge_0);

            self.update_edge_dual(edge_0, old_id, edge_1, prev_1);
            self.update_edge_dual(prev_1, prev_0, prev_old, edge_0);

            prev_0 = edge_0;
            prev_1 = edge_1;
            prev_old = old_id;

            old_id = old.forward;
    
            if old_id == start_id {
                break
            }
        }

        self.update_edge_dual(prev_1, prev_0, prev_old, first_0);
        self.edge_mut(first_0).dual = prev_1;

        // update final edge
    }

    fn triangles(&self) -> Tensor<usize> {
        let mut vec = TensorVec::<[usize; 3]>::new();

        let ext_edges = 6;
        let tail_vert = self.vertices.len() - 3;
        for i in ext_edges + 1..self.edges.len() {
            let edge = &self.edges[i];

            let v0 = edge.verts[0];
            let v1 = edge.verts[1];
            let v2 = self.edges[edge.forward.index()].verts[1];

            if i < edge.forward.index() && i < edge.reverse.index()
                && v0.index() < tail_vert
                && v1.index() < tail_vert
                && v2.index() < tail_vert {
                let verts = [
                    v0.index(),
                    v1.index(),
                    v2.index(),
                ];

                vec.push(verts);
            }
        }

        vec.into_tensor()
    }

    fn in_triangle_circle(&self, v: VertId, edge_id: EdgeId) -> f32 {
        let p = self.vertices[v.index()];

        let [a, b, c] = self.triangle_points(edge_id);

        in_circle(p, a, b, c)
    }

    fn in_triangle(&self, v: VertId, edge_id: EdgeId) -> bool {
        let p = self.vertices[v.index()];

        let [a, b, c] = self.triangle_points(edge_id);

        in_triangle(p, a, b, c)
    }

    fn is_internal(&self, edge_id: EdgeId) -> bool {
        let edge = &self.edges[edge_id.index()];
        let fwd = &self.edges[edge.forward.index()];
        let valid_len = self.valid_len;

        edge.verts[0].index() < valid_len
        && edge.verts[1].index() < valid_len
        && fwd.verts[1].index() < valid_len
    }

    fn triangle_points(&self, edge_id: EdgeId) -> [Point; 3] {
        let edge = &self.edges[edge_id.index()];
        let v0 = edge.verts[0];
        let v1 = edge.verts[1];
        let v2 = self.edges[edge.forward.index()].verts[1];

        let verts = &self.vertices;

        [verts[v0.index()], verts[v1.index()], verts[v2.index()]]
    }

    fn alloc_edge(&mut self, v0: VertId, v1: VertId) -> EdgeId {
        if let Some(id) = self.free_edges.pop() {
            let edge = &mut self.edges[id.index()];
            edge.verts[0] = v0;
            edge.verts[1] = v1;
            return id;
        }

        let id = EdgeId(self.edges.len());

        self.edges.push(Edge::new(v0, v1));

        id
    }

    fn free_edge(&mut self, id: EdgeId) {
        self.free_edges.push(id);
    }

    #[inline]
    fn update_edge(&mut self, id: EdgeId, forward: EdgeId, reverse: EdgeId) {
        let edge = &mut self.edges[id.index()];

        edge.forward = forward;
        edge.reverse = reverse;
    }

    #[inline]
    fn update_edge_dual(
        &mut self, 
        id: EdgeId, 
        forward: EdgeId, 
        reverse: EdgeId,
        dual: EdgeId,
    ) {
        let edge = &mut self.edges[id.index()];

        edge.forward = forward;
        edge.reverse = reverse;
        edge.dual = dual;
    }

    #[inline]
    fn edge(&self, id: EdgeId) -> &Edge {
        &self.edges[id.index()]
    }

    #[inline]
    fn edge_mut(&mut self, id: EdgeId) -> &mut Edge {
        &mut self.edges[id.index()]
    }
}

pub fn triangulate(points: &Tensor) -> Triangulation {
    let mut tri = TriDelaunay::new(points);
    tri.build();
    // tri.remove_ext_triangle();
    tri.to_triangulation()
}

fn initial_points(points: &Tensor) -> Vec<Point> {
    let (mut x_min, mut x_max) = (f32::MAX, f32::MIN);
    let (mut y_min, mut y_max) = (f32::MAX, f32::MIN);

    let mut vec_points: Vec<Point> = Vec::new();

    for point in points.iter_slice() {
        let (x, y) = (point[0], point[1]);

        vec_points.push(Point(x, y));

        (x_min, x_max) = (x_min.min(x), x_max.max(x));
        (y_min, y_max) = (y_min.min(y), y_max.max(y));
    }

    // TODO: either need more sophisticated sorting (Hilbert curve) and
    // retaining original order, or no sorting at all.
    // vec_points.sort_by(|a, b| a.0.total_cmp(&b.0));

    let x_mid= 0.5 * (x_min + x_max);
    let (w, h) = (x_max - x_min, y_max - y_min);

    vec_points.push(Point(x_mid - w - 1., y_min - 1.));
    vec_points.push(Point(x_mid + w + 1., y_min - 1.));
    vec_points.push(Point(x_mid, y_max + h + 1.));


    vec_points
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct VertId(usize);

impl VertId {
    #[inline]
    fn index(&self) -> usize {
        self.0
    }

    #[inline]
    fn none() -> VertId {
        VertId(usize::MAX)
    }

    #[cfg(test)]
    #[inline]
    fn is_none(&self) -> bool {
        self.0 == usize::MAX
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Hash, Eq)]
pub struct EdgeId(usize);

impl EdgeId {
    #[inline]
    fn index(&self) -> usize {
        self.0
    }
}

#[derive(Clone)]
pub struct Edge {
    verts: [VertId; 2],

    dual: EdgeId,
    forward: EdgeId,
    reverse: EdgeId,
}

impl Edge {
    fn new(v0: VertId, v1: VertId) -> Self {
        Self {
            verts: [v0, v1],
            dual: EdgeId(0),
            forward: EdgeId(0),
            reverse: EdgeId(0),
        }
    }
}

impl fmt::Debug for Edge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Edge[{}, {}](fwd: {}, rev: {}, dual: {})",
            self.verts[0].index(), self.verts[1].index(),
            self.forward.index(),
            self.reverse.index(),
            self.dual.index())
    }
}

/// d in tri-circle(a, b, c) if result > 0
#[inline]
fn in_circle(p: Point, a: Point, b: Point, c: Point) -> f32 {
    let d = p;

    // Cheng 2013, via GW Lucas
    let ad_sq = (a.0 - d.0).powi(2) + (a.1 - d.1).powi(2);
    let bd_sq = (b.0 - d.0).powi(2) + (b.1 - d.1).powi(2);
    let cd_sq = (c.0 - d.0).powi(2) + (c.1 - d.1).powi(2);

    det(
        [a.0 - d.0, a.1 - d.1, ad_sq],
        [b.0 - d.0, b.1 - d.1, bd_sq],
        [c.0 - d.0, c.1 - d.1, cd_sq]
    )
}

#[inline]
fn in_triangle(p: Point, a: Point, b: Point, c: Point) -> bool {
    let d1 = edge_sign(p, a, b);
    let d2 = edge_sign(p, b, c);
    let d3 = edge_sign(p, c, a);

    (d1 < 0.) && (d2 < 0.) && (d3 < 0.)
    || (d1 > 0.) && (d2 > 0.) && (d3 > 0.)
}

/// sign of the half-plane for p in a, b
#[inline]
fn edge_sign(p: Point, a: Point, b: Point) -> f32 {
    (p.0 - b.0) * (a.1 - b.1) - (a.0 - b.0) * (p.1 - b.1)
}

#[inline]
fn det(r0: [f32; 3], r1: [f32; 3], r2: [f32; 3]) -> f32 {
    r0[0] * (r1[1] * r2[2] - r2[1] * r1[2])
        - r0[1] * (r1[0] * r2[2] - r2[0] * r1[2])
        + r0[2] * (r1[0] * r2[1] - r2[0] * r1[1])
}

#[cfg(test)]
mod test {
    use essay_plot_base::Point;
    use essay_tensor::{tf32, Tensor};

    use crate::tri::{delaunay::{TriDelaunay, VertId, EdgeId, in_triangle}, triangulate::Triangulation};

    use super::{initial_points};

    #[test]
    fn init_triangle_points() {
        let t = tf32!([[0., 0.], [1., 1.]]);
        let points = initial_points(&t);
        let (a, b, c) = (points[2], points[3], points[4]);
        assert_in_triangle(&t, a, b, c);

        let t = tf32!([[1000., -1000.], [1001., 0.]]);
        let points = initial_points(&t);
        let (a, b, c) = (points[2], points[3], points[4]);
        assert_in_triangle(&t, a, b, c);

        let t = tf32!([[-1000., 1000.], [-1., 1001.]]);
        let points = initial_points(&t);
        let (a, b, c) = (points[2], points[3], points[4]);
        assert_in_triangle(&t, a, b, c);
    }

    #[test]
    fn init_edges() {
        let t = tf32!([[0., 0.], [1., 1.]]);
        let tri = TriDelaunay::new(&t);
        let edges = &tri.edges;

        assert_eq!(edges.len(), 7);

        // ghost edges first for validation. Its triangle contains infinity
        assert_eq!(format!("{:?}", edges[1]), "Edge[3, 2](fwd: 3, rev: 2, dual: 4)");
        assert_eq!(format!("{:?}", edges[2]), "Edge[4, 3](fwd: 1, rev: 3, dual: 5)");
        assert_eq!(format!("{:?}", edges[3]), "Edge[2, 4](fwd: 2, rev: 1, dual: 6)");

        assert_eq!(edge_str(&tri, 4), "Edge[2, 3](fwd: 5, rev: 6, dual: 1)[-1.5,-1; 2.5,-1]");
        assert_eq!(edge_str(&tri, 5), "Edge[3, 4](fwd: 6, rev: 4, dual: 2)[2.5,-1; 0.5,3]");
        assert_eq!(edge_str(&tri, 6), "Edge[4, 2](fwd: 4, rev: 5, dual: 3)[0.5,3; -1.5,-1]");
    }

    // initial point added to bounding triangles
    #[test]
    fn insert_point() {
        let t = tf32!([[0., 0.], [1., 1.]]);
        let mut tri = TriDelaunay::new(&t);

        tri.add_edges(VertId(0), EdgeId(4));

        let edges = &tri.edges;

        assert_eq!(edges.len(), 13);

        // triangle with edge-4 and new point
        assert_eq!(format!("{:?}", edges[4]), "Edge[2, 3](fwd: 8, rev: 7, dual: 1)");
        assert_eq!(format!("{:?}", edges[7]), "Edge[0, 2](fwd: 4, rev: 8, dual: 12)");
        assert_eq!(format!("{:?}", edges[8]), "Edge[3, 0](fwd: 7, rev: 4, dual: 9)");

        // triangle with edge-5 and new point
        assert_eq!(format!("{:?}", edges[5]), "Edge[3, 4](fwd: 10, rev: 9, dual: 2)");
        assert_eq!(format!("{:?}", edges[9]), "Edge[0, 3](fwd: 5, rev: 10, dual: 8)");
        assert_eq!(format!("{:?}", edges[10]), "Edge[4, 0](fwd: 9, rev: 5, dual: 11)");

        // triangle with edge-6 and new point
        assert_eq!(format!("{:?}", edges[6]), "Edge[4, 2](fwd: 12, rev: 11, dual: 3)");
        assert_eq!(format!("{:?}", edges[11]), "Edge[0, 4](fwd: 6, rev: 12, dual: 10)");
        assert_eq!(format!("{:?}", edges[12]), "Edge[2, 0](fwd: 11, rev: 6, dual: 7)");
    }

    // test if point in triangle-circle marked by an edge
    #[test]
    fn in_triangle_circle() {
        let t = tf32!([[0., 0.], [1., 1.]]);
        let tri = TriDelaunay::new(&t);

        // p0 inside bounding triangle
        assert!(tri.in_triangle_circle(VertId(0), EdgeId(4)) > 0.);
        assert!(tri.in_triangle_circle(VertId(0), EdgeId(5)) > 0.);
        assert!(tri.in_triangle_circle(VertId(0), EdgeId(6)) > 0.);

        // p1 inside bounding triangle
        assert!(tri.in_triangle_circle(VertId(1), EdgeId(4)) > 0.);
        assert!(tri.in_triangle_circle(VertId(1), EdgeId(5)) > 0.);
        assert!(tri.in_triangle_circle(VertId(1), EdgeId(6)) > 0.);

        // outside ghost outer triangle
        assert!(tri.in_triangle_circle(VertId(0), EdgeId(1)) < 0.);
        assert!(tri.in_triangle_circle(VertId(0), EdgeId(2)) < 0.);
        assert!(tri.in_triangle_circle(VertId(0), EdgeId(3)) < 0.);

    }

    // find enclosing triangle
    #[test]
    fn find_enclosing_triangle_initial() {
        let t = tf32!([[0., 0.], [1., 1.]]);
        let tri = TriDelaunay::new(&t);

        // p0 inside bounding triangle
        assert_eq!(tri.find_enclosing_triangle(VertId(0), EdgeId(4)), EdgeId(4));
        // currently not normalized
        assert_eq!(tri.find_enclosing_triangle(VertId(0), EdgeId(5)), EdgeId(5));
        assert_eq!(tri.find_enclosing_triangle(VertId(0), EdgeId(6)), EdgeId(6));
    }

    // create enclosing polygon
    #[test]
    fn create_enclosing_polygon_initial() {
        let t = tf32!([[0., 0.], [1., 1.]]);
        let mut tri = TriDelaunay::new(&t);

        // p0 inside bounding triangle
        assert_eq!(tri.create_enclosing_polygon(VertId(0), EdgeId(4)), EdgeId(4));
    }

    // insert enclosing polygon
    #[test]
    fn add_point_initial() {
        let t = tf32!([[0., 0.], [1., 1.]]);
        let mut tri = TriDelaunay::new(&t);

        tri.add_vertex(VertId(0), EdgeId(4));

        let edges = &tri.edges;

        assert_eq!(edges.len(), 13);

        // triangle with edge-4 and new point
        assert_eq!(format!("{:?}", edges[4]), "Edge[2, 3](fwd: 8, rev: 7, dual: 1)");
        assert_eq!(format!("{:?}", edges[7]), "Edge[0, 2](fwd: 4, rev: 8, dual: 12)");
        assert_eq!(format!("{:?}", edges[8]), "Edge[3, 0](fwd: 7, rev: 4, dual: 9)");

        // triangle with edge-5 and new point
        assert_eq!(format!("{:?}", edges[5]), "Edge[3, 4](fwd: 10, rev: 9, dual: 2)");
        assert_eq!(format!("{:?}", edges[9]), "Edge[0, 3](fwd: 5, rev: 10, dual: 8)");
        assert_eq!(format!("{:?}", edges[10]), "Edge[4, 0](fwd: 9, rev: 5, dual: 11)");

        // triangle with edge-6 and new point
        assert_eq!(format!("{:?}", edges[6]), "Edge[4, 2](fwd: 12, rev: 11, dual: 3)");
        assert_eq!(format!("{:?}", edges[11]), "Edge[0, 4](fwd: 6, rev: 12, dual: 10)");
        assert_eq!(format!("{:?}", edges[12]), "Edge[2, 0](fwd: 11, rev: 6, dual: 7)");
    }

    // full build with two points
    #[test]
    fn two_points() {
        let t = tf32!([[0., 1.], [2., 1.]]);
        let mut tri = TriDelaunay::new(&t);

        tri.build();
        validate_edges(&tri);

        for index in 0..tri.edges.len() {
            println!("{}: {}", index, edge_str(&tri, index));
        }
    }

    // full build with two points
    #[test]
    fn three_points() {
        let t = tf32!([[0., 1.], [2., 1.], [1., 2.]]);
        let mut tri_build = TriDelaunay::new(&t);

        tri_build.build();
        validate_edges(&tri_build);

        let tri = tri_build.to_triangulation();

        let triangles = tri.triangles();
        assert_eq!(triangles.rows(), 1);

        assert_eq!(
            tri_str(&tri, triangles.slice(0).as_slice()),
            "[(0, 1), (2, 1), (1, 2)]"
        );

        assert_eq!(
            format!("{:?}", triangles.slice(0).as_slice()),
            "[0, 1, 2]"
        );
    }

    // full build with two points
    #[test]
    fn quad() {
        let t = tf32!([[0., 0.], [4., 0.], [2., 4.], [1.5, 2.]]);
        let mut tri_build = TriDelaunay::new(&t);

        tri_build.build();
        validate_edges(&tri_build);

        let tri = tri_build.to_triangulation();

        let triangles = tri.triangles();
        assert_eq!(triangles.rows(), 2);

        assert_eq!(
            format!("{:?}", triangles.slice(0).as_slice()),
            "[0, 1, 3]"
        );

        assert_eq!(
            format!("{:?}", triangles.slice(0).as_slice()),
            "[2, 0, 3]"
        );

        assert_eq!(
            format!("{:?}", triangles.slice(0).as_slice()),
            "[1, 2, 3]"
        );
    }

    fn tri_str(tri: &Triangulation, triangle: &[usize]) -> String {
        let xy = tri.vertices();

        let (x0, y0) = (xy[(triangle[0], 0)], xy[(triangle[0], 1)]);
        let (x1, y1) = (xy[(triangle[1], 0)], xy[(triangle[1], 1)]);
        let (x2, y2) = (xy[(triangle[2], 0)], xy[(triangle[2], 1)]);

        format!("[({}, {}), ({}, {}), ({}, {})]", x0, y0, x1, y1, x2, y2)
    }

    fn validate_edges(tri: &TriDelaunay) {
        for i in 0..tri.edges.len() {
            let edge = tri.edges[i].clone();

            if edge.verts[0].is_none() {
                continue;
            }

            assert!(! edge.verts[1].is_none());

            let dual = &tri.edges[edge.dual.index()];
            assert_eq!(i, dual.dual.index());
            assert_eq!(edge.verts[1], dual.verts[0]);
            assert_eq!(edge.verts[0], dual.verts[1]);

            let fwd = &tri.edges[edge.forward.index()];
            assert_eq!(i, fwd.reverse.index());
            assert_eq!(edge.verts[1], fwd.verts[0]);

            let rev = &tri.edges[edge.reverse.index()];
            assert_eq!(i, rev.forward.index());
            assert_eq!(edge.verts[0], rev.verts[1]);
        }
    }

    fn assert_in_triangle(tensor: &Tensor, a: Point, b: Point, c: Point) {
        for xy in tensor.iter_slice() {
            assert!(in_triangle(Point(xy[0], xy[1]), a, b, c))
        }
    }
    
    fn edge_str(tri: &TriDelaunay, index: usize) -> String {
        let edge = tri.edges[index].clone();

        let v0 = edge.verts[0];
        let p0 = if v0.is_none() { 
            return format!("None");
        } else { 
            tri.vertices[v0.index()]
        };

        let v1 = edge.verts[1];
        let p1 = if v1.is_none() { 
            return format!("None");
        } else { 
            tri.vertices[v1.index()]
        };

        format!("{:?}[{},{}; {},{}]", tri.edges[index], p0.0, p0.1, p1.0, p1.1)
    }
}