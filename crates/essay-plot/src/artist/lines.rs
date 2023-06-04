use egui::Rect;
use essay_tensor::Tensor;

pub struct Lines2D {
    lines: Tensor, // 2d tensor representing a graph
    bbox: Rect, // bounding box
}