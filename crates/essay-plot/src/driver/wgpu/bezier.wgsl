struct VertexInput {
    @location(0) pos: vec2<f32>,
    @location(1) tex_coord: vec2<f32>,
    @location(2) color: u32,
}

struct VertexOutput {
    @location(0) tex_coord: vec2<f32>,
    @location(1) color: vec4<f32>,
    @builtin(position) clip_position: vec4<f32>,
};

fn unpack_color(color: u32) -> vec4<f32> {
    return vec4<f32>(
        f32((color >> 24u) & 0xffu),
        f32((color >> 16u) & 0xffu),
        f32((color >> 8u) & 0xffu),
        f32(color & 0xffu),
    ) / 255.0;
}

@vertex
fn vs_bezier(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coord = model.tex_coord;
    out.clip_position = vec4<f32>(model.pos, 0.0, 1.0);
    out.color = unpack_color(model.color);
    return out;
}

@fragment
fn fs_bezier(
    in: VertexOutput,
) -> @location(0) vec4<f32> {
    let u = in.tex_coord[0];
    let v = in.tex_coord[1];
    if u * u <= v {
        return in.color;
    } else {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
}
