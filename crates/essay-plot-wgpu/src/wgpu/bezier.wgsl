struct VertexInput {
    @location(0) pos: vec2<f32>,
    @location(1) tex_coord: vec3<f32>,
}

struct StyleInput {
    @location(2) a0: vec4<f32>,
    @location(3) a1: vec4<f32>,
    @location(4) color: vec4<f32>,
}

struct VertexOutput {
    @location(0) tex_coord: vec3<f32>,
    @location(1) color: vec4<f32>,
    @builtin(position) pos: vec4<f32>,
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
    style: StyleInput,
) -> VertexOutput {
    let a0 = style.a0;
    let a1 = style.a1;
    let xp = model.pos[0];
    let yp = model.pos[1];
    let x = a0[0] * xp + a0[1] * yp + a0[3];
    let y = a1[0] * xp + a1[1] * yp + a1[3];
    var out: VertexOutput;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.tex_coord = model.tex_coord;
    out.color = style.color;
    return out;
}

@fragment
fn fs_bezier(
    in: VertexOutput,
) -> @location(0) vec4<f32> {
    let u = in.tex_coord[0];
    let v = in.tex_coord[1];
    let w = in.tex_coord[2];
    let u_sq = u * u;
    //if u * u < 1. - v {
    if w <= u_sq && u_sq <= v {
        return in.color; // vec4<f32>(0.0, 1.0, 1.1, 1.0); // in.color;
    } else {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
}

@fragment
fn fs_bezier_rev(
    in: VertexOutput,
) -> @location(0) vec4<f32> {
    let u = in.tex_coord[0];
    let v = in.tex_coord[1];
    //if u * u < 1. - v {
    if v < u * u {
        return in.color; // vec4<f32>(0.0, 1.0, 1.1, 1.0); // in.color;
    } else {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
}
