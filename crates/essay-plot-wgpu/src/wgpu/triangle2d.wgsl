//
// 2d triangles for grids and meshes.
//
// Grids with isolated (solid) colors don't share vertices in the triangles.
// Grids with bleeding colors do share vertices in the triangles.
//

struct VertexInput {
    @location(0) pos: vec2<f32>,
    @location(1) color: u32,
}

struct StyleInput {
    @location(2) a0: vec4<f32>,
    @location(3) a1: vec4<f32>,
}

struct VertexOutput {
    @location(0) color: vec4<f32>,
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
fn vs_triangle(
    model: VertexInput,
    style: StyleInput,
) -> VertexOutput {
    let xp = model.pos[0];
    let yp = model.pos[1];
    let a0 = style.a0;
    let a1 = style.a1;

    let x = a0[0] * xp + a0[1] * yp + a0[3];
    let y = a1[0] * xp + a1[1] * yp + a1[3];

    var out: VertexOutput;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    //out.color = vec4<f32>(0.0, 0.0, 0.0, 1.0); // unpack_color(model.color);
    out.color = unpack_color(model.color);
    return out;
}

@fragment
fn fs_triangle(
    in: VertexOutput,
) -> @location(0) vec4<f32> {
    return in.color;
}
