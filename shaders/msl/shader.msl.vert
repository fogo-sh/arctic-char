#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct type_VertexUniforms
{
    float4x4 mvp;
};

struct VertexMain_out
{
    float4 out_var_TEXCOORD0 [[user(locn0)]];
    float4 gl_Position [[position]];
};

struct VertexMain_in
{
    float3 in_var_TEXCOORD0 [[attribute(0)]];
    float4 in_var_TEXCOORD1 [[attribute(1)]];
};

vertex VertexMain_out VertexMain(VertexMain_in in [[stage_in]], constant type_VertexUniforms& VertexUniforms [[buffer(0)]])
{
    VertexMain_out out = {};
    out.gl_Position = VertexUniforms.mvp * float4(in.in_var_TEXCOORD0, 1.0);
    out.out_var_TEXCOORD0 = in.in_var_TEXCOORD1;
    return out;
}

