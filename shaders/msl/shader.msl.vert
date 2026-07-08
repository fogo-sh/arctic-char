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
    float3 out_var_TEXCOORD1 [[user(locn1)]];
    float2 out_var_TEXCOORD2 [[user(locn2)]];
    float4 gl_Position [[position]];
};

struct VertexMain_in
{
    float3 in_var_TEXCOORD0 [[attribute(0)]];
    float3 in_var_TEXCOORD1 [[attribute(1)]];
    float2 in_var_TEXCOORD2 [[attribute(2)]];
    float4 in_var_TEXCOORD3 [[attribute(3)]];
};

vertex VertexMain_out VertexMain(VertexMain_in in [[stage_in]], constant type_VertexUniforms& VertexUniforms [[buffer(0)]])
{
    VertexMain_out out = {};
    out.gl_Position = VertexUniforms.mvp * float4(in.in_var_TEXCOORD0, 1.0);
    out.out_var_TEXCOORD0 = in.in_var_TEXCOORD3;
    out.out_var_TEXCOORD1 = in.in_var_TEXCOORD1;
    out.out_var_TEXCOORD2 = in.in_var_TEXCOORD2;
    return out;
}

