#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct type_WorldVertexUniforms
{
    float4x4 mvp;
    float4x4 model_view;
};

struct type_SkyVertexUniforms
{
    float4x4 inv_view;
    float4x4 inv_proj;
};

struct type_FragmentUniforms
{
    float4 fog_color;
    float4 sky_top_color;
    float4 sky_horizon_color;
    float4 fog_distances;
};

struct VertexMain_out
{
    float4 out_var_TEXCOORD0 [[user(locn0)]];
    float3 out_var_TEXCOORD1 [[user(locn1)]];
    float2 out_var_TEXCOORD2 [[user(locn2)]];
    float3 out_var_TEXCOORD3 [[user(locn3)]];
    float4 gl_Position [[position]];
};

struct VertexMain_in
{
    float3 in_var_TEXCOORD0 [[attribute(0)]];
    float3 in_var_TEXCOORD1 [[attribute(1)]];
    float2 in_var_TEXCOORD2 [[attribute(2)]];
    float4 in_var_TEXCOORD3 [[attribute(3)]];
};

vertex VertexMain_out VertexMain(VertexMain_in in [[stage_in]], constant type_WorldVertexUniforms& WorldVertexUniforms [[buffer(0)]])
{
    VertexMain_out out = {};
    float4 _48 = float4(in.in_var_TEXCOORD0, 1.0);
    out.gl_Position = WorldVertexUniforms.mvp * _48;
    out.out_var_TEXCOORD0 = in.in_var_TEXCOORD3;
    out.out_var_TEXCOORD1 = in.in_var_TEXCOORD1;
    out.out_var_TEXCOORD2 = in.in_var_TEXCOORD2;
    out.out_var_TEXCOORD3 = (WorldVertexUniforms.model_view * _48).xyz;
    return out;
}
