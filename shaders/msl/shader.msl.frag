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

struct FragmentMain_out
{
    float4 out_var_SV_Target0 [[color(0)]];
};

struct FragmentMain_in
{
    float4 in_var_TEXCOORD0 [[user(locn0)]];
    float3 in_var_TEXCOORD3 [[user(locn3)]];
};

fragment FragmentMain_out FragmentMain(FragmentMain_in in [[stage_in]], constant type_FragmentUniforms& FragmentUniforms [[buffer(0)]])
{
    FragmentMain_out out = {};
    out.out_var_SV_Target0 = float4(mix(in.in_var_TEXCOORD0.xyz, FragmentUniforms.fog_color.xyz, float3(smoothstep(FragmentUniforms.fog_distances.x, FragmentUniforms.fog_distances.y, length(in.in_var_TEXCOORD3)))), in.in_var_TEXCOORD0.w);
    return out;
}
