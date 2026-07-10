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

struct SkyFragmentMain_out
{
    float4 out_var_SV_Target0 [[color(0)]];
};

struct SkyFragmentMain_in
{
    float3 in_var_TEXCOORD0 [[user(locn0)]];
};

fragment SkyFragmentMain_out SkyFragmentMain(SkyFragmentMain_in in [[stage_in]], constant type_FragmentUniforms& FragmentUniforms [[buffer(0)]])
{
    SkyFragmentMain_out out = {};
    float3 _33 = fast::normalize(in.in_var_TEXCOORD0);
    float _34 = _33.y;
    out.out_var_SV_Target0 = float4(mix(mix(FragmentUniforms.sky_horizon_color.xyz, FragmentUniforms.sky_top_color.xyz, float3(fast::clamp((_34 * 0.5) + 0.5, 0.0, 1.0))), FragmentUniforms.fog_color.xyz, float3((1.0 - smoothstep(0.0, 0.3499999940395355224609375, abs(_34))) * 0.3499999940395355224609375)), 1.0);
    return out;
}
