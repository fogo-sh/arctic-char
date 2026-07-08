#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct type_VertexUniforms
{
    float4x4 mvp;
};

struct FragmentMain_out
{
    float4 out_var_SV_Target0 [[color(0)]];
};

struct FragmentMain_in
{
    float4 in_var_TEXCOORD0 [[user(locn0)]];
};

fragment FragmentMain_out FragmentMain(FragmentMain_in in [[stage_in]])
{
    FragmentMain_out out = {};
    out.out_var_SV_Target0 = in.in_var_TEXCOORD0;
    return out;
}

