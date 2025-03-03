#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct main0_out
{
    float4 frag_color [[color(0)]];
};

struct main0_in
{
    float4 color [[user(locn0)]];
    float3 uvw [[user(locn1)]];
};

fragment main0_out main0(main0_in in [[stage_in]], texture2d<float> tex_sampler [[texture(0)]], sampler tex_samplerSmplr [[sampler(0)]])
{
    main0_out out = {};
    float2 uv = in.uvw.xy;
    out.frag_color = tex_sampler.sample(tex_samplerSmplr, uv) * in.color;
    return out;
}

