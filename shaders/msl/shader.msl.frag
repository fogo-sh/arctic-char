#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct UBO
{
    float atlas_width;
    float atlas_height;
    float4 atlas_lookup[4];
};

struct main0_out
{
    float4 frag_color [[color(0)]];
};

struct main0_in
{
    float4 color [[user(locn0)]];
    float3 uvw [[user(locn1)]];
};

fragment main0_out main0(main0_in in [[stage_in]], constant UBO& _42 [[buffer(0)]], texture2d<float> tex_sampler [[texture(0)]], sampler tex_samplerSmplr [[sampler(0)]])
{
    main0_out out = {};
    int texture_index = int(in.uvw.z);
    if ((texture_index < 0) || (texture_index > 3))
    {
        out.frag_color = float4(1.0, 0.0, 0.0, 1.0);
        return out;
    }
    float4 texture_rect = _42.atlas_lookup[texture_index];
    float2 local_uv = fract(in.uvw.xy);
    float section_width = texture_rect.z / _42.atlas_width;
    float section_height = texture_rect.w / _42.atlas_height;
    float u = (texture_rect.x / _42.atlas_width) + (local_uv.x * section_width);
    float v = (texture_rect.y / _42.atlas_height) + (local_uv.y * section_height);
    float2 atlas_uv = float2(u, v);
    out.frag_color = tex_sampler.sample(tex_samplerSmplr, atlas_uv) * in.color;
    return out;
}

