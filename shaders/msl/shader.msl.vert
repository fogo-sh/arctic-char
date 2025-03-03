#pragma clang diagnostic ignored "-Wmissing-prototypes"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct UBO
{
    float4x4 mv;
    float4x4 proj;
    float2 viewport_size;
};

struct main0_out
{
    float4 out_color [[user(locn0)]];
    float3 out_uvw [[user(locn1)]];
    float4 gl_Position [[position]];
};

struct main0_in
{
    float3 position [[attribute(0)]];
    float4 color [[attribute(1)]];
    float3 uvw [[attribute(2)]];
};

static inline __attribute__((always_inline))
float3 quantize(thread const float3& pos, thread const float& scale, constant UBO& _21)
{
    float w = (_21.proj * float4(pos, 1.0)).w;
    return (round((pos / float3(w)) * scale) / float3(scale)) * w;
}

vertex main0_out main0(main0_in in [[stage_in]], constant UBO& _21 [[buffer(0)]])
{
    main0_out out = {};
    float3 viewPos = (_21.mv * float4(in.position, 1.0)).xyz;
    float jitter = 0.5;
    float z_orig = viewPos.z;
    float scale = ((1.0 - jitter) * fast::min(_21.viewport_size.x, _21.viewport_size.y)) / 2.0;
    float3 param = viewPos;
    float param_1 = scale;
    viewPos = quantize(param, param_1, _21);
    viewPos.z = z_orig;
    out.gl_Position = _21.proj * float4(viewPos, 1.0);
    out.out_color = in.color;
    out.out_uvw = in.uvw;
    return out;
}

