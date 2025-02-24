#pragma clang diagnostic ignored "-Wmissing-prototypes"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct UBO
{
    float4x4 mvp;
    float time;
};

struct main0_out
{
    float4 out_color [[user(locn0)]];
    float2 out_uv [[user(locn1)]];
    float4 gl_Position [[position]];
};

struct main0_in
{
    float3 position [[attribute(0)]];
    float4 color [[attribute(1)]];
    float2 uv [[attribute(2)]];
};

static inline __attribute__((always_inline))
float random(thread const float3& seed)
{
    return fract(sin(dot(seed, float3(12.98980045318603515625, 78.233001708984375, 45.542999267578125))) * 43758.546875);
}

static inline __attribute__((always_inline))
float3 quantize(thread const float3& pos, thread const float& scale)
{
    return floor(pos * scale) / float3(scale);
}

vertex main0_out main0(main0_in in [[stage_in]], constant UBO& _51 [[buffer(0)]])
{
    main0_out out = {};
    float quantizationLevel = 16.0;
    float jitterAmount = 0.006000000052154064178466796875;
    float stepSize = 0.5;
    float discreteTime = floor(_51.time / stepSize) * stepSize;
    float3 param = in.position + float3(discreteTime);
    float3 param_1 = in.position.yzx + float3(discreteTime);
    float3 param_2 = in.position.zxy + float3(discreteTime);
    float3 jitter = float3(random(param) - 0.5, random(param_1) - 0.5, random(param_2) - 0.5) * jitterAmount;
    float3 param_3 = in.position + jitter;
    float param_4 = quantizationLevel;
    float3 quantizedPos = quantize(param_3, param_4);
    out.gl_Position = _51.mvp * float4(quantizedPos, 1.0);
    out.out_color = in.color;
    out.out_uv = in.uv;
    return out;
}

