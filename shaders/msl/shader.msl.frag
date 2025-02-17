#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct main0_out
{
    float4 color [[color(0)]];
};

fragment main0_out main0(float4 gl_FragCoord [[position]])
{
    main0_out out = {};
    float2 uv = gl_FragCoord.xy / float2(512.0);
    out.color = float4(uv.x, uv.y, 0.5, 1.0);
    return out;
}

