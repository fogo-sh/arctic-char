#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct TextColor
{
    float4 color;
};

struct main0_out
{
    float4 fragColor [[color(0)]];
};

fragment main0_out main0(constant TextColor& u_color [[buffer(0)]])
{
    main0_out out = {};
    out.fragColor = u_color.color;
    return out;
}

