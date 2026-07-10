#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct type_UiUniforms
{
    float4 ui_rect;
    float4 ui_color;
    float4 ui_corner_radii;
    float4 ui_border_widths;
    float2 ui_viewport;
    float2 ui_padding;
};

struct UiFragmentMain_out
{
    float4 out_var_SV_Target0 [[color(0)]];
};

struct UiFragmentMain_in
{
    float2 in_var_TEXCOORD0 [[user(locn0)]];
};

fragment UiFragmentMain_out UiFragmentMain(UiFragmentMain_in in [[stage_in]], constant type_UiUniforms& UiUniforms [[buffer(0)]])
{
    UiFragmentMain_out out = {};
    float4 _42 = precise::min(UiUniforms.ui_corner_radii, float4(precise::min(UiUniforms.ui_rect.z, UiUniforms.ui_rect.w) * 0.5));
    float _43 = _42.x;
    bool _50;
    if (_43 > 0.0)
    {
        _50 = in.in_var_TEXCOORD0.x < _43;
    }
    else
    {
        _50 = false;
    }
    bool _56;
    if (_50)
    {
        _56 = in.in_var_TEXCOORD0.y < _43;
    }
    else
    {
        _56 = false;
    }
    float _63;
    if (_56)
    {
        _63 = step(length(in.in_var_TEXCOORD0 - float2(_43)), _43);
    }
    else
    {
        _63 = 1.0;
    }
    float _64 = _42.y;
    bool _72;
    if (_64 > 0.0)
    {
        _72 = in.in_var_TEXCOORD0.x > (UiUniforms.ui_rect.z - _64);
    }
    else
    {
        _72 = false;
    }
    bool _78;
    if (_72)
    {
        _78 = in.in_var_TEXCOORD0.y < _64;
    }
    else
    {
        _78 = false;
    }
    float _87;
    if (_78)
    {
        _87 = _63 * step(length(in.in_var_TEXCOORD0 - float2(UiUniforms.ui_rect.z - _64, _64)), _64);
    }
    else
    {
        _87 = _63;
    }
    float _88 = _42.z;
    bool _95;
    if (_88 > 0.0)
    {
        _95 = in.in_var_TEXCOORD0.x < _88;
    }
    else
    {
        _95 = false;
    }
    bool _102;
    if (_95)
    {
        _102 = in.in_var_TEXCOORD0.y > (UiUniforms.ui_rect.w - _88);
    }
    else
    {
        _102 = false;
    }
    float _111;
    if (_102)
    {
        _111 = _87 * step(length(in.in_var_TEXCOORD0 - float2(_88, UiUniforms.ui_rect.w - _88)), _88);
    }
    else
    {
        _111 = _87;
    }
    float _112 = _42.w;
    bool _120;
    if (_112 > 0.0)
    {
        _120 = in.in_var_TEXCOORD0.x > (UiUniforms.ui_rect.z - _112);
    }
    else
    {
        _120 = false;
    }
    bool _127;
    if (_120)
    {
        _127 = in.in_var_TEXCOORD0.y > (UiUniforms.ui_rect.w - _112);
    }
    else
    {
        _127 = false;
    }
    float _137;
    if (_127)
    {
        _137 = _111 * step(length(in.in_var_TEXCOORD0 - float2(UiUniforms.ui_rect.z - _112, UiUniforms.ui_rect.w - _112)), _112);
    }
    else
    {
        _137 = _111;
    }
    if (_137 < 0.5)
    {
        discard_fragment();
    }
    bool _149;
    if ((isunordered(UiUniforms.ui_border_widths.x, 0.0) || UiUniforms.ui_border_widths.x <= 0.0))
    {
        _149 = UiUniforms.ui_border_widths.y > 0.0;
    }
    else
    {
        _149 = true;
    }
    bool _155;
    if (!_149)
    {
        _155 = UiUniforms.ui_border_widths.z > 0.0;
    }
    else
    {
        _155 = true;
    }
    bool _161;
    if (!_155)
    {
        _161 = UiUniforms.ui_border_widths.w > 0.0;
    }
    else
    {
        _161 = true;
    }
    if (_161)
    {
        float2 _171 = precise::max(UiUniforms.ui_rect.zw - float2(UiUniforms.ui_border_widths.x + UiUniforms.ui_border_widths.y, UiUniforms.ui_border_widths.z + UiUniforms.ui_border_widths.w), float2(0.0));
        float2 _173 = in.in_var_TEXCOORD0 - float2(UiUniforms.ui_border_widths.xz);
        float4 _180 = precise::max(_42 - float4(precise::min(UiUniforms.ui_border_widths.x, UiUniforms.ui_border_widths.z), precise::min(UiUniforms.ui_border_widths.y, UiUniforms.ui_border_widths.z), precise::min(UiUniforms.ui_border_widths.x, UiUniforms.ui_border_widths.w), precise::min(UiUniforms.ui_border_widths.y, UiUniforms.ui_border_widths.w)), float4(0.0));
        float _181 = _173.x;
        bool _187;
        if (_181 >= 0.0)
        {
            _187 = _173.y >= 0.0;
        }
        else
        {
            _187 = false;
        }
        bool _192;
        if (_187)
        {
            _192 = _181 <= _171.x;
        }
        else
        {
            _192 = false;
        }
        bool _198;
        if (_192)
        {
            _198 = _173.y <= _171.y;
        }
        else
        {
            _198 = false;
        }
        bool _293;
        if (_198)
        {
            float _201 = _180.x;
            bool _206;
            if (_201 > 0.0)
            {
                _206 = _181 < _201;
            }
            else
            {
                _206 = false;
            }
            bool _211;
            if (_206)
            {
                _211 = _173.y < _201;
            }
            else
            {
                _211 = false;
            }
            float _218;
            if (_211)
            {
                _218 = step(length(_173 - float2(_201)), _201);
            }
            else
            {
                _218 = 1.0;
            }
            float _219 = _180.y;
            bool _226;
            if (_219 > 0.0)
            {
                _226 = _181 > (_171.x - _219);
            }
            else
            {
                _226 = false;
            }
            bool _231;
            if (_226)
            {
                _231 = _173.y < _219;
            }
            else
            {
                _231 = false;
            }
            float _241;
            if (_231)
            {
                _241 = _218 * step(length(_173 - float2(_171.x - _219, _219)), _219);
            }
            else
            {
                _241 = _218;
            }
            float _242 = _180.z;
            bool _247;
            if (_242 > 0.0)
            {
                _247 = _181 < _242;
            }
            else
            {
                _247 = false;
            }
            bool _254;
            if (_247)
            {
                _254 = _173.y > (_171.y - _242);
            }
            else
            {
                _254 = false;
            }
            float _264;
            if (_254)
            {
                _264 = _241 * step(length(_173 - float2(_242, _171.y - _242)), _242);
            }
            else
            {
                _264 = _241;
            }
            float _265 = _180.w;
            bool _272;
            if (_265 > 0.0)
            {
                _272 = _181 > (_171.x - _265);
            }
            else
            {
                _272 = false;
            }
            bool _279;
            if (_272)
            {
                _279 = _173.y > (_171.y - _265);
            }
            else
            {
                _279 = false;
            }
            float _291;
            if (_279)
            {
                _291 = _264 * step(length(_173 - float2(_171.x - _265, _171.y - _265)), _265);
            }
            else
            {
                _291 = _264;
            }
            _293 = _291 >= 0.5;
        }
        else
        {
            _293 = false;
        }
        if (_293)
        {
            discard_fragment();
        }
    }
    out.out_var_SV_Target0 = UiUniforms.ui_color;
    return out;
}
