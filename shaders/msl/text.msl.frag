#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct type_SlugVertexUniforms
{
    float4x4 mvp;
    float2 viewport;
};

struct type_SlugFragmentUniforms
{
    float weightBoost;
};

struct TextFragmentMain_out
{
    float4 out_var_SV_Target0 [[color(0)]];
};

struct TextFragmentMain_in
{
    float4 in_var_TEXCOORD0 [[user(locn0)]];
    float2 in_var_TEXCOORD1 [[user(locn1)]];
    float4 in_var_TEXCOORD2 [[user(locn2), flat]];
    int4 in_var_TEXCOORD3 [[user(locn3)]];
};

fragment TextFragmentMain_out TextFragmentMain(TextFragmentMain_in in [[stage_in]], constant type_SlugFragmentUniforms& SlugFragmentUniforms [[buffer(0)]], texture2d<float> curveTexture [[texture(0)]], texture2d<uint> bandTexture [[texture(1)]])
{
    TextFragmentMain_out out = {};
    float2 _70 = fwidth(in.in_var_TEXCOORD1);
    float2 _71 = float2(1.0) / _70;
    int2 _72 = in.in_var_TEXCOORD3.zw;
    int _74 = in.in_var_TEXCOORD3.w & 255;
    _72.y = _74;
    int2 _81 = clamp(int2((in.in_var_TEXCOORD1 * in.in_var_TEXCOORD2.xy) + in.in_var_TEXCOORD2.zw), int2(0), _72);
    uint4 _89 = bandTexture.read(uint2(int3(in.in_var_TEXCOORD3.x + _81.y, in.in_var_TEXCOORD3.y, 0).xy), 0);
    int _92 = in.in_var_TEXCOORD3.x + int(_89.y);
    int _94 = in.in_var_TEXCOORD3.y + (_92 >> 12);
    int _97 = int(uint(_92) & 4095u);
    float _99;
    float _102;
    _99 = 0.0;
    _102 = 0.0;
    float _100;
    float _103;
    for (int _104 = 0; _104 < int(_89.x); _99 = _100, _102 = _103, _104++)
    {
        int2 _117 = int2(bandTexture.read(uint2(int3(_97 + _104, _94, 0).xy), 0).xy);
        int _118 = _117.x;
        float4 _127 = curveTexture.read(uint2(int3(_118, _117.y, 0).xy), 0) - float4(in.in_var_TEXCOORD1.xyxy);
        float2 _134 = curveTexture.read(uint2(int3(_118 + 1, _117.y, 0).xy), 0).xy - in.in_var_TEXCOORD1;
        float _135 = _127.x;
        float _140 = _71.x;
        if ((precise::max(precise::max(_135, _127.z), _134.x) * _140) < (-0.5))
        {
            break;
        }
        float _145 = _127.y;
        uint _161 = 11892u >> ((((as_type<uint>(_134.y) >> 29u) & 4u) | ((((as_type<uint>(_127.w) >> 30u) & 2u) | ((as_type<uint>(_145) >> 31u) & 4294967293u)) & 4294967291u)) & 31u);
        uint _162 = _161 & 257u;
        if (_162 != 0u)
        {
            float2 _166 = _127.xy;
            float2 _167 = _127.zw;
            float2 _170 = (_166 - (_167 * 2.0)) + _134;
            float2 _171 = _166 - _167;
            float _172 = _170.y;
            float _173 = 1.0 / _172;
            float _174 = _171.y;
            float _180 = sqrt(precise::max((_174 * _174) - (_172 * _145), 0.0));
            float _190;
            float _191;
            if (abs(_172) < 1.52587890625e-05)
            {
                float _189 = _145 * (0.5 / _174);
                _190 = _189;
                _191 = _189;
            }
            else
            {
                _190 = (_174 + _180) * _173;
                _191 = (_174 - _180) * _173;
            }
            float _192 = _170.x;
            float _195 = _171.x * 2.0;
            float2 _204 = float2((((_192 * _191) - _195) * _191) + _135, (((_192 * _190) - _195) * _190) + _135) * _140;
            float _218;
            float _219;
            if ((_161 & 1u) != 0u)
            {
                float _209 = _204.x;
                _218 = precise::max(_99, fast::clamp(1.0 - (abs(_209) * 2.0), 0.0, 1.0));
                _219 = _102 + fast::clamp(_209 + 0.5, 0.0, 1.0);
            }
            else
            {
                _218 = _99;
                _219 = _102;
            }
            float _232;
            float _233;
            if (_162 > 1u)
            {
                float _223 = _204.y;
                _232 = precise::max(_218, fast::clamp(1.0 - (abs(_223) * 2.0), 0.0, 1.0));
                _233 = _219 - fast::clamp(_223 + 0.5, 0.0, 1.0);
            }
            else
            {
                _232 = _218;
                _233 = _219;
            }
            _100 = _232;
            _103 = _233;
        }
        else
        {
            _100 = _99;
            _103 = _102;
        }
    }
    uint4 _241 = bandTexture.read(uint2(int3(((in.in_var_TEXCOORD3.x + _74) + 1) + _81.x, in.in_var_TEXCOORD3.y, 0).xy), 0);
    int _244 = in.in_var_TEXCOORD3.x + int(_241.y);
    int _246 = in.in_var_TEXCOORD3.y + (_244 >> 12);
    int _249 = int(uint(_244) & 4095u);
    float _251;
    float _254;
    _251 = 0.0;
    _254 = 0.0;
    float _252;
    float _255;
    for (int _256 = 0; _256 < int(_241.x); _251 = _252, _254 = _255, _256++)
    {
        int2 _269 = int2(bandTexture.read(uint2(int3(_249 + _256, _246, 0).xy), 0).xy);
        int _270 = _269.x;
        float4 _279 = curveTexture.read(uint2(int3(_270, _269.y, 0).xy), 0) - float4(in.in_var_TEXCOORD1.xyxy);
        float2 _286 = curveTexture.read(uint2(int3(_270 + 1, _269.y, 0).xy), 0).xy - in.in_var_TEXCOORD1;
        float _287 = _279.y;
        float _292 = _71.y;
        if ((precise::max(precise::max(_287, _279.w), _286.y) * _292) < (-0.5))
        {
            break;
        }
        float _297 = _279.x;
        uint _313 = 11892u >> ((((as_type<uint>(_286.x) >> 29u) & 4u) | ((((as_type<uint>(_279.z) >> 30u) & 2u) | ((as_type<uint>(_297) >> 31u) & 4294967293u)) & 4294967291u)) & 31u);
        uint _314 = _313 & 257u;
        if (_314 != 0u)
        {
            float2 _318 = _279.xy;
            float2 _319 = _279.zw;
            float2 _322 = (_318 - (_319 * 2.0)) + _286;
            float2 _323 = _318 - _319;
            float _324 = _322.x;
            float _325 = 1.0 / _324;
            float _326 = _323.x;
            float _332 = sqrt(precise::max((_326 * _326) - (_324 * _297), 0.0));
            float _342;
            float _343;
            if (abs(_324) < 1.52587890625e-05)
            {
                float _341 = _297 * (0.5 / _326);
                _342 = _341;
                _343 = _341;
            }
            else
            {
                _342 = (_326 + _332) * _325;
                _343 = (_326 - _332) * _325;
            }
            float _344 = _322.y;
            float _347 = _323.y * 2.0;
            float2 _356 = float2((((_344 * _343) - _347) * _343) + _287, (((_344 * _342) - _347) * _342) + _287) * _292;
            float _370;
            float _371;
            if ((_313 & 1u) != 0u)
            {
                float _361 = _356.x;
                _370 = precise::max(_251, fast::clamp(1.0 - (abs(_361) * 2.0), 0.0, 1.0));
                _371 = _254 - fast::clamp(_361 + 0.5, 0.0, 1.0);
            }
            else
            {
                _370 = _251;
                _371 = _254;
            }
            float _384;
            float _385;
            if (_314 > 1u)
            {
                float _375 = _356.y;
                _384 = precise::max(_370, fast::clamp(1.0 - (abs(_375) * 2.0), 0.0, 1.0));
                _385 = _371 + fast::clamp(_375 + 0.5, 0.0, 1.0);
            }
            else
            {
                _384 = _370;
                _385 = _371;
            }
            _252 = _384;
            _255 = _385;
        }
        else
        {
            _252 = _251;
            _255 = _254;
        }
    }
    float _397 = fast::clamp(precise::max(abs((_102 * _99) + (_254 * _251)) / precise::max(_99 + _251, 1.52587890625e-05), precise::min(abs(_102), abs(_254))), 0.0, 1.0);
    float _404;
    if (SlugFragmentUniforms.weightBoost > 0.5)
    {
        _404 = sqrt(_397);
    }
    else
    {
        _404 = _397;
    }
    out.out_var_SV_Target0 = float4(in.in_var_TEXCOORD0.xyz, in.in_var_TEXCOORD0.w * _404);
    return out;
}
