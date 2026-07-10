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
    float2 _68 = fwidth(in.in_var_TEXCOORD1);
    float2 _69 = float2(1.0) / _68;
    int2 _70 = in.in_var_TEXCOORD3.zw;
    int _72 = in.in_var_TEXCOORD3.w & 255;
    _70.y = _72;
    int2 _79 = clamp(int2((in.in_var_TEXCOORD1 * in.in_var_TEXCOORD2.xy) + in.in_var_TEXCOORD2.zw), int2(0), _70);
    uint4 _87 = bandTexture.read(uint2(int3(in.in_var_TEXCOORD3.x + _79.y, in.in_var_TEXCOORD3.y, 0).xy), 0);
    int _90 = in.in_var_TEXCOORD3.x + int(_87.y);
    int _92 = in.in_var_TEXCOORD3.y + (_90 >> 12);
    int _95 = int(uint(_90) & 4095u);
    float _97;
    float _100;
    _97 = 0.0;
    _100 = 0.0;
    float _98;
    float _101;
    for (int _102 = 0; _102 < int(_87.x); _97 = _98, _100 = _101, _102++)
    {
        int2 _115 = int2(bandTexture.read(uint2(int3(_95 + _102, _92, 0).xy), 0).xy);
        int _116 = _115.x;
        float4 _125 = curveTexture.read(uint2(int3(_116, _115.y, 0).xy), 0) - float4(in.in_var_TEXCOORD1.xyxy);
        float2 _132 = curveTexture.read(uint2(int3(_116 + 1, _115.y, 0).xy), 0).xy - in.in_var_TEXCOORD1;
        float _133 = _125.x;
        float _138 = _69.x;
        if ((precise::max(precise::max(_133, _125.z), _132.x) * _138) < (-0.5))
        {
            break;
        }
        float _143 = _125.y;
        uint _159 = 11892u >> ((((as_type<uint>(_132.y) >> 29u) & 4u) | ((((as_type<uint>(_125.w) >> 30u) & 2u) | ((as_type<uint>(_143) >> 31u) & 4294967293u)) & 4294967291u)) & 31u);
        uint _160 = _159 & 257u;
        if (_160 != 0u)
        {
            float2 _164 = _125.xy;
            float2 _165 = _125.zw;
            float2 _168 = (_164 - (_165 * 2.0)) + _132;
            float2 _169 = _164 - _165;
            float _170 = _168.y;
            float _171 = 1.0 / _170;
            float _172 = _169.y;
            float _178 = sqrt(precise::max((_172 * _172) - (_170 * _143), 0.0));
            float _188;
            float _189;
            if (abs(_170) < 1.52587890625e-05)
            {
                float _187 = _143 * (0.5 / _172);
                _188 = _187;
                _189 = _187;
            }
            else
            {
                _188 = (_172 + _178) * _171;
                _189 = (_172 - _178) * _171;
            }
            float _190 = _168.x;
            float _193 = _169.x * 2.0;
            float2 _202 = float2((((_190 * _189) - _193) * _189) + _133, (((_190 * _188) - _193) * _188) + _133) * _138;
            float _216;
            float _217;
            if ((_159 & 1u) != 0u)
            {
                float _207 = _202.x;
                _216 = precise::max(_97, fast::clamp(1.0 - (abs(_207) * 2.0), 0.0, 1.0));
                _217 = _100 + fast::clamp(_207 + 0.5, 0.0, 1.0);
            }
            else
            {
                _216 = _97;
                _217 = _100;
            }
            float _230;
            float _231;
            if (_160 > 1u)
            {
                float _221 = _202.y;
                _230 = precise::max(_216, fast::clamp(1.0 - (abs(_221) * 2.0), 0.0, 1.0));
                _231 = _217 - fast::clamp(_221 + 0.5, 0.0, 1.0);
            }
            else
            {
                _230 = _216;
                _231 = _217;
            }
            _98 = _230;
            _101 = _231;
        }
        else
        {
            _98 = _97;
            _101 = _100;
        }
    }
    uint4 _239 = bandTexture.read(uint2(int3(((in.in_var_TEXCOORD3.x + _72) + 1) + _79.x, in.in_var_TEXCOORD3.y, 0).xy), 0);
    int _242 = in.in_var_TEXCOORD3.x + int(_239.y);
    int _244 = in.in_var_TEXCOORD3.y + (_242 >> 12);
    int _247 = int(uint(_242) & 4095u);
    float _249;
    float _252;
    _249 = 0.0;
    _252 = 0.0;
    float _250;
    float _253;
    for (int _254 = 0; _254 < int(_239.x); _249 = _250, _252 = _253, _254++)
    {
        int2 _267 = int2(bandTexture.read(uint2(int3(_247 + _254, _244, 0).xy), 0).xy);
        int _268 = _267.x;
        float4 _277 = curveTexture.read(uint2(int3(_268, _267.y, 0).xy), 0) - float4(in.in_var_TEXCOORD1.xyxy);
        float2 _284 = curveTexture.read(uint2(int3(_268 + 1, _267.y, 0).xy), 0).xy - in.in_var_TEXCOORD1;
        float _285 = _277.y;
        float _290 = _69.y;
        if ((precise::max(precise::max(_285, _277.w), _284.y) * _290) < (-0.5))
        {
            break;
        }
        float _295 = _277.x;
        uint _311 = 11892u >> ((((as_type<uint>(_284.x) >> 29u) & 4u) | ((((as_type<uint>(_277.z) >> 30u) & 2u) | ((as_type<uint>(_295) >> 31u) & 4294967293u)) & 4294967291u)) & 31u);
        uint _312 = _311 & 257u;
        if (_312 != 0u)
        {
            float2 _316 = _277.xy;
            float2 _317 = _277.zw;
            float2 _320 = (_316 - (_317 * 2.0)) + _284;
            float2 _321 = _316 - _317;
            float _322 = _320.x;
            float _323 = 1.0 / _322;
            float _324 = _321.x;
            float _330 = sqrt(precise::max((_324 * _324) - (_322 * _295), 0.0));
            float _340;
            float _341;
            if (abs(_322) < 1.52587890625e-05)
            {
                float _339 = _295 * (0.5 / _324);
                _340 = _339;
                _341 = _339;
            }
            else
            {
                _340 = (_324 + _330) * _323;
                _341 = (_324 - _330) * _323;
            }
            float _342 = _320.y;
            float _345 = _321.y * 2.0;
            float2 _354 = float2((((_342 * _341) - _345) * _341) + _285, (((_342 * _340) - _345) * _340) + _285) * _290;
            float _368;
            float _369;
            if ((_311 & 1u) != 0u)
            {
                float _359 = _354.x;
                _368 = precise::max(_249, fast::clamp(1.0 - (abs(_359) * 2.0), 0.0, 1.0));
                _369 = _252 - fast::clamp(_359 + 0.5, 0.0, 1.0);
            }
            else
            {
                _368 = _249;
                _369 = _252;
            }
            float _382;
            float _383;
            if (_312 > 1u)
            {
                float _373 = _354.y;
                _382 = precise::max(_368, fast::clamp(1.0 - (abs(_373) * 2.0), 0.0, 1.0));
                _383 = _369 + fast::clamp(_373 + 0.5, 0.0, 1.0);
            }
            else
            {
                _382 = _368;
                _383 = _369;
            }
            _250 = _382;
            _253 = _383;
        }
        else
        {
            _250 = _249;
            _253 = _252;
        }
    }
    float _395 = fast::clamp(precise::max(abs((_100 * _97) + (_252 * _249)) / precise::max(_97 + _249, 1.52587890625e-05), precise::min(abs(_100), abs(_252))), 0.0, 1.0);
    float _402;
    if (SlugFragmentUniforms.weightBoost > 0.5)
    {
        _402 = sqrt(_395);
    }
    else
    {
        _402 = _395;
    }
    out.out_var_SV_Target0 = in.in_var_TEXCOORD0 * _402;
    return out;
}
