cbuffer SlugVertexUniforms : register(b0, space1)
{
    column_major float4x4 mvp;
    float2 viewport;
};

cbuffer SlugFragmentUniforms : register(b0, space3)
{
    float weightBoost;
};

Texture2D<float4> curveTexture : register(t0, space2);
Texture2D<uint2> bandTexture : register(t1, space2);

static const int kLogBandTextureWidth = 12;

struct TextVertexInput
{
    float4 pos : TEXCOORD0;
    float4 tex : TEXCOORD1;
    float4 jac : TEXCOORD2;
    float4 bnd : TEXCOORD3;
    float4 col : TEXCOORD4;
};

struct TextVertexOutput
{
    float4 position : SV_Position;
    float4 color : TEXCOORD0;
    float2 texcoord : TEXCOORD1;
    nointerpolation float4 banding : TEXCOORD2;
    nointerpolation int4 glyph : TEXCOORD3;
};

void SlugUnpack(float4 tex, float4 bnd, out float4 vbnd, out int4 vgly)
{
    uint2 g = asuint(tex.zw);
    vgly = int4(g.x & 0xFFFFu, g.x >> 16u, g.y & 0xFFFFu, g.y >> 16u);
    vbnd = bnd;
}

float2 SlugDilate(float4 pos, float4 tex, float4 jac, float4 m0, float4 m1, float4 m3, float2 dim, out float2 vpos)
{
    float2 n = normalize(pos.zw);
    float s = dot(m3.xy, pos.xy) + m3.w;
    float t = dot(m3.xy, n);

    float u = (s * dot(m0.xy, n) - t * (dot(m0.xy, pos.xy) + m0.w)) * dim.x;
    float v = (s * dot(m1.xy, n) - t * (dot(m1.xy, pos.xy) + m1.w)) * dim.y;

    float s2 = s * s;
    float st = s * t;
    float uv = u * u + v * v;
    float2 d = pos.zw * (s2 * (st + sqrt(uv)) / (uv - st * st));

    vpos = pos.xy + d;
    return float2(tex.x + dot(d, jac.xy), tex.y + dot(d, jac.zw));
}

TextVertexOutput TextVertexMain(TextVertexInput input)
{
    float2 p;

    float4 m0 = float4(mvp[0][0], mvp[1][0], mvp[2][0], mvp[3][0]);
    float4 m1 = float4(mvp[0][1], mvp[1][1], mvp[2][1], mvp[3][1]);
    float4 m2 = float4(mvp[0][2], mvp[1][2], mvp[2][2], mvp[3][2]);
    float4 m3 = float4(mvp[0][3], mvp[1][3], mvp[2][3], mvp[3][3]);

    TextVertexOutput output;
    output.texcoord = SlugDilate(input.pos, input.tex, input.jac, m0, m1, m3, viewport, p);
    output.position = float4(
        p.x * m0.x + p.y * m0.y + m0.w,
        p.x * m1.x + p.y * m1.y + m1.w,
        p.x * m2.x + p.y * m2.y + m2.w,
        p.x * m3.x + p.y * m3.y + m3.w);
    SlugUnpack(input.tex, input.bnd, output.banding, output.glyph);
    output.color = input.col;
    return output;
}

uint CalcRootCode(float y1, float y2, float y3)
{
    uint i1 = asuint(y1) >> 31u;
    uint i2 = asuint(y2) >> 30u;
    uint i3 = asuint(y3) >> 29u;

    uint shift = (i2 & 2u) | (i1 & ~2u);
    shift = (i3 & 4u) | (shift & ~4u);

    return ((0x2E74u >> shift) & 0x0101u);
}

float2 SolveHorizPoly(float4 p12, float2 p3)
{
    float2 a = p12.xy - p12.zw * 2.0 + p3;
    float2 b = p12.xy - p12.zw;
    float ra = 1.0 / a.y;
    float rb = 0.5 / b.y;

    float d = sqrt(max(b.y * b.y - a.y * p12.y, 0.0));
    float t1 = (b.y - d) * ra;
    float t2 = (b.y + d) * ra;

    if (abs(a.y) < 1.0 / 65536.0) {
        t1 = p12.y * rb;
        t2 = t1;
    }

    return float2((a.x * t1 - b.x * 2.0) * t1 + p12.x,
                  (a.x * t2 - b.x * 2.0) * t2 + p12.x);
}

float2 SolveVertPoly(float4 p12, float2 p3)
{
    float2 a = p12.xy - p12.zw * 2.0 + p3;
    float2 b = p12.xy - p12.zw;
    float ra = 1.0 / a.x;
    float rb = 0.5 / b.x;

    float d = sqrt(max(b.x * b.x - a.x * p12.x, 0.0));
    float t1 = (b.x - d) * ra;
    float t2 = (b.x + d) * ra;

    if (abs(a.x) < 1.0 / 65536.0) {
        t1 = p12.x * rb;
        t2 = t1;
    }

    return float2((a.y * t1 - b.y * 2.0) * t1 + p12.y,
                  (a.y * t2 - b.y * 2.0) * t2 + p12.y);
}

int2 CalcBandLoc(int2 glyphLoc, uint offset)
{
    int2 bandLoc = int2(glyphLoc.x + int(offset), glyphLoc.y);
    bandLoc.y += bandLoc.x >> kLogBandTextureWidth;
    bandLoc.x &= (1u << kLogBandTextureWidth) - 1u;
    return bandLoc;
}

float CalcCoverage(float xcov, float ycov, float xwgt, float ywgt)
{
    float coverage = max(abs(xcov * xwgt + ycov * ywgt) / max(xwgt + ywgt, 1.0 / 65536.0),
                         min(abs(xcov), abs(ycov)));
    return saturate(coverage);
}

float4 TextFragmentMain(TextVertexOutput input) : SV_Target0
{
    float2 renderCoord = input.texcoord;
    float4 bandTransform = input.banding;
    int4 glyphData = input.glyph;

    float2 emsPerPixel = fwidth(renderCoord);
    float2 pixelsPerEm = 1.0 / emsPerPixel;

    int2 bandMax = glyphData.zw;
    bandMax.y &= 0x00FF;

    int2 bandIndex = clamp(int2(renderCoord * bandTransform.xy + bandTransform.zw), int2(0, 0), bandMax);
    int2 glyphLoc = glyphData.xy;

    float xcov = 0.0;
    float xwgt = 0.0;

    uint2 hbandData = bandTexture.Load(int3(glyphLoc.x + bandIndex.y, glyphLoc.y, 0));
    int2 hbandLoc = CalcBandLoc(glyphLoc, hbandData.y);

    for (int curveIndex = 0; curveIndex < int(hbandData.x); curveIndex++) {
        int2 curveLoc = int2(bandTexture.Load(int3(hbandLoc.x + curveIndex, hbandLoc.y, 0)));
        float4 p12 = curveTexture.Load(int3(curveLoc, 0)) - float4(renderCoord, renderCoord);
        float2 p3 = curveTexture.Load(int3(curveLoc.x + 1, curveLoc.y, 0)).xy - renderCoord;

        if (max(max(p12.x, p12.z), p3.x) * pixelsPerEm.x < -0.5) break;

        uint code = CalcRootCode(p12.y, p12.w, p3.y);
        if (code != 0u) {
            float2 r = SolveHorizPoly(p12, p3) * pixelsPerEm.x;

            if ((code & 1u) != 0u) {
                xcov += clamp(r.x + 0.5, 0.0, 1.0);
                xwgt = max(xwgt, clamp(1.0 - abs(r.x) * 2.0, 0.0, 1.0));
            }
            if (code > 1u) {
                xcov -= clamp(r.y + 0.5, 0.0, 1.0);
                xwgt = max(xwgt, clamp(1.0 - abs(r.y) * 2.0, 0.0, 1.0));
            }
        }
    }

    float ycov = 0.0;
    float ywgt = 0.0;

    uint2 vbandData = bandTexture.Load(int3(glyphLoc.x + bandMax.y + 1 + bandIndex.x, glyphLoc.y, 0));
    int2 vbandLoc = CalcBandLoc(glyphLoc, vbandData.y);

    for (int curveIndex = 0; curveIndex < int(vbandData.x); curveIndex++) {
        int2 curveLoc = int2(bandTexture.Load(int3(vbandLoc.x + curveIndex, vbandLoc.y, 0)));
        float4 p12 = curveTexture.Load(int3(curveLoc, 0)) - float4(renderCoord, renderCoord);
        float2 p3 = curveTexture.Load(int3(curveLoc.x + 1, curveLoc.y, 0)).xy - renderCoord;

        if (max(max(p12.y, p12.w), p3.y) * pixelsPerEm.y < -0.5) break;

        uint code = CalcRootCode(p12.x, p12.z, p3.x);
        if (code != 0u) {
            float2 r = SolveVertPoly(p12, p3) * pixelsPerEm.y;

            if ((code & 1u) != 0u) {
                ycov -= clamp(r.x + 0.5, 0.0, 1.0);
                ywgt = max(ywgt, clamp(1.0 - abs(r.x) * 2.0, 0.0, 1.0));
            }
            if (code > 1u) {
                ycov += clamp(r.y + 0.5, 0.0, 1.0);
                ywgt = max(ywgt, clamp(1.0 - abs(r.y) * 2.0, 0.0, 1.0));
            }
        }
    }

    float coverage = CalcCoverage(xcov, ycov, xwgt, ywgt);
    if (weightBoost > 0.5) coverage = sqrt(coverage);
    return float4(input.color.rgb, input.color.a * coverage);
}
