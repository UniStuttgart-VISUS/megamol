#version 400

// Same defines as in SDFFont.h!
#define RENDERMODE_FILL    1
#define RENDERMODE_OUTLINE 2

#define SDF_THRESHOLD      0.5

in vec2 texCoord;
in vec4 color;

uniform sampler2D fontTexture;
uniform vec3      outlineColor       = vec3(0.0, 0.0, 0.0);
uniform float     outlineThickness   = 0.2; // in [0,1]
uniform int       renderMode         = RENDERMODE_FILL;
uniform bool      smoothMode         = true; // TODO only implemented for RENDERMODE_FILL!

layout(location = 0) out vec4 outFragColor;

float supersample(const in vec2 uv, const in float w) {
    return smoothstep(SDF_THRESHOLD - w, SDF_THRESHOLD + w, texture(fontTexture, uv).a);
}

void main(void) {

    float outline_fraction = 0.0;
    float distance = texture(fontTexture, texCoord).a;
    if (distance == 0.0)  {
        discard;
    }
    float smootingEdge = fwidth(distance); // = 0.7 * length(vec2(dFdx(distance), dFdy(distance)));  // dFdxFine(), dFdyFine() only for glsl >=450
    if (renderMode == RENDERMODE_FILL && smoothMode) {
        distance = smoothstep((SDF_THRESHOLD - smootingEdge), (SDF_THRESHOLD + smootingEdge), distance);
    }
    else if (renderMode == RENDERMODE_OUTLINE) {
        float sdf_border = (outlineThickness/2.0) * 0.49; // scale from range [0, 1] to range [0, 0.49]
        float distance_alpha = smoothstep((SDF_THRESHOLD - sdf_border - smootingEdge), (SDF_THRESHOLD - sdf_border + smootingEdge), distance);
        outline_fraction = smoothstep((SDF_THRESHOLD - smootingEdge ), (SDF_THRESHOLD + smootingEdge), distance);
        distance = distance_alpha;
    }
    float alpha = distance;
    if (alpha <= 0.0) discard;

    // Supersample, 4 extra points --------------------------------------------
    /// Is supersampling within sdf font texture reasonable anyway?
    float dscale = 0.354; // half of 1/sqrt2; you can play with this
    vec2 duv = dscale * (dFdx(texCoord) + dFdy(texCoord));
    vec4 box = vec4(texCoord-duv, texCoord+duv);
    float asum = supersample( box.xy, smootingEdge )
               + supersample( box.zw, smootingEdge )
               + supersample( box.xw, smootingEdge )
               + supersample( box.zy, smootingEdge );
    // Weighted average, with 4 extra points having 0.5 weight each, so 1 + 0.5*4 = 3 is the divisor
    // -------------------------------------------------------------------------
       
    if (renderMode == RENDERMODE_FILL) {
        if (smoothMode) {
            // Apply 4xSS
            alpha = (alpha + 0.5 * asum) / 3.0;
            alpha = clamp(alpha, 0.0, 1.0);
            if (alpha <= 0.0) discard;

            outFragColor = vec4(color.rgb, color.a * alpha);
        } else {
            if (distance < 0.5f) {
                discard;
            }
            outFragColor = color;
        }
    }
    else if (renderMode == RENDERMODE_OUTLINE) {
        // Apply 4xSS
        alpha = (alpha + (1.0 - outline_fraction) + 0.5 * asum) / 3.0;
        alpha = clamp(alpha, 0.0, 1.0);
        if (alpha <= 0.0) discard;

        outFragColor = vec4(mix(outlineColor, color.rgb, outline_fraction), (color.a * alpha));
    }
}
