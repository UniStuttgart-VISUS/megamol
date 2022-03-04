#version 430

uniform sampler2D texLowResFBO;

layout (binding=0, rgba8) uniform image2D imgRead;
layout (binding=1, rgba8) uniform image2D imgWrite;
layout (binding=2, r32f) uniform image2D imgDistRead;
layout (binding=3, r32f) uniform image2D imgDistWrite;

uniform int amortLevel;
uniform int w;
uniform int h;
uniform int frameIdx;
uniform mat4 shiftMx;
uniform bool skipInterpolation;

in vec2 uvCoords;

out vec4 fragOut;

void main() {
    const ivec2 imgCoord = ivec2(int(uvCoords.x * float(w)), int(uvCoords.y * float(h)));
    const ivec2 quadCoord = imgCoord % amortLevel; // Position within the current a*a quad on the high res texture.

    int idx = (amortLevel * quadCoord.y + quadCoord.x);
    vec4 color = vec4(0.0f);

    if (frameIdx == idx) {
        // Current high res pixel matches exactly the low res pixel of the current pass.
        color = vec4(texelFetch(texLowResFBO, imgCoord / amortLevel, 0).xyz, 1.0f);
        imageStore(imgDistWrite, imgCoord, vec4(1.0f, 0.0f, 0.0f, 0.0f));
    } else {
        // Find shifted image coords. This is where the current high res position was in the previous frame.
        const vec4 p = vec4(2.0f * uvCoords - vec2(1.0f), 0.0f, 1.0f);
        const vec2 shiftedP = (shiftMx * p).xy;
        const ivec2 shiftedImgCoord = ivec2(int((shiftedP.x / 2.0f + 0.5f) * float(w)), int((shiftedP.y / 2.0f + 0.5f) * float(h)));

        // Position of the current low res pixel within the a*a quad.
        const ivec2 idxCoord = ivec2(frameIdx % amortLevel, frameIdx / amortLevel);

        // Calculate distance between current position in the a*a quad and the frameIdx position.
        float dist = length(vec2(quadCoord) - vec2(idxCoord));
        // Normalize distance to [0, 1]. Maximum distance is diagonal of the a*a quad, but measured from pixel centers.
        dist = dist / (sqrt(2.0f) * float(amortLevel - 1));
        // Invert so 1.0 means nearest and 0.0 meanst farest, because texture ist zero initialized.
        dist = clamp(1.0f - dist, 0.0f, 1.0f);

        float oldDist = imageLoad(imgDistRead, shiftedImgCoord).r;

        if (dist >= oldDist && !skipInterpolation) {
            color = vec4(texelFetch(texLowResFBO, imgCoord / amortLevel, 0).xyz, 1.0);
            imageStore(imgDistWrite, imgCoord, vec4(dist, 0.0f, 0.0f, 0.0f));
        } else {
            color = imageLoad(imgRead, shiftedImgCoord);
            imageStore(imgDistWrite, imgCoord, vec4(oldDist, 0.0f, 0.0f, 0.0f));
        }
    }

    imageStore(imgWrite, imgCoord, color);
    fragOut = color;
}
