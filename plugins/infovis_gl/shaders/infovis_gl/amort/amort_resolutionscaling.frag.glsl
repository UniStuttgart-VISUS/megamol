#version 430

uniform sampler2D texLowResFBO;

layout (binding=0, rgba8) uniform image2D imgRead;
layout (binding=1, rgba8) uniform image2D imgWrite;
layout (binding=2, r32f) uniform image2D imgDistRead;
layout (binding=3, r32f) uniform image2D imgDistWrite;

uniform int amortLevel;
uniform ivec2 resolution;
uniform ivec2 lowResResolution;
uniform int frameIdx;
uniform mat4 shiftMx;
uniform bool skipInterpolation;

in vec2 uvCoords;

out vec4 fragOut;

/**
 *         | Left block    | Current block | Right block   |
 *         | 0 | 1 | 2 | 3 | 0 | 1 | 2 | 3 | 0 | 1 | 2 | 3 |
 * Case 1: | L |   |   |   | L | H |   |   | L |   |   |   |
 *         |   |   |   | L |   |   | H | L |   |   |   | L |
 * Case 2: |   |   |   | L | H |   |   | L |   |   |   | L |
 * Case 3: | L |   |   |   | L |   |   | H | L |   |   |   |
 *
 * Case 1: The lowest distance between low res and high res pixel is within the same block.
 *         This is always the case if the distance < amortLevel / 2.
 * Case 2: The low res pixel of the left block is nearest to the current high res pixel.
 *         This is the case if distance > amortLevel / 2 and high < low.
 * Case 3: The low res pixel of the right block is nearest to the current high res pixel.
 *         This is the case if distance > amortLevel / 2 and low > high.
 */
ivec2 pixelDistanceWithOffset1D(int highResPos, int lowResPos, int currentIdx, int totalSize) {
    // Case 1
    if (highResPos == lowResPos) {
        return ivec2(0, 0);
    }

    // Case 1
    int dist = abs(highResPos - lowResPos);
    if (dist <= amortLevel / 2) {
        return ivec2(dist, 0);
    }

    if (highResPos < lowResPos) {
        // Case 2
        if (currentIdx > 0) {
            dist = highResPos + amortLevel - lowResPos;
            return ivec2(dist, -1);
        } else {
            return ivec2(dist, 0);
        }
    } else {
        // Case 3
        if (currentIdx < totalSize - 1) {
            dist = lowResPos + amortLevel - highResPos;
            return ivec2(dist, 1);
        } else {
            return ivec2(dist, 0);
        }
    }
}

void main() {
    const ivec2 imgCoord = ivec2(int(uvCoords.x * float(resolution.x)), int(uvCoords.y * float(resolution.y)));
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
        const ivec2 shiftedImgCoord = ivec2(int((shiftedP.x / 2.0f + 0.5f) * float(resolution.x)), int((shiftedP.y / 2.0f + 0.5f) * float(resolution.y)));

        // Position of the current low res pixel within the a*a quad.
        const ivec2 idxCoord = ivec2(frameIdx % amortLevel, frameIdx / amortLevel);

        ivec2 lowResTexCoord = imgCoord / amortLevel;

        const ivec2 distOffsetX = pixelDistanceWithOffset1D(quadCoord.x, idxCoord.x, lowResTexCoord.x, lowResResolution.x);
        const ivec2 distOffsetY = pixelDistanceWithOffset1D(quadCoord.y, idxCoord.y, lowResTexCoord.y, lowResResolution.y);
        const ivec2 distXY = ivec2(distOffsetX.x, distOffsetY.x); // Component wise distance to nearest sample.
        const ivec2 offset = ivec2(distOffsetX.y, distOffsetY.y); // Tex coord offset for lookup in low res texture.

        lowResTexCoord += offset;

        // Calculate distance between current position in the a*a quad and the nearest frameIdx position.
        float dist = length(vec2(distXY));
        // Normalize distance to [0, 1]. Maximum distance is diagonal of the a*a quad, but measured from pixel centers.
        dist = dist / (sqrt(2.0f) * float(amortLevel - 1));
        // Invert so 1.0 means nearest and 0.0 meanst farest, because texture ist zero initialized.
        dist = clamp(1.0f - dist, 0.0f, 1.0f);

        float oldDist = imageLoad(imgDistRead, shiftedImgCoord).r;

        if (dist >= oldDist && !skipInterpolation) {
            color = vec4(texelFetch(texLowResFBO, lowResTexCoord, 0).xyz, 1.0);
            imageStore(imgDistWrite, imgCoord, vec4(dist, 0.0f, 0.0f, 0.0f));
        } else {
            color = imageLoad(imgRead, shiftedImgCoord);
            imageStore(imgDistWrite, imgCoord, vec4(oldDist, 0.0f, 0.0f, 0.0f));
        }
    }

    imageStore(imgWrite, imgCoord, color);
    fragOut = color;
}
