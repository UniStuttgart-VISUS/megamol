#version 450

uniform sampler2D texLowResFBO;

layout (binding=0, rgba8) uniform image2D imgRead;
layout (binding=1, rgba8) uniform image2D imgWrite;
layout (binding=2, rg32f) uniform image2D imgPosRead;
layout (binding=3, rg32f) uniform image2D imgPosWrite;

uniform ivec2 amortLevel;
uniform ivec2 resolution;
uniform ivec2 lowResResolution;
uniform int frameIdx;
uniform mat4 shiftMx;
uniform vec3 camCenter;
uniform float camAspect;
uniform float frustumHeight;
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
ivec2 pixelOffsetsHighLowRes1D(int highResPos, int lowResPos, int currentIdx, int totalSize, int aLvl) {
    // Case 1
    if (highResPos == lowResPos) {
        return ivec2(0, 0);
    }

    // Case 1
    int offset = lowResPos - highResPos;
    if (abs(offset) <= aLvl / 2) {
        return ivec2(offset, 0);
    }

    if (highResPos < lowResPos) {
        // Case 2
        if (currentIdx > 0) {
            offset = lowResPos - (highResPos + aLvl);
            return ivec2(offset, -1);
        } else {
            return ivec2(offset, 0);
        }
    } else {
        // Case 3
        if (currentIdx < totalSize - 1) {
            offset = lowResPos + aLvl - highResPos;
            return ivec2(offset, 1);
        } else {
            return ivec2(offset, 0);
        }
    }
}

vec2 readPosition(ivec2 coords) {
    if (coords.x >= 0 && coords.x < resolution.x && coords.y >= 0 && coords.y < resolution.y) {
        return imageLoad(imgPosRead, coords).xy;
    }
    return vec2(-3.40282e+38, -3.40282e+38);
}

void main() {
    const vec2 frustumSize = vec2(frustumHeight * camAspect, frustumHeight);

    const ivec2 imgCoord = ivec2(int(uvCoords.x * float(resolution.x)), int(uvCoords.y * float(resolution.y)));
    const vec2 posWorldSpace = camCenter.xy + frustumSize * (uvCoords - 0.5f);

    const ivec2 quadCoord = imgCoord % amortLevel; // Position within the current a*a quad on the high res texture.
    const int idx = (amortLevel.x * quadCoord.y + quadCoord.x); // Linear version of quadCoord as as frame id.
    vec4 color = vec4(0.0f);

    if (frameIdx == idx) {
        // Current high res pixel matches exactly the low res pixel of the current pass.
        color = texelFetch(texLowResFBO, imgCoord / amortLevel, 0);

        imageStore(imgPosWrite, imgCoord, vec4(posWorldSpace, 0.0f, 0.0f));
    } else {
        // Find shifted image coords. This is where the current high res position was in the previous frame.
        const vec4 posClipSpace = vec4(2.0f * uvCoords - 1.0f, 0.0f, 1.0f);
        const vec4 lastPosClipSpace = shiftMx * posClipSpace;
        const ivec2 lastImgCoord = ivec2((lastPosClipSpace.xy / 2.0f + 0.5f) * vec2(resolution));

        const vec2 lastPosWorldSpace = readPosition(lastImgCoord);

        // Position of the current low res pixel within the a*a quad.
        const ivec2 idxCoord = ivec2(frameIdx % amortLevel.x, frameIdx / amortLevel.x);

        ivec2 lowResImgCoord = imgCoord / amortLevel;

        const ivec2 offsetsX = pixelOffsetsHighLowRes1D(quadCoord.x, idxCoord.x, lowResImgCoord.x, lowResResolution.x, amortLevel.x);
        const ivec2 offsetsY = pixelOffsetsHighLowRes1D(quadCoord.y, idxCoord.y, lowResImgCoord.y, lowResResolution.y, amortLevel.y);
        const ivec2 offsetHighRes = ivec2(offsetsX.x, offsetsY.x); // Component wise offset to nearest sample.
        const ivec2 offsetLowRes = ivec2(offsetsX.y, offsetsY.y); // Tex coord offset for lookup in low res texture.

        lowResImgCoord += offsetLowRes;
        const ivec2 sampleImgCoord = imgCoord + offsetHighRes;

        const vec2 sampleUvCoords = (vec2(sampleImgCoord) + 0.5f) / vec2(resolution);
        const vec2 samplePosWorldSpace = camCenter.xy + frustumSize * (sampleUvCoords - 0.5f);

        const float distOldSample = length(posWorldSpace - lastPosWorldSpace);
        const float distNewSample = length(posWorldSpace - samplePosWorldSpace);

        if (distNewSample <= distOldSample && !skipInterpolation) {
            color = texelFetch(texLowResFBO, lowResImgCoord, 0);
            imageStore(imgPosWrite, imgCoord, vec4(samplePosWorldSpace, 0.0f, 0.0f));
        } else {
            color = imageLoad(imgRead, lastImgCoord); // If this sample is nearer the coords should be within the bounds.
            imageStore(imgPosWrite, imgCoord, vec4(lastPosWorldSpace, 0.0f, 0.0f));
        }
    }

    imageStore(imgWrite, imgCoord, color);
    fragOut = color;
}
