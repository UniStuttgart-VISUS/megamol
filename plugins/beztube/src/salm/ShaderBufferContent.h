#pragma once

#include "vislib/graphics/gl/IncludeAllGL.h"
#include <glm/glm.hpp>

namespace megamol {
namespace beztube {
namespace salm {

    class ShaderBufferContent {
        float *content;
        unsigned int float4BlockCount;
        size_t size;

    public:
        ShaderBufferContent();
        ~ShaderBufferContent();

        inline float *Content() { return content; }
        inline const float *Content() const { return content; }
        inline unsigned int Float4BlockCount() const { return float4BlockCount; }
        inline size_t Size() const { return size; }

        void Allocate(unsigned int float4BlockCount);
        
        inline void Fill(int offset, int subOffset, float newContent) { 
            content[offset * 4 + subOffset] = newContent; 
        }

        inline void Fill(int offset, int subOffset, const glm::vec2& newContent) {
            offset = offset * 4 + subOffset;
            content[offset] = newContent.x;
            content[++offset] = newContent.y;
        }
        inline void Fill(int offset, int subOffset, const glm::vec3& newContent) {
            offset = offset * 4 + subOffset;
            content[offset] = newContent.x;
            content[++offset] = newContent.y;
            content[++offset] = newContent.z;
        }
        inline void Fill(int offset, const glm::vec4& newContent) {
            offset *= 4;
            content[offset] = newContent.x;
            content[++offset] = newContent.y;
            content[++offset] = newContent.z;
            content[++offset] = newContent.w;
        }
        inline void Fill(int offset, const glm::mat4& newContent) {
            offset *= 4;
            content[offset]   = newContent[0][0]; // unsure about the order...
            content[++offset] = newContent[0][1];
            content[++offset] = newContent[0][2];
            content[++offset] = newContent[0][3];
            content[++offset] = newContent[1][0];
            content[++offset] = newContent[1][1];
            content[++offset] = newContent[1][2];
            content[++offset] = newContent[1][3];
            content[++offset] = newContent[2][0];
            content[++offset] = newContent[2][1];
            content[++offset] = newContent[2][2];
            content[++offset] = newContent[2][3];
            content[++offset] = newContent[3][0];
            content[++offset] = newContent[3][1];
            content[++offset] = newContent[3][2];
            content[++offset] = newContent[3][3];
        }
        inline void Fill(int offset, int subOffset, int newContent) {
            content[offset * 4 + subOffset] = *((float*)(&newContent));
        }

    };

}
}
}
