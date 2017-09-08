#pragma once

#include "vislib/graphics/gl/IncludeAllGL.h"
#include "salm/ShaderBufferContent.h"

namespace megamol {
namespace beztube {
namespace salm {

    class ShaderBuffer {
    public:
        enum BufferType {
            INVALID,
            UBO, SSBO
        };

        static inline GLenum ToBufferTarget(BufferType value) {
            switch (value) {
            case UBO: return GL_UNIFORM_BUFFER;
            case SSBO: return GL_SHADER_STORAGE_BUFFER;
            }
            return 0;
        }
        static inline GLenum ToBufferRangeTarget(BufferType value) {
            switch (value) {
            case UBO: return GL_UNIFORM_BUFFER;
            case SSBO: return GL_SHADER_STORAGE_BUFFER;
            }
            return 0;
        }
        static inline GLenum ToProgramInterface(BufferType value) {
            switch (value) {
            case UBO: return GL_UNIFORM_BLOCK;
            case SSBO: return GL_SHADER_STORAGE_BLOCK;
            }
            return 0;
        }

    private:

        GLuint handle;
        int bufferIndex = -1;
        BufferType type = INVALID;
        size_t size;
        int float4BlockCount;
        GLenum bufferTargetType;
        GLenum bufferRangeTargetType;
        GLenum programInterfaceType;
        GLenum usage;

    public:

        ShaderBuffer();
        ~ShaderBuffer();

        inline int Handle() const { return handle; }
        inline int BufferIndex() const { return bufferIndex; }
        inline BufferType Type() const { return type; }
        inline int Float4BlockCount() const { return float4BlockCount; }

        inline GLenum BufferTargetType() const { return bufferTargetType; }
        inline GLenum BufferRangeTargetType() const { return bufferRangeTargetType; }
        inline GLenum ProgramInterfaceType() const { return programInterfaceType; }

        void Allocate(BufferType type, int float4BlockCount);
        void Allocate(BufferType type, int float4BlockCount, GLenum usage = GL_DYNAMIC_DRAW);
        void Allocate(BufferType type, const ShaderBufferContent& data, GLenum usage = GL_DYNAMIC_DRAW);

        void BindToIndex(int bufferIndex);
        int GetProgramResourceIndex(GLuint programHandle, const char* blockName);
        void Bind();
        void Unbind();

        void Update(const ShaderBufferContent& data);

    };

}
}
}
