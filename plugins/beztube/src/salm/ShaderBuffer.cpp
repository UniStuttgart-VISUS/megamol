#include "stdafx.h"
#include "ShaderBuffer.h"

megamol::beztube::salm::ShaderBuffer::ShaderBuffer() : handle(0) {
    ::glGenBuffers(1, &handle);
    type = INVALID;
    bufferTargetType = ToBufferTarget(type);
    bufferRangeTargetType = ToBufferRangeTarget(type);
    programInterfaceType = ToProgramInterface(type);
    float4BlockCount = 0;
    size = 0;
}

megamol::beztube::salm::ShaderBuffer::~ShaderBuffer() {
    ::glDeleteBuffers(1, &handle);
    type = INVALID;
    bufferTargetType = ToBufferTarget(type);
    bufferRangeTargetType = ToBufferRangeTarget(type);
    programInterfaceType = ToProgramInterface(type);
    handle = 0;
    float4BlockCount = 0;
    size = 0;
}

void megamol::beztube::salm::ShaderBuffer::Allocate(BufferType type, int float4BlockCount) {
    this->type = type;
    bufferTargetType = ToBufferTarget(type);
    bufferRangeTargetType = ToBufferRangeTarget(type);
    programInterfaceType = ToProgramInterface(type);
    this->float4BlockCount = float4BlockCount;
    size = 4 * sizeof(float) * float4BlockCount;
}

void megamol::beztube::salm::ShaderBuffer::Allocate(BufferType type, int float4BlockCount, GLenum usage) {
    this->type = type;
    bufferTargetType = ToBufferTarget(type);
    bufferRangeTargetType = ToBufferRangeTarget(type);
    programInterfaceType = ToProgramInterface(type);
    this->float4BlockCount = float4BlockCount;
    size = 4 * sizeof(float) * float4BlockCount;
    this->usage = usage;

    Bind();
    glBufferData(bufferTargetType, size, nullptr, usage);
    Unbind();
}

void megamol::beztube::salm::ShaderBuffer::Allocate(BufferType type, const ShaderBufferContent& data, GLenum usage) {
    this->type = type;
    bufferTargetType = ToBufferTarget(type);
    bufferRangeTargetType = ToBufferRangeTarget(type);
    programInterfaceType = ToProgramInterface(type);
    this->float4BlockCount = data.Float4BlockCount();
    size = data.Size();
    this->usage = usage;

    Bind();
    glBufferData(bufferTargetType, size, data.Content(), usage);
    Unbind();
}

void megamol::beztube::salm::ShaderBuffer::BindToIndex(int bufferIndex) {
    this->bufferIndex = bufferIndex;
    glBindBufferBase(bufferRangeTargetType, bufferIndex, handle); // Bind to Buffer Index
}

int megamol::beztube::salm::ShaderBuffer::GetProgramResourceIndex(GLuint programHandle, const char* blockName) {
    return glGetProgramResourceIndex(programInterfaceType, programInterfaceType, blockName);
}

void megamol::beztube::salm::ShaderBuffer::Bind() {
    glBindBuffer(bufferTargetType, handle);
}

void megamol::beztube::salm::ShaderBuffer::Unbind() {
    glBindBuffer(bufferTargetType, 0);
}

void megamol::beztube::salm::ShaderBuffer::Update(const ShaderBufferContent& data) {
    if (size == data.Size()) {
        Bind();
        glBufferSubData(bufferTargetType, 0, data.Size(), data.Content());
        Unbind();
    } else {
        Allocate(type, data, usage);
    }
}
