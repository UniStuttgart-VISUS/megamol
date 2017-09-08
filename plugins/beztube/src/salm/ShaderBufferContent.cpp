#include "stdafx.h"
#include "ShaderBufferContent.h"

megamol::beztube::salm::ShaderBufferContent::ShaderBufferContent() : content(nullptr), float4BlockCount(0), size(0) {
}

megamol::beztube::salm::ShaderBufferContent::~ShaderBufferContent() {
    delete[] content;
    content = nullptr;
    float4BlockCount = 0;
    size = 0;
}

void megamol::beztube::salm::ShaderBufferContent::Allocate(unsigned int float4BlockCount) {
    delete[] content;
    content = new float[float4BlockCount * 4];
    this->float4BlockCount = float4BlockCount;
    size = float4BlockCount * 4 * sizeof(float);
}
