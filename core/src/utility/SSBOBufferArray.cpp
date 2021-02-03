/*
 * SSBOBufferArray.cpp
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/utility/SSBOBufferArray.h"


using namespace megamol::core::utility;


SSBOBufferArray::SSBOBufferArray(const std::string& debugLabel)
    : theSSBOs()
    //, fence(0)
    , maxBufferSize(0)
    , numBuffers(0)
    , srcStride(0)
    , dstStride(0)
    , theData(nullptr)
    , numItems(0)
    , numChunks(0)
    , numItemsPerChunk(0)
    , debugLabel(debugLabel) {}


SSBOBufferArray::~SSBOBufferArray() {

    if (!this->theSSBOs.empty()) {
        glDeleteBuffers(this->theSSBOs.size(), this->theSSBOs.data());
    }
}


void SSBOBufferArray::upload(const std::function<void(void *, const void *)> &copyOp) {
    const auto chunk_src_size = this->srcStride * this->numItemsPerChunk;

    if (this->maxSSBOSize == 0) {
        glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &this->maxSSBOSize);
    }
    ASSERT(this->maxBufferSize <= this->maxSSBOSize && "The size per SSBO is larger than your OpenGL implementation allows!");

    // either we can grab all the data at once or we need the copyOp to re-arrange stuff for us
    ASSERT(this->dstStride == this->srcStride || copyOp);

    if (!this->theSSBOs.empty()) {
        glDeleteBuffers(this->theSSBOs.size(), this->theSSBOs.data());
    }

    this->theSSBOs.resize(this->numChunks);
    this->actualItemsPerChunk.resize(this->numChunks);
    std::vector<char> temp;
    if (copyOp) {
        temp.resize(this->dstStride * std::min<uint64_t>(this->numItemsPerChunk, this->numItems));
    }

    glGenBuffers(this->numChunks, this->theSSBOs.data());
    for (unsigned int x = 0; x < this->numChunks; ++x) {
        std::string sublabel =
            this->debugLabel + "(" + std::to_string(x + 1) + "/" + std::to_string(this->numChunks) + ")";
        const char* ptr = reinterpret_cast<const char*>(this->theData);
        ptr += chunk_src_size * x;
        this->actualItemsPerChunk[x] =
            std::min<GLuint>(this->numItemsPerChunk, (this->numItems - (this->numItemsPerChunk * x)));
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->theSSBOs[x]);
#if _DEBUG
        glObjectLabel(GL_BUFFER, this->theSSBOs[x], sublabel.length(), sublabel.c_str());
#endif
        if (copyOp) {
//#pragma omp parallel for
            for (auto l = 0; l < this->actualItemsPerChunk[x]; ++l) {
                // todo we are all going to die
                copyOp(&temp[this->dstStride * l], &ptr[this->srcStride * l]);
            }
            glBufferData(GL_SHADER_STORAGE_BUFFER, this->actualItemsPerChunk[x] * this->dstStride, temp.data(), GL_STATIC_DRAW);
        } else {
            glBufferData(GL_SHADER_STORAGE_BUFFER, this->actualItemsPerChunk[x] * this->srcStride, ptr, GL_STATIC_DRAW);
        }
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void SSBOBufferArray::SetData(const void* data, GLuint srcStride, GLuint dstStride, size_t numItems,
    const std::function<void(void*, const void*)>& copyOp) {

    if (data == nullptr || srcStride == 0 || dstStride == 0 || numItems == 0) {
        this->theData = nullptr;
        return;
    }

    this->dstStride = dstStride;
    this->srcStride = srcStride;
    this->numItems = numItems;
    this->theData = data;
    this->numItemsPerChunk = numItems;
    this->numChunks = 1;
    this->maxBufferSize = numItems * dstStride;

    upload(copyOp);
}

GLuint SSBOBufferArray::SetDataWithSize(const void* data, GLuint srcStride, GLuint dstStride, size_t numItems,
    GLuint maxBufferSize, const std::function<void(void*, const void*)>& copyOp) {

    if (data == nullptr || srcStride == 0 || dstStride == 0 || numItems == 0 || maxBufferSize == 0) {
        this->theData = nullptr;
        return 0;
    }

    this->dstStride = dstStride;
    this->srcStride = srcStride;
    this->numItems = numItems;
    this->theData = data;
    this->numItemsPerChunk = maxBufferSize / dstStride;
    this->numChunks = (this->numItems + this->numItemsPerChunk - 1) / this->numItemsPerChunk; // round up int division!
    this->maxBufferSize = maxBufferSize;

    upload(copyOp);

    return this->numChunks;
}


GLuint SSBOBufferArray::SetDataWithItems(const void* data, GLuint srcStride, GLuint dstStride, size_t numItems,
    GLuint numItemsPerChunk, const std::function<void(void*, const void*)>& copyOp) {

    if (data == nullptr || srcStride == 0 || dstStride == 0 || numItems == 0 || numItemsPerChunk == 0) {
        this->theData = nullptr;
        return 0;
    }

    this->dstStride = dstStride;
    this->srcStride = srcStride;
    this->numItems = numItems;
    this->theData = data;
    this->numItemsPerChunk = numItemsPerChunk;
    this->numChunks = (this->numItems + this->numItemsPerChunk - 1) / this->numItemsPerChunk; // round up int division!
    this->maxBufferSize = this->dstStride * numItemsPerChunk;

    upload(copyOp);

    return this->maxBufferSize;
}


