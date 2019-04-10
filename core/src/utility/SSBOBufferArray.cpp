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
    : theSSBOs(), fence(0), bufferSize(0), numBuffers(0), srcStride(0), dstStride(0), theData(nullptr),
      numItems(0), numChunks(0), debugLabel(debugLabel) {

}


SSBOBufferArray::~SSBOBufferArray() {

    if (this->theSSBOs.size() > 0) {
        glDeleteBuffers(this->theSSBOs.size(), this->theSSBOs.data());
    }
    if (this->fence) {
        glDeleteSync(this->fence);
    }
}


GLuint SSBOBufferArray::SetDataWithSize(
    const void* data, GLuint srcStride, GLuint dstStride, size_t numItems, GLuint bufferSize) {

    if (data == nullptr || srcStride == 0 || dstStride == 0 || numItems == 0 || bufferSize == 0) {
        this->theData = nullptr;
        return 0;
    }

    this->dstStride = dstStride;
    this->srcStride = srcStride;
    this->numItems = numItems;
    this->theData = data;
    this->numItemsPerChunk = this->GetNumItemsPerChunkAligned(bufferSize / dstStride);
    this->numChunks = (this->numItems + this->numItemsPerChunk - 1) / this->numItemsPerChunk; // round up int division!

    const auto itemSize = this->srcStride * this->numItemsPerChunk;

    this->waitSignal(this->fence);
    if (!this->theSSBOs.empty()) {
        glDeleteBuffers(this->theSSBOs.size(), this->theSSBOs.data());
    }

    this->theSSBOs.resize(this->numChunks);
    this->actualItemsPerChunk.resize(this->numChunks);

    glGenBuffers(this->numChunks, this->theSSBOs.data());
    for (unsigned int x = 0; x < this->numChunks; ++x) {
        std::string sublabel = this->debugLabel + "(" + std::to_string(x+1) + "/" + std::to_string(this->numChunks) + ")";
        const char* ptr = reinterpret_cast<const char*>(this->theData);
        ptr += itemSize * x;
        this->actualItemsPerChunk[x] = std::min<GLuint>(this->numItemsPerChunk, (this->numItems - (this->numItemsPerChunk * x)));
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->theSSBOs[x]);
#if _DEBUG
        glObjectLabel(GL_BUFFER, this->theSSBOs[x], this->debugLabel.length(), this->debugLabel.c_str());
#endif
        glBufferData(GL_SHADER_STORAGE_BUFFER, this->actualItemsPerChunk[x] * this->srcStride, ptr, GL_STATIC_DRAW);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    return this->numChunks;
}


GLuint SSBOBufferArray::SetDataWithItems(const void* data, GLuint srcStride, GLuint dstStride, size_t numItems,
    GLuint numItemsPerChunk) {

    if (data == nullptr || srcStride == 0 || dstStride == 0 || numItems == 0 ||  numItemsPerChunk == 0) {
        this->theData = nullptr;
        return 0;
    }

    this->dstStride = dstStride;
    this->srcStride = srcStride;
    this->numItems = numItems;
    this->theData = data;
    this->numItemsPerChunk = numItemsPerChunk;
    this->numChunks = (this->numItems + this->numItemsPerChunk - 1) / this->numItemsPerChunk; // round up int division!

    const auto itemSize = this->srcStride * this->numItemsPerChunk;

    this->waitSignal(this->fence);
    if (!this->theSSBOs.empty()) {
        glDeleteBuffers(this->theSSBOs.size(), this->theSSBOs.data());
    }

    this->theSSBOs.resize(this->numChunks);
    this->actualItemsPerChunk.resize(this->numChunks);

    glGenBuffers(this->numChunks, this->theSSBOs.data());
    for (unsigned int x = 0; x < this->numChunks; ++x) {
        std::string sublabel = this->debugLabel + "(" + std::to_string(x + 1) + "/" + std::to_string(this->numChunks) + ")";
        const char* ptr = reinterpret_cast<const char*>(this->theData);
        ptr += itemSize * x;
        this->actualItemsPerChunk[x] = std::min<GLuint>(this->numItemsPerChunk, (this->numItems - (this->numItemsPerChunk * x)));
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->theSSBOs[x]);
#if _DEBUG
        glObjectLabel(GL_BUFFER, this->theSSBOs[x], this->debugLabel.length(), this->debugLabel.c_str());
#endif
        glBufferData(GL_SHADER_STORAGE_BUFFER, this->actualItemsPerChunk[x] * this->srcStride, ptr, GL_STATIC_DRAW);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    return this->bufferSize;
}


GLuint SSBOBufferArray::GetNumItemsPerChunkAligned(GLuint numItemsPerChunk, bool up) const {

    // Lazy initialisation of offset alignment because OGl context must be available.
    if (this->offsetAlignment == 0) {
        glGetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, (GLint*)&this->offsetAlignment);
    }

    // Rounding the number of items per chunk is important for alignment and thus performance.
    // That means, if we synchronize with another buffer that has tiny items, we have to make 
    // sure that we do not get non-aligned chunks with due to the number of items.
    // For modern GPUs, we use GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT (which for NVidia results in 32),
    // i.e., we upload in multiples of eight to get 8 * 4 = 32 (no data shorter than uint32_t is allowed).
    const GLuint multiRound = this->offsetAlignment / 4;

    return (((numItemsPerChunk) / multiRound) + (up ? 1 : 0)) * multiRound;
}


void SSBOBufferArray::SignalCompletion() {

    queueSignal(this->fence);
}


void SSBOBufferArray::queueSignal(GLsync& syncObj) {

    if (syncObj) {
        glDeleteSync(syncObj);
    }
    syncObj = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}


void SSBOBufferArray::waitSignal(GLsync& syncObj) {

    if (syncObj) {
        //XXX: Spinlocks in user code are a really bad idea.
        while (true) {
            const GLenum wait = glClientWaitSync(syncObj, GL_SYNC_FLUSH_COMMANDS_BIT, 1);
            if (wait == GL_ALREADY_SIGNALED || wait == GL_CONDITION_SATISFIED) {
                return;
            }
        }
    }
}
