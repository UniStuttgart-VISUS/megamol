/*
* SSBOBufferArray.cpp
*
* Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "mmcore/utility/SSBOBufferArray.h"
#include <algorithm>
#include "vislib/assert.h"
#include <iostream>
#include <sstream>

using namespace megamol::core::utility;

SSBOBufferArray::SSBOBufferArray(const std::string& debugLabel)
    : theSSBOs(), fence(0), bufferSize(0), numBuffers(0), srcStride(0), dstStride(0), theData(nullptr),
      numItems(0), numChunks(0), numThr(omp_get_max_threads()), debugLabel(debugLabel) {
}

SSBOBufferArray::~SSBOBufferArray() {
    if (this->theSSBOs.size() > 0) {
        glDeleteBuffers(theSSBOs.size(), this->theSSBOs.data());
    }
    if (this->fence) {
        glDeleteSync(this->fence);
    }
}

GLuint SSBOBufferArray::SetDataWithSize(
    const void* data, GLuint srcStride, GLuint dstStride, size_t numItems, GLuint bufferSize) {

    if (data == nullptr || srcStride == 0 || dstStride == 0 || numItems == 0 ||
            bufferSize == 0) {
        theData = nullptr;
        return 0;
    }

    this->dstStride = dstStride;
    this->srcStride = dstStride;
    this->numItems = numItems;
    this->theData = data;
    this->numItemsPerChunk = GetNumItemsPerChunkAligned(bufferSize / dstStride);
    this->numChunks = (numItems + numItemsPerChunk - 1) / numItemsPerChunk; // round up int division!

    const auto itemSize = this->srcStride * numItemsPerChunk;

    waitSignal(this->fence);
    if (!this->theSSBOs.empty()) {
        glDeleteBuffers(this->theSSBOs.size(), this->theSSBOs.data());
    }

    this->theSSBOs.resize(numChunks);
    this->actualItemsPerChunk.resize(numChunks);
    glGenBuffers(numChunks, this->theSSBOs.data());
    for (unsigned int x = 0; x < numChunks; ++x) {
        std::string sublabel = debugLabel + "(" + std::to_string(x+1) + "/" + std::to_string(numChunks) + ")";
        const char* ptr = reinterpret_cast<const char*>(data);
        ptr += itemSize * x;
        actualItemsPerChunk[x] = std::min<GLuint>(numItemsPerChunk, numItems - numItemsPerChunk * x);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->theSSBOs[x]);
#if _DEBUG
        glObjectLabel(GL_BUFFER, this->theSSBOs[x], debugLabel.length(), debugLabel.c_str());
#endif
        glBufferData(GL_SHADER_STORAGE_BUFFER, actualItemsPerChunk[x] * this->srcStride, ptr, GL_STATIC_DRAW);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return numChunks;
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

GLuint SSBOBufferArray::SetDataWithItems(const void* data, GLuint srcStride, GLuint dstStride, size_t numItems,
    GLuint numItemsPerChunk) {
    if (data == nullptr || srcStride == 0 || dstStride == 0 || numItems == 0 ||
        numItemsPerChunk == 0) {
        theData = nullptr;
        return 0;
    }

    this->dstStride = dstStride;
    this->srcStride = dstStride;
    this->numItems = numItems;
    this->theData = data;
    this->numItemsPerChunk = numItemsPerChunk;
    this->numChunks = (numItems + numItemsPerChunk - 1) / numItemsPerChunk; // round up int division!

    const auto itemSize = this->srcStride * numItemsPerChunk;

    waitSignal(this->fence);
    if (!this->theSSBOs.empty()) {
        glDeleteBuffers(this->theSSBOs.size(), this->theSSBOs.data());
    }

    this->theSSBOs.resize(numChunks);
    this->actualItemsPerChunk.resize(numChunks);
    glGenBuffers(numChunks, this->theSSBOs.data());
    for (unsigned int x = 0; x < numChunks; ++x) {
        std::string sublabel = debugLabel + "(" + std::to_string(x + 1) + "/" + std::to_string(numChunks) + ")";
        const char* ptr = reinterpret_cast<const char*>(data);
        ptr += itemSize * x;
        actualItemsPerChunk[x] = std::min<GLuint>(numItemsPerChunk, numItems - numItemsPerChunk * x);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->theSSBOs[x]);
#if _DEBUG
        glObjectLabel(GL_BUFFER, this->theSSBOs[x], debugLabel.length(), debugLabel.c_str());
#endif
        glBufferData(GL_SHADER_STORAGE_BUFFER, actualItemsPerChunk[x] * this->srcStride, ptr, GL_STATIC_DRAW);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    return this->bufferSize;
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
