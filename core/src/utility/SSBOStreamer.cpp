/*
* SSBOStreamer.cpp
*
* Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "mmcore/utility/SSBOStreamer.h"
#include <algorithm>
#include "vislib/assert.h"
#include <iostream>
#include <sstream>

using namespace megamol::core::utility;

SSBOStreamer::SSBOStreamer(const std::string& debugLabel)
    : theSSBO(0), bufferSize(0), numBuffers(0), srcStride(0), dstStride(0), theData(nullptr),
      mappedMem(nullptr), numItems(0), numChunks(0), currIdx(0), numThr(omp_get_max_threads()), debugLabel(debugLabel) {
}

SSBOStreamer::~SSBOStreamer() {
    if (this->mappedMem != nullptr && this->theSSBO != 0) {
        glUnmapNamedBuffer(this->theSSBO);
    }
    if (this->theSSBO != 0) {
        glDeleteBuffers(1, &this->theSSBO);
    }
    for (auto &x : fences) {
        if (x) {
            glDeleteSync(x);
        }
    }
}

GLuint SSBOStreamer::SetDataWithSize(const void *data, GLuint srcStride, GLuint dstStride,
        size_t numItems, GLuint numBuffers, GLuint bufferSize) {

    if (data == nullptr || srcStride == 0 || dstStride == 0 || numItems == 0 ||
            numBuffers == 0 || bufferSize == 0) {
        theData = nullptr;
        return 0;
    }

    genBufferAndMap(numBuffers, bufferSize);

    this->dstStride = dstStride;
    this->srcStride = dstStride;
    this->numItems = numItems;
    this->theData = data;
    this->numItemsPerChunk = GetNumItemsPerChunkAligned(bufferSize / dstStride);
    this->numChunks = (numItems + numItemsPerChunk - 1) / numItemsPerChunk; // round up int division!
    this->fences.resize(numBuffers, nullptr);
    return numChunks;
}

GLuint SSBOStreamer::GetNumItemsPerChunkAligned(GLuint numItemsPerChunk, bool up) const {
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

void SSBOStreamer::genBufferAndMap(GLuint numBuffers, GLuint bufferSize) {
    if (this->theSSBO == 0) {
        glGenBuffers(1, &this->theSSBO);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->theSSBO);
#if _DEBUG
        glObjectLabel(GL_BUFFER, this->theSSBO, debugLabel.length(), debugLabel.c_str());
#endif
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }
    if (bufferSize != this->bufferSize || numBuffers != this->numBuffers) {
        if (this->mappedMem != nullptr && this->theSSBO != 0) {
            glUnmapNamedBuffer(this->theSSBO);
        }
        const size_t mapSize = bufferSize * numBuffers;
        glNamedBufferStorage(this->theSSBO, mapSize, nullptr,
                             GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT);
        this->mappedMem = glMapNamedBufferRange(this->theSSBO, 0,
                                                mapSize,
                                                GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT);
        if (this->mappedMem == nullptr) {
	    std::stringstream err;
	    err << std::string("SSBOStreamer: could not map memory (") << mapSize << std::string(" bytes)") << std::endl;
	    throw std::runtime_error(err.str());
        }
        this->bufferSize = bufferSize;
        this->numBuffers = numBuffers;
    }
}

GLuint SSBOStreamer::SetDataWithItems(const void *data, GLuint srcStride, GLuint dstStride, size_t numItems,
        GLuint numBuffers, GLuint numItemsPerChunk) {
    if (data == nullptr || srcStride == 0 || dstStride == 0 || numItems == 0 ||
        numBuffers == 0 || numItemsPerChunk == 0) {
        theData = nullptr;
        return 0;
    }

    const GLuint bufferSize = numItemsPerChunk * dstStride;

    genBufferAndMap(numBuffers, bufferSize);

    this->dstStride = dstStride;
    this->srcStride = dstStride;
    this->numItems = numItems;
    this->theData = data;
    this->numItemsPerChunk = numItemsPerChunk;
    this->numChunks = (numItems + numItemsPerChunk - 1) / numItemsPerChunk; // round up int division!
    this->fences.resize(numBuffers, nullptr);
    return this->bufferSize;
}

void SSBOStreamer::UploadChunk(unsigned int idx, GLuint& numItems, unsigned int& sync,
        GLsizeiptr& dstOffset, GLsizeiptr& dstLength) {
    if (theData == nullptr || idx > this->numChunks - 1) return;
    
    // we did not succeed doing anything yet
    numItems = sync = 0;

    dstOffset = this->bufferSize * this->currIdx;
    GLsizeiptr srcOffset = this->numItemsPerChunk * this->srcStride * idx;
    void *dst = static_cast<char*>(this->mappedMem) + dstOffset;
    const void *src = static_cast<const char*>(this->theData) + srcOffset;
    const size_t itemsThisTime = std::min<unsigned int>(
        this->numItems - idx * this->numItemsPerChunk, this->numItemsPerChunk);
    dstLength = itemsThisTime * this->dstStride;

    //printf("going to upload %llu x %u bytes to offset %lld from %lld\n", itemsThisTime,
    //    this->dstStride, dstOffset, srcOffset);

    waitSignal(this->fences[currIdx]);

    ASSERT(this->srcStride == this->dstStride);
    memcpy(dst, src, itemsThisTime * this->srcStride);

    glFlushMappedNamedBufferRange(this->theSSBO, 
        dstOffset, itemsThisTime * this->dstStride);
    numItems = itemsThisTime;

    sync = currIdx;
    currIdx = (currIdx + 1) % this->numBuffers;
}

void SSBOStreamer::SignalCompletion(unsigned int sync) {
    queueSignal(this->fences[sync]);
}


void SSBOStreamer::queueSignal(GLsync& syncObj) {
    if (syncObj) {
        glDeleteSync(syncObj);
    }
    syncObj = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}

void SSBOStreamer::waitSignal(GLsync& syncObj) {
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
