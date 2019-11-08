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
    //if (this->fence) {
    //    glDeleteSync(this->fence);
    //}
}


void SSBOBufferArray::upload(const std::function<void(void *, const void *)> &copyOp) {
    const auto chunk_src_size = this->srcStride * this->numItemsPerChunk;

    // either we can grab all the data at once or we need the copyOp to re-arrange stuff for us
    ASSERT(this->dstStride == this->srcStride || copyOp);

    //this->waitSignal(this->fence);
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
    this->numItemsPerChunk = this->GetNumItemsPerChunkAligned(maxBufferSize / dstStride);
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


GLuint SSBOBufferArray::GetNumItemsPerChunkAligned(GLuint numItemsPerChunk, bool up) const {

    // Lazy initialization of offset alignment because OGl context must be available.
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


//void SSBOBufferArray::SignalCompletion() { queueSignal(this->fence); }
//
//
//void SSBOBufferArray::queueSignal(GLsync& syncObj) {
//
//    if (syncObj) {
//        glDeleteSync(syncObj);
//    }
//    syncObj = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
//}
//
//
//void SSBOBufferArray::waitSignal(GLsync& syncObj) {
//
//    if (syncObj) {
//        // XXX: Spin locks in user code are a really bad idea.
//        while (true) {
//            const GLenum wait = glClientWaitSync(syncObj, GL_SYNC_FLUSH_COMMANDS_BIT, 1);
//            if (wait == GL_ALREADY_SIGNALED || wait == GL_CONDITION_SATISFIED) {
//                return;
//            }
//        }
//    }
//}
