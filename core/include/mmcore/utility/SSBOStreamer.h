/*
* SSBOStreamer.h
*
* Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
* Alle Rechte vorbehalten.
*/

#ifndef MEGAMOLCORE_SSBOSTREAMER_H_INCLUDED
#define MEGAMOLCORE_SSBOSTREAMER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include <vector>
#include <algorithm>
#include <cinttypes>
#include "mmcore/api/MegaMolCore.std.h"

namespace megamol {
namespace core {
namespace utility {

    class MEGAMOLCORE_API SSBOStreamer {
    public:

         SSBOStreamer();
        ~SSBOStreamer();

        /// @param data the pointer to the original data
        /// @param srcStride the size of a single data item in the original data
        /// @param dstStride the size of a single data item that will be uploaded
        ///                   and must not be split across buffers
        /// @param numItems the length of the original data in multiples of stride
        /// @param numBuffers how long the ring buffer should be
        /// @param bufferSize the size of a ring buffer in bytes
        /// @returns number of chunks
        GLuint SetData(const void *data, GLuint srcStride, GLuint dstStride, size_t numItems,
            GLuint numBuffers, GLuint bufferSize);

        /// @param idx the chunk to upload [0..SetData()-1]
        /// @param numItems returns the number of items in this chunk
        ///                 (last one is probably shorter than bufferSize)
        /// @param sync returns the internal ID of a sync object abstraction
        /// @param dstOffset the buffer offset required for binding the buffer range 
        /// @param dstLength the buffer length required for binding the buffer range
        void UploadChunk(unsigned int idx, GLuint& numItems, unsigned int& sync,
            GLsizeiptr& dstOffset, GLsizeiptr& dstLength);

        /// use this uploader if you want to add a per-item transformation
        /// like [](double d) -> float { return d - localOrigin; }
        template<class TSrc, class TDst, class fun>
        void UploadChunk(unsigned int idx, fun unaryOp,
            GLuint& numItems, unsigned int& sync,
            GLsizeiptr& dstOffset, GLsizeiptr& dstLength);

        /// @param sync the abstract sync object to wait for
        void WaitForCompletion(unsigned int sync);

        GLuint GetHandle(void) const {
            return theSSBO;
        }

        GLuint GetNumChunks(void) const {
            return numChunks;
        }

        GLuint GetMaxNumItemsPerChunk(void) const {
            return numItemsPerChunk;
        }

    private:
        static void queueSignal(GLsync &syncObj);
        static void waitSignal(GLsync &syncObj);

        GLuint theSSBO;
        /// in bytes!
        GLuint bufferSize;
        GLuint numBuffers;
        GLuint srcStride;
        GLuint dstStride;
        const void* theData;
        void* mappedMem;
        size_t numItems;
        GLuint numChunks;
        GLuint numItemsPerChunk;
        /// which ring element we upload to next
        GLuint currIdx;
        std::vector<GLsync> fences;
    };

    template<class TSrc, class TDst, class fun>
    void SSBOStreamer::UploadChunk(unsigned int idx, fun unaryOp, GLuint& numItems,
        unsigned int& sync, GLsizeiptr& dstOffset, GLsizeiptr& dstLength) {
        if (theData == nullptr || idx > this->numChunks - 1) return;

        // we did not succeed doing anything yet
        numItems = sync = 0;

        dstOffset = this->bufferSize * this->currIdx;
        GLsizeiptr srcOffset = this->numItemsPerChunk * this->srcStride * idx;
        void *dst = static_cast<char*>(this->mappedMem) + dstOffset;
        const char *src = static_cast<const char*>(this->theData) + srcOffset;
        const size_t itemsThisTime = std::min<unsigned int>(
            this->numItems - idx * this->numItemsPerChunk, this->numItemsPerChunk);
        dstLength = itemsThisTime * this->dstStride;
        const void *srcEnd = src + itemsThisTime * srcStride;

        //printf("going to upload %llu x %u bytes to offset %lld from %lld\n", itemsThisTime,
        //    this->dstStride, dstOffset, srcOffset);

        waitSignal(this->fences[currIdx]);

        std::transform(reinterpret_cast<const TSrc*>(src), 
            reinterpret_cast<const TSrc*>(srcEnd),
            reinterpret_cast<TDst*>(dst), unaryOp);

        glFlushMappedNamedBufferRange(this->theSSBO,
            this->bufferSize * this->currIdx, itemsThisTime * this->dstStride);
        numItems = itemsThisTime;

        queueSignal(this->fences[currIdx]);
        sync = currIdx;
        currIdx = (currIdx + 1) % this->numBuffers;
    }


} /* end namespace utility */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_SSBOSTREAMER_H_INCLUDED */