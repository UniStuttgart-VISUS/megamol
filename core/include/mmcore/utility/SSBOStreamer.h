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
#include <omp.h>

namespace megamol {
namespace core {
namespace utility {

    /// A class that helps you stream some memory to a persistently mapped
    /// buffer that can be used as a SSBO. Abstracts some micro-management
    /// like items/chunk and the sync objects. You can align multiple streamers
    /// by giving the first a desired buffer size and make all others follow
    /// the resulting GetMaxNumItemsPerChunk to set their buffer sizes automatically.
    /// Note that the user must SignalCompletion after the rendering command if
    /// the buffer can be freed afterwards (or re-upload of data will be performed)
    /// See NGSphereRenderer for a usage example.
    class MEGAMOLCORE_API SSBOStreamer {
    public:

         SSBOStreamer(const std::string& debugLabel = std::string());
        ~SSBOStreamer();

        /// this is for defining the max number of items that fit in a desired buffer size,
        /// i.e. for the largest data stream, the 'master'
        /// @param data the pointer to the original data
        /// @param srcStride the size of a single data item in the original data
        /// @param dstStride the size of a single data item that will be uploaded
        ///                   and must not be split across buffers
        /// @param numItems the length of the original data in multiples of stride
        /// @param numBuffers how long the ring buffer should be
        /// @param bufferSize the size of a ring buffer in bytes
        /// @returns number of chunks
        GLuint SetDataWithSize(const void *data, GLuint srcStride, GLuint dstStride, size_t numItems,
            GLuint numBuffers, GLuint bufferSize);

        /// this is for the smaller data streams that need be aligned with numChunks of the
        /// 'master' stream.
        /// @param data the pointer to the original data
        /// @param srcStride the size of a single data item in the original data
        /// @param dstStride the size of a single data item that will be uploaded
        ///                   and must not be split across buffers
        /// @param numItems the length of the original data in multiples of stride
        /// @param numBuffers how long the ring buffer should be
        /// @param numItemsPerChunk number of items per chunk in the master buffer
        /// @returns the size of a ring buffer in bytes
        GLuint SetDataWithItems(const void *data, GLuint srcStride, GLuint dstStride, size_t numItems,
            GLuint numBuffers, GLuint numItemsPerChunk);

        /// @param idx the chunk to upload [0..SetData()-1]
        /// @param numItems returns the number of items in this chunk
        ///                 (last one is probably shorter than bufferSize)
        /// @param sync returns the internal ID of a sync object abstraction
        /// @param dstOffset the buffer offset required for binding the buffer range 
        /// @param dstLength the buffer length required for binding the buffer range
        void UploadChunk(unsigned int idx, GLuint& numItems, unsigned int& sync,
            GLsizeiptr& dstOffset, GLsizeiptr& dstLength);

        /// use this uploader if you want to add a per-item transformation
        /// that will be executed inside an omp parallel for
        /// @param idx the chunk to upload [0..SetData()-1]
        /// @param copyOp the lambda you want to execute. A really hacky subset-changing
        ///            one could be:
        ///            [vertStride](const char *src, char *dst) -> void {
        ///                memcpy(dst, src, vertStride);
        ///                *reinterpret_cast<float *>(dst + 4) =
        ///                    *reinterpret_cast<const float *>(src + 4) - 100.0f;
        ///            }
        /// @param numItems returns the number of items in this chunk
        ///                 (last one is probably shorter than bufferSize)
        /// @param sync returns the internal ID of a sync object abstraction
        /// @param dstOffset the buffer offset required for binding the buffer range 
        /// @param dstLength the buffer length required for binding the buffer range
        template<class fun>
        void UploadChunk(unsigned int idx, fun copyOp,
            GLuint& numItems, unsigned int& sync,
            GLsizeiptr& dstOffset, GLsizeiptr& dstLength);

        /// @param sync the abstract sync object to signal as done
        void SignalCompletion(unsigned int sync);

		/// @param numItemsPerChunk the minimum number of items per chunk
		/// @param up rounds up if true, otherwise rounds down.
		/// @returns the alignment-friendly (rounded) number of items per chunk
        GLuint GetNumItemsPerChunkAligned(GLuint numItemsPerChunk, bool up = false) const;

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
        void genBufferAndMap(GLuint numBuffers, GLuint bufferSize);

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
        int numThr;
        std::string debugLabel;
        int offsetAlignment = 0;
    };

    template<class fun>
    void SSBOStreamer::UploadChunk(unsigned int idx, fun copyOp, GLuint& numItems,
        unsigned int& sync, GLsizeiptr& dstOffset, GLsizeiptr& dstLength) {
        if (theData == nullptr || idx > this->numChunks - 1) return;

        // we did not succeed doing anything yet
        numItems = sync = 0;

        dstOffset = this->bufferSize * this->currIdx;
        GLsizeiptr srcOffset = this->numItemsPerChunk * this->srcStride * idx;
        char *dst = static_cast<char*>(this->mappedMem) + dstOffset;
        const char *src = static_cast<const char*>(this->theData) + srcOffset;
        const size_t itemsThisTime = std::min<unsigned int>(
            this->numItems - idx * this->numItemsPerChunk, this->numItemsPerChunk);
        dstLength = itemsThisTime * this->dstStride;
        const void *srcEnd = src + itemsThisTime * srcStride;

        //printf("going to upload %llu x %u bytes to offset %lld from %lld\n", itemsThisTime,
        //    this->dstStride, dstOffset, srcOffset);

        waitSignal(this->fences[currIdx]);

#pragma omp parallel for
        for (INT64 i = 0; i < itemsThisTime; ++i) {
            copyOp(src + i * this->srcStride, dst + i * this->dstStride);
        }

        glFlushMappedNamedBufferRange(this->theSSBO,
            this->bufferSize * this->currIdx, itemsThisTime * this->dstStride);
        numItems = itemsThisTime;

        sync = currIdx;
        currIdx = (currIdx + 1) % this->numBuffers;
    }


} /* end namespace utility */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_SSBOSTREAMER_H_INCLUDED */