/*
* SSBOBufferArray.h
*
* Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
* Alle Rechte vorbehalten.
*/

#ifndef MEGAMOLCORE_SSBOBUFFERARRAY_H_INCLUDED
#define MEGAMOLCORE_SSBOBUFFERARRAY_H_INCLUDED
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

    /// A class that helps you chunk and upload a huge buffer to multiple
    /// 'static' SSBOs - meaning they will not change very often. You can align multiple streamers
    /// by giving the first a desired buffer size and make all others follow
    /// the resulting GetMaxNumItemsPerChunk to set their buffer sizes automatically.
    /// Note that the user must SignalCompletion after the rendering command if
    /// the buffer can be freed afterwards (or re-upload of data will be performed)
    /// See NGSphereRenderer for a usage example.
class MEGAMOLCORE_API SSBOBufferArray {
    public:

         SSBOBufferArray(const std::string& debugLabel = std::string());
        ~SSBOBufferArray();

        /// this is for defining the max number of items that fit in a desired chunk size,
        /// i.e. for the largest data stream, the 'master'
        /// @param data the pointer to the original data
        /// @param srcStride the size of a single data item in the original data
        /// @param dstStride the size of a single data item that will be uploaded
        ///                   and must not be split across buffers
        /// @param numItems the length of the original data in multiples of stride
        /// @param bufferSize the size of a ring buffer in bytes
        /// @param sync returns the internal ID of a sync object abstraction
        /// @returns number of chunks
        GLuint SetDataWithSize(const void *data, GLuint srcStride, GLuint dstStride, size_t numItems, GLuint bufferSize);

        /// this is for the smaller data streams that need be aligned with numChunks of the
        /// 'master' stream.
        /// @param data the pointer to the original data
        /// @param srcStride the size of a single data item in the original data
        /// @param dstStride the size of a single data item that will be uploaded
        ///                   and must not be split across buffers
        /// @param numItems the length of the original data in multiples of stride
        /// @param numItemsPerChunk number of items per chunk in the master buffer
        /// @param sync returns the internal ID of a sync object abstraction
        /// @returns the size of a ring buffer in bytes
        GLuint SetDataWithItems(const void* data, GLuint srcStride, GLuint dstStride, size_t numItems,
            GLuint numItemsPerChunk);

        /// @param sync the abstract sync object to signal as done
        void SignalCompletion();

        /// @param numItemsPerChunk the minimum number of items per chunk
        /// @param up rounds up if true, otherwise rounds down.
        /// @returns the alignment-friendly (rounded) number of items per chunk
        GLuint GetNumItemsPerChunkAligned(GLuint numItemsPerChunk, bool up = false) const;

        /// returns the GL object of the SSBO corresponding to chunk idx
        GLuint GetHandle(unsigned int idx) const {
            if (idx >= 0 && idx < theSSBOs.size()) {
                return theSSBOs[idx];
            } else {
                return 0;
            }
        }

        GLuint GetNumItems(unsigned int idx) const {
            if (idx >= 0 && idx < theSSBOs.size()) {
                return actualItemsPerChunk[idx];
            } else {
                return 0;
            }
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

        std::vector<GLuint> theSSBOs;
        std::vector<GLuint> actualItemsPerChunk;
        /// in bytes!
        GLuint bufferSize;
        GLuint numBuffers;
        GLuint srcStride;
        GLuint dstStride;
        const void* theData;
        size_t numItems;
        GLuint numChunks;
        GLuint numItemsPerChunk;
        GLsync fence;
        int numThr;
        std::string debugLabel;
        int offsetAlignment = 0;
    };

} /* end namespace utility */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_SSBOBUFFERARRAY_H_INCLUDED */