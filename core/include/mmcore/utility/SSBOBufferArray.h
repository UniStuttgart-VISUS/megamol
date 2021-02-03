/*
* SSBOBufferArray.h
*
* Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
* Alle Rechte vorbehalten.
*/

#ifndef MEGAMOLCORE_SSBOBUFFERARRAY_H_INCLUDED
#define MEGAMOLCORE_SSBOBUFFERARRAY_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "mmcore/api/MegaMolCore.std.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/graphics/gl/GLSLShader.h"

#include <vector>
#include <algorithm>
#include <cinttypes>
#include <algorithm>
#include "vislib/assert.h"
#include <iostream>
#include <sstream>
#include <functional>


namespace megamol {
namespace core {
namespace utility {

    /// A class that helps you chunk and upload a huge buffer to multiple
    /// 'static' SSBOs - meaning they will not change very often. You can align multiple streamers
    /// by giving the first a desired buffer size and make all others follow
    /// the resulting GetMaxNumItemsPerChunk to set their buffer sizes automatically.
    /// Note that the user must SignalCompletion after the rendering command if
    /// the buffer can be freed afterwards (or re-upload of data will be performed)
    /// See NG render mode of SphereRenderer for a usage example.
class MEGAMOLCORE_API SSBOBufferArray {
    public:

         SSBOBufferArray(const std::string& debugLabel = std::string());
        ~SSBOBufferArray();
         void upload(const std::function<void(void *, const void *)> &copyOp);

        /// this is for data that by definition will fit in a single block of GL_MAX_SHADER_STORAGE_BLOCK_SIZE.
        /// if it does not, an assertion will happen!
        /// @param data the pointer to the original data
        /// @param srcStride the size of a single data item in the original data
        /// @param dstStride the size of a single data item that will be uploaded
        ///                   and must not be split across buffers
        /// @param numItems the length of the original data in multiples of stride
        /// @param maxBufferSize the size of a ring buffer in bytes
        /// @param copyOp (optional) copyOp to transform src into dst (per item, gets correctly offset pointers (dst,
        /// src))
        /// @returns number of chunks
        void SetData(const void* data, GLuint srcStride, GLuint dstStride, size_t numItems,
            const std::function<void(void*, const void*)>& copyOp = nullptr);

        /// this is for defining the max number of items that fit in a desired chunk size,
        /// i.e. for the largest data stream, the 'master'
        /// @param data the pointer to the original data
        /// @param srcStride the size of a single data item in the original data
        /// @param dstStride the size of a single data item that will be uploaded
        ///                   and must not be split across buffers
        /// @param numItems the length of the original data in multiples of stride
        /// @param maxBufferSize the size of a ring buffer in bytes
        /// @param copyOp (optional) copyOp to transform src into dst (per item, gets correctly offset pointers (dst,
        /// src))
        /// @returns number of chunks
        GLuint SetDataWithSize(const void* data, GLuint srcStride, GLuint dstStride, size_t numItems, GLuint maxBufferSize,
            const std::function<void(void*, const void*)>& copyOp = nullptr);

        /// this is for the smaller data streams that need be aligned with numChunks of the
        /// 'master' stream.
        /// @param data the pointer to the original data
        /// @param srcStride the size of a single data item in the original data
        /// @param dstStride the size of a single data item that will be uploaded
        ///                   and must not be split across buffers
        /// @param numItems the length of the original data in multiples of stride
        /// @param numItemsPerChunk number of items per chunk in the master buffer
        /// @param copyOp (optional) copyOp to transform src into dst (per item, gets correctly offset pointers (dst,
        /// src))
        /// @returns the size of a ring buffer in bytes
        GLuint SetDataWithItems(const void* data, GLuint srcStride, GLuint dstStride, size_t numItems,
            GLuint numItemsPerChunk, const std::function<void(void*, const void*)>& copyOp = nullptr);

        /// @returns the GL object of the SSBO corresponding to chunk idx
        GLuint GetHandle(unsigned int idx) const {
            if (idx >= 0 && idx < this->theSSBOs.size()) {
                return this->theSSBOs[idx];
            } else {
                return 0;
            }
        }

        GLuint GetNumItems(unsigned int idx) const {
            if (idx >= 0 && idx < this->actualItemsPerChunk.size()) {
                return this->actualItemsPerChunk[idx];
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

        /// @returns how much stuff you need to upload to the GPU (in bytes)
        GLuint GetUsedBufferSize(void) const { return numItemsPerChunk * dstStride; }

    private:

        std::vector<GLuint> theSSBOs;
        std::vector<GLuint> actualItemsPerChunk;
        /// in bytes!
        GLuint maxBufferSize;
        GLuint numBuffers;
        GLuint srcStride;
        GLuint dstStride;
        const void* theData;
        size_t numItems;
        GLuint numChunks;
        GLuint numItemsPerChunk;
        std::string debugLabel;
        GLint64 maxSSBOSize = 0;
    };

} /* end namespace utility */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_SSBOBUFFERARRAY_H_INCLUDED */