//
// AbstractGPUSurface.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 17, 2013
// Author     : scharnkn
//

#ifndef MMPROTEINCUDAPLUGIN_ABSTRACTGPUSURFACE_H_INCLUDED
#define MMPROTEINCUDAPLUGIN_ABSTRACTGPUSURFACE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "stdafx.h"
#include "vislib/graphics/gl/IncludeAllGL.h"

namespace megamol {
namespace protein_cuda {


/*
 * Class representing a triangle mesh stored solely on the GPU (accessible
 * through VBOs)
 */
class AbstractGPUSurface {

public:

    /// Vertex data buffer offset for positions
    static const size_t vertexDataOffsPos;

    /// Vertex data buffer offset for normals
    static const size_t vertexDataOffsNormal;

    /// Vertex data buffer offset for tex coords
    static const size_t vertexDataOffsTexCoord;

    /// Vertex data buffer element size
    static const size_t vertexDataStride;

    /** DTor */
    AbstractGPUSurface();

    /**
     * A copy constructor that makes a deep copy of the surface object.
     *
     * @param other The object to be copied
     */
    AbstractGPUSurface(const AbstractGPUSurface& other);

    /** CTor */
    virtual ~AbstractGPUSurface() = 0;

    /**
     * Initializes the vertex buffer objects based on the given vertex and
     * triangle count. Sets the 'ready' flag to true.
     *
     * @param triangleCnt The number of triangles.
     */
    bool InitTriangleIdxVBO(size_t triangleCnt);

    /**
     * Initializes the vertex buffer objects based on the given vertex and
     * triangle count. Sets the 'ready' flag to true.
     *
     * @param vertexCnt   The number of vertices.
     */
    bool InitVertexDataVBO(size_t vertexCnt);

    /**
     * Initializes the vertex buffer objects based on the given vertex and
     * triangle count. Sets the 'ready' flag to true.
     *
     * @param triangleCnt The number of triangles.
     */
    bool InitTriangleIdxVBO(size_t triangleCnt, GLuint &vbo);

    /**
     * Initializes the vertex buffer objects based on the given vertex and
     * triangle count. Sets the 'ready' flag to true.
     *
     * @param vertexCnt   The number of vertices.
     */
    bool InitVertexDataVBO(size_t vertexCnt, GLuint &vbo);

    /**
     * Answers the number of triangles.
     *
     * @return the number of triangles.
     */
    size_t GetTriangleCnt() const {
        return this->triangleCnt;
    }

    /**
     * Answers the GPU handle for the VBO with the triangle indices. Needs
     * the 'ready flag to be true.
     *
     * @return The GPU handle for the vertex buffer object or NULL if !ready
     */
    GLuint GetTriangleIdxVBO() const {
        if (this->triangleIdxReady) {
            return this->vboTriangleIdx;
        } else {
            return 0;
        }
    }

    /**
     * Answersd the number of vertices
     *
     * @return The number of vertices.
     */
    size_t GetVertexCnt()  const {
        return this->vertexCnt;
    }

    /**
     * Answers the GPU handle for the VBO with the vertex data. Needs
     * the 'ready flag to be true.
     *
     * @return The GPU handle for the vertex buffer object or NULL if !ready
     */
    GLuint GetVtxDataVBO() const {
        if (this->vertexDataReady) {
            return this->vboVtxData;
        } else {
            return 0;
        }
    }

    /**
     * Initializes all extensions necessary for the class to work properly.
     *
     * @return 'True' if all extensions could be initialized, 'false' otherwise.
     */
    static bool InitExtensions() {
        // Init extensions
        if (!isExtAvailable("GL_ARB_copy_buffer")
            || !isExtAvailable("GL_ARB_vertex_buffer_object")) {
            return false;
        }
        return true;
    }

    /**
     * Assignment operator (makes deep copy).
     *
     * @param rhs The assigned surface object
     * @return The returned surface object
     */
    AbstractGPUSurface& operator=(const AbstractGPUSurface &rhs);

    /**
     * Deallocate all allocated memory.
     */
    void Release();

protected:

    /// Flag that tells whether the vertex data VBO has been initialized
    bool vertexDataReady;

    /// Flag that tells whether the triangle index VBO has been initialized
    bool triangleIdxReady;

    /// Vertex Buffer Object handle for vertex data
    GLuint vboVtxData;

    /// Vertex Buffer Object handle for triangle indices
    GLuint vboTriangleIdx;

    /// The number of vertices
    size_t vertexCnt;

    /// The number of triangles
    size_t triangleCnt;

private:

};

} // namespace protein_cuda
} // namespace megamol

#endif // MMPROTEINCUDAPLUGIN_ABSTRACTGPUSURFACE_H_INCLUDED
