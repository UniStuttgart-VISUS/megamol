//
// AbstractGPUSurface.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 17, 2013
// Author     : scharnkn
//

#include "stdafx.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "AbstractGPUSurface.h"
#include "ogl_error_check.h"

using namespace megamol;
using namespace megamol::protein_cuda;


// Offsets and stride for vbos holding surface data
const size_t AbstractGPUSurface::vertexDataOffsPos = 0;
const size_t AbstractGPUSurface::vertexDataOffsNormal = 3;
const size_t AbstractGPUSurface::vertexDataOffsTexCoord = 6;
const size_t AbstractGPUSurface::vertexDataStride = 9;


/*
 * AbstractGPUSurface::AbstractGPUSurface
 */
AbstractGPUSurface::AbstractGPUSurface() : vertexDataReady(false),
        triangleIdxReady(false), vboVtxData(0), vboTriangleIdx(0),
        vertexCnt(0), triangleCnt(0) {
}


/*
 * AbstractGPUSurface::AbstractGPUSurface
 */
AbstractGPUSurface::AbstractGPUSurface(const AbstractGPUSurface& other) {

    /* Make deep copy of triangle index buffer */

    if (other.triangleIdxReady) {

        this->triangleCnt = other.triangleCnt;

        // Destroy if necessary
        if (this->triangleIdxReady) {
            glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, this->vboTriangleIdx);
            glDeleteBuffersARB(1, &this->vboTriangleIdx);
            glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, 0);
            this->vboTriangleIdx = 0;
        }

        // Create vertex buffer object for triangle indices
        glGenBuffersARB(1, &this->vboTriangleIdx);

        CheckForGLError();

        // Map as copy buffer
        glBindBufferARB(GL_COPY_READ_BUFFER, other.vboTriangleIdx);
        glBindBufferARB(GL_COPY_WRITE_BUFFER, this->vboTriangleIdx);
        glBufferDataARB(GL_COPY_WRITE_BUFFER,
                sizeof(unsigned int)*this->triangleCnt*3, 0, GL_DYNAMIC_DRAW);
        // Copy data
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                sizeof(unsigned int)*this->triangleCnt*3);
        glBindBufferARB(GL_COPY_WRITE_BUFFER, 0);
        glBindBufferARB(GL_COPY_READ_BUFFER, 0);

        this->triangleIdxReady = other.triangleIdxReady;

        CheckForGLError();
    }

    /* Make deep copy of vertex data buffer */

    if (other.vertexDataReady) {

        this->vertexCnt = other.vertexCnt;

        // Destroy if necessary
        if (this->vertexDataReady) {
            glBindBufferARB(GL_ARRAY_BUFFER, this->vboVtxData);
            glDeleteBuffersARB(1, &this->vboVtxData);
            this->vboVtxData = 0;
            glBindBufferARB(GL_ARRAY_BUFFER, 0);
        }

        // Create vertex buffer object for vertex data
        glGenBuffersARB(1, &this->vboVtxData);

        CheckForGLError();

        //    // Map as copy buffer
        glBindBufferARB(GL_COPY_READ_BUFFER, other.vboVtxData);
        glBindBufferARB(GL_COPY_WRITE_BUFFER, this->vboVtxData);
        glBufferDataARB(GL_COPY_WRITE_BUFFER,
                this->vertexCnt*this->vertexDataStride*sizeof(float), 0, GL_DYNAMIC_DRAW);
        // Copy data
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                this->vertexCnt*this->vertexDataStride*sizeof(float));
        CheckForGLError();
        glBindBufferARB(GL_COPY_WRITE_BUFFER, 0);
        glBindBufferARB(GL_COPY_READ_BUFFER, 0);


        this->vertexDataReady = other.vertexDataReady;

        CheckForGLError();
    }

}


/*
 * AbstractGPUSurface::~AbstractGPUSurface
 */
AbstractGPUSurface::~AbstractGPUSurface() {
}


/*
 * AbstractGPUSurface::InitTriangleIdxVBO
 */
bool AbstractGPUSurface::InitTriangleIdxVBO(size_t triangleCnt) {

    // Destroy if necessary
    if (this->triangleIdxReady) {
        glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, this->vboTriangleIdx);
        glDeleteBuffersARB(1, &this->vboTriangleIdx);
        glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, 0);
        this->vboTriangleIdx = 0;
    }

    // Create vertex buffer object for triangle indices
    glGenBuffersARB(1, &this->vboTriangleIdx);
    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, this->vboTriangleIdx);
    glBufferDataARB(GL_ELEMENT_ARRAY_BUFFER,
            sizeof(unsigned int)*triangleCnt*3, 0, GL_DYNAMIC_DRAW);
    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, 0);

//    printf("InitTriangleIdxVBO: %u bytes\n", sizeof(unsigned int)*triangleCnt*3);

    this->triangleCnt = triangleCnt;
    this->triangleIdxReady = true;

    return CheckForGLError();
}



/*
 * AbstractGPUSurface::InitVertexDataVBO
 */
bool AbstractGPUSurface::InitVertexDataVBO(size_t vertexCnt) {

    // Destroy if necessary
    if (this->vertexDataReady) {
        glBindBufferARB(GL_ARRAY_BUFFER, this->vboVtxData);
        glDeleteBuffersARB(1, &this->vboVtxData);
        glBindBufferARB(GL_ARRAY_BUFFER, 0);
        this->vboVtxData = 0;
    }

    // Create vertex buffer object for vertex data
    glGenBuffersARB(1, &this->vboVtxData);
    glBindBufferARB(GL_ARRAY_BUFFER, this->vboVtxData);
    glBufferDataARB(GL_ARRAY_BUFFER,
            vertexCnt*this->vertexDataStride*sizeof(float), 0, GL_DYNAMIC_DRAW);
    glBindBufferARB(GL_ARRAY_BUFFER, 0);

//    printf("InitVertexDataVBO: %u bytes\n", vertexCnt*this->vertexDataStride*sizeof(float));

    this->vertexCnt = vertexCnt;
    this->vertexDataReady = true;

    return CheckForGLError();
}


/*
 * AbstractGPUSurface::AbstractGPUSurface
 */
AbstractGPUSurface& AbstractGPUSurface::operator=(const AbstractGPUSurface &rhs) {

    /* Make deep copy of triangle index buffer */

    if (rhs.triangleIdxReady) {

        this->triangleCnt = rhs.triangleCnt;

        // Destroy if necessary
        if (this->triangleIdxReady) {
            glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, this->vboTriangleIdx);
            glDeleteBuffersARB(1, &this->vboTriangleIdx);
            glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, 0);
            this->vboTriangleIdx = 0;
        }

        // Create vertex buffer object for triangle indices
        glGenBuffersARB(1, &this->vboTriangleIdx);

        CheckForGLError();

        // Map as copy buffer
        glBindBufferARB(GL_COPY_READ_BUFFER, rhs.vboTriangleIdx);
        glBindBufferARB(GL_COPY_WRITE_BUFFER, this->vboTriangleIdx);
        glBufferDataARB(GL_COPY_WRITE_BUFFER,
                sizeof(unsigned int)*this->triangleCnt*3, 0, GL_DYNAMIC_DRAW);
        // Copy data
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                sizeof(unsigned int)*this->triangleCnt*3);
        glBindBufferARB(GL_COPY_WRITE_BUFFER, 0);
        glBindBufferARB(GL_COPY_READ_BUFFER, 0);

        this->triangleIdxReady = rhs.triangleIdxReady;

        CheckForGLError();
    }

    /* Make deep copy of vertex data buffer */

    if (rhs.vertexDataReady) {

        this->vertexCnt = rhs.vertexCnt;

        // Destroy if necessary
        if (this->vertexDataReady) {
            glBindBufferARB(GL_ARRAY_BUFFER, this->vboVtxData);
            glDeleteBuffersARB(1, &this->vboVtxData);
            this->vboVtxData = 0;
            glBindBufferARB(GL_ARRAY_BUFFER, 0);
        }

        // Create vertex buffer object for vertex data
        glGenBuffersARB(1, &this->vboVtxData);

        CheckForGLError();

        //    // Map as copy buffer
        glBindBufferARB(GL_COPY_READ_BUFFER, rhs.vboVtxData);
        glBindBufferARB(GL_COPY_WRITE_BUFFER, this->vboVtxData);
        glBufferDataARB(GL_COPY_WRITE_BUFFER,
                this->vertexCnt*this->vertexDataStride*sizeof(float), 0, GL_DYNAMIC_DRAW);
        // Copy data
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                this->vertexCnt*this->vertexDataStride*sizeof(float));
        CheckForGLError();
        glBindBufferARB(GL_COPY_WRITE_BUFFER, 0);
        glBindBufferARB(GL_COPY_READ_BUFFER, 0);


        this->vertexDataReady = rhs.vertexDataReady;

        CheckForGLError();
    }

    return *this;

}


/*
 * AbstractGPUSurface::Release
 */
void AbstractGPUSurface::Release() {

    if (this->vboVtxData) {
        glBindBufferARB(GL_ARRAY_BUFFER, this->vboVtxData);
        glDeleteBuffersARB(1, &this->vboVtxData);
        this->vboVtxData = 0;
        CheckForGLError();
        glBindBufferARB(GL_ARRAY_BUFFER, 0);
    }
    if (this->vboTriangleIdx) {
        glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, this->vboTriangleIdx);
        glDeleteBuffersARB(1, &this->vboTriangleIdx);
        this->vboTriangleIdx = 0;
        CheckForGLError();
        glBindBufferARB(GL_ARRAY_BUFFER, 0);
    }

    this->vertexDataReady = false;
    this->triangleIdxReady = false;
    this->vertexCnt = 0;
    this->triangleCnt = 0;
}
