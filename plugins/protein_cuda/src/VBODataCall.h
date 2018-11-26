//
// VBODataCall.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: May 31, 2013
//     Author: scharnkn
//

#ifndef MMPROTEINCUDAPLUGIN_VBODATACALL_H_INCLUDED
#define MMPROTEINCUDAPLUGIN_VBODATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/math/Cuboid.h"
#include "mmcore/view/CallRender3D.h"
#include "vislib/graphics/gl/IncludeAllGL.h"

namespace megamol {
namespace protein_cuda {

class VBODataCall : public core::Call {

public:

    /// Index of the 'GetCamparams' function
    static const unsigned int CallForGetExtent;

    /// Index of the 'SetCamparams' function
    static const unsigned int CallForGetData;

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char *ClassName(void) {
        return "VBODataCall";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char *Description(void) {
        return "Call to transmit one vertex buffer object handle and the according bounding box";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) {
        return 2;
    }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char * FunctionName(unsigned int idx) {
        switch( idx) {
        case 0:
            return "getExtent";
        case 1:
            return "getData";
        }
        return NULL;
    }

    /**
     * Answers the bounding box
     *
     * @return The bounding box
     */
    core::BoundingBoxes &GetBBox() {
        return this->bbox;
    }

    /**
     * Answers the data buffer's offset for positions
     *
     * @return The data buffer's position offset
     */
    inline unsigned int GetDataOffsPosition() {
        return this->dataOffsPosition;
    }

    /**
     * Answers the data buffer's offset for normals
     *
     * @return The data buffer's normal offset
     */
    inline unsigned int GetDataOffsNormal() {
        return this->dataOffsNormal;
    }

    /**
     * Answers the data buffer's offset for tex coords
     *
     * @return The data buffer's tex coord offset
     */
    inline unsigned int GetDataOffsTexCoord() {
        return this->dataOffsTexCoord;
    }

    /**
     * Answers the data buffer's stride
     *
     * @return The data buffer's stride
     */
    inline unsigned int GetDataStride() {
        return this->dataStride;
    }

    /**
     * Answers the data's frame cnt
     *
     * @return The data's frame cnt
     */
    inline unsigned int GetFrameCnt() {
        return this->frameCnt;
    }

    /**
     * Answers the texture id
     *
     * @return The texture id
     */
    inline GLuint GetTexId() {
        return this->tex;
    }

    /**
     * Answers the texture maximum value
     *
     * @return The texture maximum value
     */
    inline float GetTexValMax() {
        return this->texValMax;
    }

    /**
     * Answers the texture minimum value
     *
     * @return The texture minimum value
     */
    inline float GetTexValMin() {
        return this->texValMin;
    }

    /**
     * Answers the vbo handle
     *
     * @return The VBO handle
     */
    inline GLuint GetVbo() {
        return this->vbo;
    }

    /**
     * Answers the cuda device handle (if any)
     *
     * @return The VBO handle
     */
    inline struct cudaGraphicsResource **GetCudaRessourceHandle() {
        return this->vboHandle;
    }

    /**
     * Answers the vbo triangle index handle
     *
     * @return The vbo triangle index handle
     */
    inline GLuint GetVboTriangleIdx() {
        return this->vboTriangleIdx;
    }

    /**
     * Answers the triangle count.
     *
     * @return The number of triangles
     */
    inline unsigned int GetTriangleCnt() {
        return this->triangleCnt;
    }

    /**
     * Answers the vertex count.
     *
     * @return The number of vertices
     */
    inline unsigned int GetVertexCnt() {
        return this->vertexCnt;
    }

    /**
     * Sets the data sets bounding box
     *
     * @param bbox the bounding box
     */
    inline void SetBBox(core::BoundingBoxes bbox) {
        this->bbox = bbox;
    }

    /**
     * Sets the data buffer's offsets for position, normals and tex coords.
     * -1 indicates invalid values.
     *
     * @param dataStride The data buffer's stride
     */
    inline void SetDataOffs(int offsPos, int offsNormal, int offsTexCoord) {
        this->dataOffsPosition = offsPos;
        this->dataOffsNormal = offsNormal;
        this->dataOffsTexCoord = offsTexCoord;
    }

    /**
     * Sets the data buffer's stride value.
     *
     * @param dataStride The data buffer's stride
     */
    inline void SetDataStride(int dataStride) {
        this->dataStride = dataStride;
    }

    /**
     * Sets the number of frames
     *
     * @param frameCnt The data's frame count
     */
    inline void SetFrameCnt(unsigned int frameCnt ) {
        this->frameCnt = frameCnt;
    }

    /**
     * Sets the texture handle.
     *
     * @param tex The texture handle
     */
    inline void SetTex(GLuint tex) {
        this->tex = tex;
    }

    /**
     * Sets the texture minimum and maximum value
     *
     * @param min The minimum value
     * @param max The maximum value
     */
    inline void SetTexValRange(float min, float max) {
        this->texValMin = min;
        this->texValMax = max;
    }

    /**
     * Sets the VBO handle.
     *
     * @param vbo The vbo handle
     */
    inline void SetVbo(GLuint vbo) {
        this->vbo = vbo;
    }

    /**
     * Sets the VBO device pointer
     *
     * @param vbo The vbo handle
     */
    inline void SetCudaRessourceHandle(struct cudaGraphicsResource **vboHandle) {
        this->vboHandle = vboHandle;
    }

    /**
     * Sets the VBO triangle index handle.
     *
     * @param vbo The vbo triangle index handle
     */
    inline void SetVboTriangleIdx(GLuint vboTriangleIdx) {
        this->vboTriangleIdx = vboTriangleIdx;
    }

    /**
     * Sets the vertex count
     *
     * @param vertexCnt The vertex count
     */
    inline void SetVertexCnt(unsigned int vertexCnt) {
        this->vertexCnt = vertexCnt;
    }

    /**
     * Sets the triangle count
     *
     * @param triangleCnt The triangle count
     */
    inline void SetTriangleCnt(unsigned int triangleCnt) {
        this->triangleCnt = triangleCnt;
    }

    /** Ctor. */
    VBODataCall(void);

    /** Dtor. */
    virtual ~VBODataCall(void);


protected:


private:

    /// The OpenGL VBO handle for vertex data
    GLuint vbo;

    /// The number of vertices
    unsigned int vertexCnt;

    /// The OpenGL VBO handle for triangle indices
    GLuint vboTriangleIdx;

    /// The number of triangles
    unsigned int triangleCnt;

    /// The OpenGL texture handle
    GLuint tex;

    /// The bounding box
    core::BoundingBoxes bbox;

    /// The vertex data stride
    int dataStride;

    /// The vertex data offset for positions
    int dataOffsPosition;

    /// The vertex data offset for normals
    int dataOffsNormal;

    /// The vertex data offset for tex coords
    int dataOffsTexCoord;

    /// The number of frames
    unsigned int frameCnt;

    /// Maximum texture value
    float texValMax;

    /// Minimum texture value
    float texValMin;

    /// VBO device pointer
    struct cudaGraphicsResource **vboHandle;

};

/// Description class typedef
typedef core::factories::CallAutoDescription<VBODataCall> VBODataCallDescription;


} // end namespace protein_cuda
} // end namespace megamol

#endif // MMPROTEINCUDAPLUGIN_VBODATACALL_H_INCLUDED
