//
// DeformableGPUSurfaceMTCall.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 18, 2013
// Author     : scharnkn
//

#ifndef MMPROTEINCUDAPLUGIN_DEFORMABLEGPUSURFACEMTCALL_H_INCLUDED
#define MMPROTEINCUDAPLUGIN_DEFORMABLEGPUSURFACEMTCALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/math/Cuboid.h"
#include "mmcore/view/CallRender3D.h"
#include <GL/gl.h>
#include "AbstractGPUSurfaceCall.h"

namespace megamol {
namespace protein_cuda {

class DeformableGPUSurfaceMTCall : public core::Call {

public:

    /// Index of the 'GetCamparams' function
    static const unsigned int CallForGetExtent;

    /// Index of the 'SetCamparams' function
    static const unsigned int CallForGetData;

    /** Ctor. */
    DeformableGPUSurfaceMTCall(void) : AbstractGPUSurfaceCall() {}

    /** Dtor. */
    virtual ~DeformableGPUSurfaceMTCall(void) {
    }

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char *ClassName(void) {
        return "AbstractGPUSurfaceCall";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char *Description(void) {
        return "Call to transmit a deformable surface VBO with positions,\
                normals, and texture coordinates.";
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
        return "";
    }

    /**
     * Answers the data buffer's offset for positions
     *
     * @return The data buffer's position offset
     */
    inline unsigned int GetDataOffsMappedPosition() {
        return this->surface->vertexDataOffsMappedPos;
    }

    /**
     * Answers the data buffer's offset for normals
     *
     * @return The data buffer's normal offset
     */
    inline unsigned int GetDataOffsMappedNormal() {
        return this->surface->vertexDataOffsMappedNormal;
    }

    /**
     * Answers the data buffer's offset for tex coords
     *
     * @return The data buffer's tex coord offset
     */
    inline unsigned int GetDataOffsMappedTexCoord() {
        return this->surface->vertexDataOffsMappedTexCoord;
    }

    /**
     * Answers the data buffer's stride
     *
     * @return The data buffer's stride
     */
    inline unsigned int GetMappedDataMStride() {
        return this->surface->vertexDataMappedStride;
    }

    /**
     * Answers the handle for the mapped vertex data buffer
     *
     * @return The VBO handle
     */
    inline GLuint GetMappedVertexVbo() {
        return this->surface->GetMappedVtxDataVBO()();
    }

    /**
     * Sets the surface of this call.
     */
    inline void SetSurface(const DeformableGPUSurfaceMT *surface) {
        this->surface = surface;
    }

protected:

private:

    /// The surface
    const DeformableGPUSurfaceMT *surface;

};

/// Description class typedef
typedef core::factories::CallAutoDescription<DeformableGPUSurfaceMTCall> DeformableGPUSurfaceMTCallDescription;


} // end namespace protein_cuda
} // end namespace megamol


#endif // MMPROTEINCUDAPLUGIN_DEFORMABLEGPUSURFACEMTCALL_H_INCLUDED
