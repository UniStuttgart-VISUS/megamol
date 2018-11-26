//
// CUDAStreamlines.h
//
//  Created on: Nov 6, 2013
//      Author: scharnkn
//

#ifndef MMPROTEINCUDAPLUGIN_CUDASTREAMLINES_H_INCLUDED
#define MMPROTEINCUDAPLUGIN_CUDASTREAMLINES_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "stdafx.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "CudaDevArr.h"

namespace megamol {
namespace protein_cuda {

class CUDAStreamlines {

public:

    // Offset in output VBO for positions
    static const int vboOffsPos;

    // Offset in output VBO for color (RGBA)
    static const int vboOffsCol;

    // VBO vertex data stride
    static const int vboStride;

    /// Enum describing possible integration directions
    enum Direction {FORWARD=1, BACKWARD, BIDIRECTIONAL};

    /** CTor */
    CUDAStreamlines();

    /** DTor */
    ~CUDAStreamlines();

    /**
     * Initializes all the necessary parameters for the streamline integration
     * and creates the line strip VBO.
     *
     * @param nSegments The number of line segments per streamline
     * @param nStreamlines  The number of streamlines
     * @param dir The direction of the integration
     * @return 'True' on success, 'false' otherwise
     */
    bool InitStreamlines(int nSegments, int nStreamlines, Direction dir);

    /**
     * Execute streamline integration starting at the given seedpoint array
     * using fourth order Runge-Kutta integration.
     *
     * @param seedPoints    The starting point of the integration
     * @param step          The step size of the integration
     * @param vecField      The vector field to be integrated
     * @param vecFieldDim   The dimensions of the lattice
     * @param vecFieldOrg   The WS origin of the lattice
     * @param vecFieldDelta The spacing of the lattice
     * @return 'True' on success, 'false' otherwise
     */
    bool IntegrateRK4(
            const float *seedPoints,
            float step,
            float *vecField,
            int3 vecFieldDim,
            float3 vecFieldOrg,
            float3 vecFieldDelta);

    /**
     * Render the streamlines using GL_Line_Strip.
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool RenderLineStrip();

    /**
     * Render the streamlines using GL_Line_Strip, while setting the color
     * pointer.
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool RenderLineStripWithColor();

    /**
     * Samples a given scalar field and stores the value in the alpha component
     * of the RGBA color value in the VBO.
     *
     * @param field      The scalar field to be sampled
     * @param fieldDim   The dimensions of the lattice
     * @param fieldOrg   The WS origin of the lattice
     * @param fieldDelta The spacing of the lattice
     * @return 'True' on success, 'false' otherwise
     */
    bool SampleScalarFieldToAlpha(
            float *field,
            int3 fieldDim,
            float3 fieldOrg,
            float3 fieldDelta);

    /**
     * Samples a given vector field and stores the value in the RGB components
     * of the RGBA color value in the VBO.
     *
     * @param vecField   The vector field to be sampled
     * @param fieldDim   The dimensions of the lattice
     * @param fieldOrg   The WS origin of the lattice
     * @param fieldDelta The spacing of the lattice
     * @return 'True' on success, 'false' otherwise
     */
    bool SampleVecFieldToRGB(
            float *vecField,
            int3 fieldDim,
            float3 fieldOrg,
            float3 fieldDelta);

    /**
     * Initializes the RGB part of the color with a uniform value
     *
     * @param col The uniform color
     * @return 'True' on success, 'false' otherwise
     */
    bool SetUniformRGBColor(float3 col);

protected:

    /**
     * Unregisters the cuda resource handle with the vertex buffer object and
     * destroys the vertex buffer object.
     */
    void destroyVBO();

    /**
     * Initializes the vertex buffer object holding the line strip and registers
     * the cuda resource with it.
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool initVBO();

private:

    /// Cuda graphics resource associated with the vertex data VBO
    struct cudaGraphicsResource *cudaToken;

    /// The OpenGL handle for the vertex buffer object holding the line strip
    GLuint lineStripVBO;

    /// The maximum number of line segments
    int nSegments;

    /// The number of streamlines
    int nStreamlines;

    /// The direction of the integration
    Direction dir;

    // Temporary device array for a vector field
    CudaDevArr<float> vecField_D;

    // Temporary device array for a scalar field
    CudaDevArr<float> sclField_D;

};

} // end namespace protein_cuda
} // end namespace megamol

#endif // MMPROTEINCUDAPLUGIN_CUDASTREAMLINES_H_INCLUDED
