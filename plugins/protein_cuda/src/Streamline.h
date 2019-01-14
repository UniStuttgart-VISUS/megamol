//
// Streamline.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//

#ifndef MMPROTEINCUDAPLUGIN_STREAMLINE_H_INCLUDED
#define MMPROTEINCUDAPLUGIN_STREAMLINE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/math/Vector.h"
#include "vislib/Array.h"
#include "VecField3f.h"
//#include "vislib_vector_typedefs.h"

namespace megamol {
namespace protein_cuda {

class Streamline {
public:

    /// Enum describing possible integration directions
    enum Direction {FORWARD=1, BACKWARD, BIDIRECTIONAL};

    /** CTor */
    Streamline() {
    }

    /** DTor */
    ~Streamline() {
    }

    /**
     * Answer the starting point position of the streamline.
     * Note: this is not the seedpoint
     *
     * @return The position of the start point
     */
    Vec3f GetStartPos() {
        return this->startPos;
    }

    /**
     * Answer the ending point position of the streamline.
     *
     * @return The position of the start point
     */
    Vec3f GetEndPos() {
        return this->endPos;
    }

    /**
     * Answer the (current) length of the streamline.
     *
     * @return The length of the streamline.
     */
    SIZE_T GetLength() {
        return this->vertexArr.Count()/3;
    }

    /**
     * Execute streamline integration starting at 'start' using fourth
     * order Runge-Kutta integration. Arrays are cleaned before integrating.
     * 3D texture coordinates are computed based on the grid parameters of the
     * vector field ranging from 0 .. 1.
     *
     * @param start     The starting point of the integration
     * @param v         The vector field to be integrated
     * @param maxLength The maximum length of the integration
     * @param step      The step size of the integration
     * @param eps       Value at which the field is defined to be 'vanishing'
     * @param dir       The direction of the integration
     */
    void IntegrateRK4(Vec3f start, VecField3f &v, unsigned int maxLength,
            float step, float eps, Direction dir);

    /**
     * Returns a pointer to the array containing the vertex positions.
     *
     * @return A pointer to the array containing the vertex positions
     */
    const float *PeekVertexArr() {
        return this->vertexArr.PeekElements();
    }

    /**
     * Returns a pointer to the array containing the vertex tangents.
     *
     * @return A pointer to the array containing the vertex tangents
     */
    const float *PeekTangentArr() {
        return this->tangentArr.PeekElements();
    }

    /**
     * Returns a pointer to the array containing the texture coordinates.
     *
     * @return A pointer to the array containing the texture coordinates
     */
    const float *PeekTexCoordArr() {
        return this->texCoordArr.PeekElements();
    }

    /**
     * Test for equality
     *
     * @param rhs The right hand side operand
     *
     * @return True if this and rhs are equal
     */
    bool operator==(const Streamline& rhs) const {
        return (this->startPos == rhs.startPos)
            && (this->endPos == rhs.endPos);
    }

private:

    /// Array containing the vertex positions
    vislib::Array<float> vertexArr;

    /// Array containing the vertex tangents
    vislib::Array<float> tangentArr;

    /// Array containing the texture coordinates
    vislib::Array<float> texCoordArr;

    /// Starting and ending positions of the streamline
    Vec3f startPos, endPos;

};

} // end namespace protein_cuda
} // end namespace megamol

#endif /* MMPROTEINCUDAPLUGIN_STREAMLINE_H_INCLUDED */
