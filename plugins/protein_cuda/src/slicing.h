/*
 * Copyright (c) 2008  Martin Falk <falk@visus.uni-stuttgart.de>
 *                     Visualization Research Center (VISUS), 
 *                     Universität Stuttgart, Germany
 *
 * This source code is distributed as part of the publication 
 * "Output-Sensitive 3D Line Integral Convolution". 
 * Sample images and movies of this application can be found 
 * at http://www.vis.uni-stuttgart.de/texflowvis . 
 * This file may be distributed, modified, and used free of charge 
 * as long as this copyright notice is included in its original 
 * form. Commercial use is strictly prohibited. However we request
 * that acknowledgement is given to the following article
 *
 *     M. Falk, D. Weiskopf.
 *     Output-Sensitive 3D Line Integral Convolution,
 *     IEEE Transactions on Visualization and Computer Graphics, 2008.
 *
 * Filename: slicing.h
 * 
 * $Id: slicing.h 3 2009-07-03 11:19:06Z falkmn $ 
 */
#ifndef _SLICING_H_
#define _SLICING_H_

#ifdef _WIN32
#  include <windows.h>
#endif

#include "vislib/graphics/gl/IncludeAllGL.h"
//#include "mmath.h"


#define VS_EPS      1.0e-8

#define VS_FACE_FRONT    1
#define VS_FACE_BACK     2
#define VS_FACE_RIGHT    4
#define VS_FACE_LEFT     8
#define VS_FACE_BOTTOM  16
#define VS_FACE_TOP     32

// class for view-aligned slicing of a volume
class ViewSlicing
{
public: 
    ViewSlicing(void) {};
    ~ViewSlicing(void) {};

    // setup slicing
    // needs the model view matrix (column oriented),
    // the sampling distance, and the extends of the volume
    // returns total number of slices
    int setupSlicing(float *mvMatrix, float sampDist, float *extents);
    void drawSlice(int slice);
    void drawSlices(GLenum mode, int frontToBack, int maxSlices);

    int getNumSlices(void) { return _numSlices; }

    void setupSingleSlice(double *viewVec, float *ext);
    void drawSingleSlice(float dist);

protected:
    // compares two points, if equal return true, false otherwise
    inline bool pointCmp(float *p1, float *p2, float eps);

private:

    static const char cubeEdges[12];

    // store the model view matrix (column oriented)
    float _m[16];
    // view direction
    float _v[3];
    // extent of the volume
    float _ext[3];
    // sampling distance between two slices
    double _sampDist;
    // max distance from the origin to the volume
    float _d;
    // coordinates of all 12 possible vertices
    float _p[12][3];

    int _numSlices;
    int _maxSlices;
};

#endif // _SLICING_H_
