/*
 * Copyright (c) 2008  Institute for Visualization and Interactive
 * Systems, University of Stuttgart, Germany
 *
 * This source code is distributed as part of the output sensitive 
 * 3D LIC project. Details about this project can be found on the 
 * project web page at http://www.vis.uni-stuttgart.de/texflowvis . 
 * This file may be distributed, modified, and used free of charge 
 * as long as this copyright notice is included in its original 
 * form. Commercial use is strictly prohibited.
 *
 * Filename: slicing.cpp
 * 
 * $Id: slicing.cpp 24 2009-08-06 15:53:03Z falkmn $ 
 */

#include "stdafx.h"

#ifdef _WIN32
#  include <windows.h>
#endif

#define _USE_MATH_DEFINES 1

#include "vislib/graphics/gl/IncludeAllGL.h"

#include <math.h>
#include <assert.h>
#include <stdio.h>
#include "vislib/math/Vector.h"
#include "slicing.h"


const char ViewSlicing::cubeEdges[12] = 
{ VS_FACE_FRONT + VS_FACE_BOTTOM, VS_FACE_FRONT + VS_FACE_TOP, 
VS_FACE_BACK + VS_FACE_BOTTOM,  VS_FACE_BACK + VS_FACE_TOP,
VS_FACE_FRONT + VS_FACE_LEFT,   VS_FACE_FRONT + VS_FACE_RIGHT,
VS_FACE_BACK + VS_FACE_LEFT,    VS_FACE_BACK + VS_FACE_RIGHT,
VS_FACE_LEFT + VS_FACE_BOTTOM,  VS_FACE_RIGHT + VS_FACE_BOTTOM,
VS_FACE_LEFT + VS_FACE_TOP,     VS_FACE_RIGHT + VS_FACE_TOP };


int ViewSlicing::setupSlicing(float *mvMatrix, float sampDist, float *extents) {
    float invNorm;
    float xMax, yMax, zMax;
    float dv[7];
    int i;
    float v[3];

    _sampDist = sampDist;
    memcpy((void*)_m, (void*)mvMatrix, 16*sizeof(float));
    _ext[0] = xMax = 0.5f*extents[0];
    _ext[1] = yMax = 0.5f*extents[1];
    _ext[2] = zMax = 0.5f*extents[2];

    _p[0][1] =  -yMax;  _p[0][2] =  -zMax;
    _p[1][1] =   yMax;  _p[1][2] =  -zMax;
    _p[2][1] =  -yMax;  _p[2][2] =   zMax;
    _p[3][1] =   yMax;  _p[3][2] =   zMax;
    _p[4][0] =  -xMax;  _p[4][2] =  -zMax;
    _p[5][0] =   xMax;  _p[5][2] =  -zMax;
    _p[6][0] =  -xMax;  _p[6][2] =   zMax;
    _p[7][0] =   xMax;  _p[7][2] =   zMax;
    _p[8][0] =  -xMax;  _p[8][1]  = -yMax;
    _p[9][0] =   xMax;  _p[9][1]  = -yMax;
    _p[10][0] = -xMax;  _p[10][1] =  yMax;
    _p[11][0] =  xMax;  _p[11][1] =  yMax;

    // calculate the view vector
    invNorm = 1.0f / (_m[14] - _m[15]);
    _v[0] = (_m[2]  - _m[3]) * invNorm;
    _v[1] = (_m[6]  - _m[7]) * invNorm;
    _v[2] = (_m[10] - _m[11]) * invNorm;

    // normalize it
    invNorm = 1.0f / sqrt( (_v[0]*_v[0]) + (_v[1]*_v[1]) + (_v[2]*_v[2]));
    _v[0] *= invNorm;
    _v[1] *= invNorm;
    _v[2] *= invNorm;

    v[0] = fabs(_v[0]);
    v[1] = fabs(_v[1]);
    v[2] = fabs(_v[2]);

    dv[0] = v[0]*xMax;
    dv[1] = v[1]*yMax;
    dv[2] = v[2]*zMax;
    dv[3] = v[0]*xMax + v[1]*yMax;
    dv[4] = v[0]*xMax + v[2]*zMax;
    dv[5] = v[1]*yMax + v[2]*zMax;
    dv[6] = v[0]*xMax + v[1]*yMax + v[2]*zMax;

    // find max depth
    _d = dv[6];
    for (i=0; i<6; ++i)
        if (_d < dv[i])
            _d = dv[i];

    _d *= 2.0;

    _numSlices = (int) (_d / _sampDist) + 1;
    return _numSlices;
}


void ViewSlicing::drawSlice(int slice) {
    float xMax, yMax, zMax;
    char validHit[12] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    char edgeCode[12] = { cubeEdges[0], cubeEdges[1], cubeEdges[2], 
        cubeEdges[3], cubeEdges[4], cubeEdges[5], 
        cubeEdges[6], cubeEdges[7], cubeEdges[8], 
        cubeEdges[9], cubeEdges[10], cubeEdges[11] };
    float d = -0.5f*_d + (slice+0.5f)*_d/_numSlices;
    char actEdge = (char) 0xff;
    int numIntersect = 0, oldInt;
    int idx;
    int tmp[6];
    int valid[6] = { 1, 1, 1, 1, 1, 1 };
    float p_sorted[6][3];
    float orient;
    vislib::math::Vector<float, 3> v_10, v_12;
    vislib::math::Vector<float, 3> normal;

    xMax = _ext[0];
    yMax = _ext[1];
    zMax = _ext[2];

    if (fabs(_v[0]) < VS_EPS)
        _v[0] = 0.0;
    if (fabs(_v[1]) < VS_EPS)
        _v[1] = 0.0;
    if (fabs(_v[2]) < VS_EPS)
        _v[2] = 0.0;

    _p[0][0] = (d + _v[1]*yMax + _v[2]*zMax) / _v[0];
    _p[1][0] = (d - _v[1]*yMax + _v[2]*zMax) / _v[0];
    _p[2][0] = (d + _v[1]*yMax - _v[2]*zMax) / _v[0];
    _p[3][0] = (d - _v[1]*yMax - _v[2]*zMax) / _v[0];

    _p[4][1] = (d + _v[0]*xMax + _v[2]*zMax) / _v[1];
    _p[5][1] = (d - _v[0]*xMax + _v[2]*zMax) / _v[1];
    _p[6][1] = (d + _v[0]*xMax - _v[2]*zMax) / _v[1];
    _p[7][1] = (d - _v[0]*xMax - _v[2]*zMax) / _v[1];

    _p[8][2] =  (d + _v[0]*xMax + _v[1]*yMax) / _v[2];
    _p[9][2] =  (d - _v[0]*xMax + _v[1]*yMax) / _v[2];
    _p[10][2] = (d + _v[0]*xMax - _v[1]*yMax) / _v[2];
    _p[11][2] = (d - _v[0]*xMax - _v[1]*yMax) / _v[2];

    for (int i=0; i<4; ++i) {
        if (fabs(_p[i][0]) < (xMax + VS_EPS)) {
            validHit[i] = 1;
            numIntersect++;
        }
    }
    for (int i=4; i<8; ++i) {
        if (fabs(_p[i][1]) < (yMax + VS_EPS)) {
            validHit[i] = 1;
            numIntersect++;
        }
    }
    for (int i=8; i<12; ++i) {
        if (fabs(_p[i][2]) < (zMax + VS_EPS)) {
            validHit[i] = 1;
            numIntersect++;
        }
    }

    oldInt = numIntersect;

    // eliminate double vertizes
    if (numIntersect > 2) {
        for (int i=0; i<12; ++i) {
            for (int j=i+1; j<12; ++j) {
                if (validHit[i] && validHit[j]) {
                    if (pointCmp(_p[i], _p[j], (float) VS_EPS)) {
                        validHit[j] = 0;
                        numIntersect--;
                        edgeCode[i] |= edgeCode[j];
                    }
                }
            }
        }
    }

    // there are only 6 possible intersections
    assert(numIntersect < 7);

    // move hits to smaller array
    idx = 0;
    for (int i=0; i<12; ++i) {
        if (validHit[i]) {
            tmp[idx] = i;
            idx++;
        }
    }

    // sort and move final position to a sorted array
    idx = 0;
    for (int j=0; j<numIntersect; ++j) {
        for (int i=0; i<numIntersect; ++i) {
            if ((edgeCode[tmp[i]] & actEdge) && valid[i]) {
                p_sorted[idx][0] = _p[tmp[i]][0] + _ext[0];
                p_sorted[idx][1] = _p[tmp[i]][1] + _ext[1];
                p_sorted[idx][2] = _p[tmp[i]][2] + _ext[2];
                actEdge = edgeCode[tmp[i]];
                valid[i] = 0;
                idx++;
            }
        }
    }

    // determine orientation (counter clockwise / clockwise)
    normal.SetX( _v[0]);
    normal.SetY( _v[1]);
    normal.SetZ( _v[2]);

    v_10.SetX( p_sorted[0][0] - p_sorted[1][0]);
    v_10.SetY( p_sorted[0][1] - p_sorted[1][1]);
    v_10.SetZ( p_sorted[0][2] - p_sorted[1][2]);

    v_12.SetX( p_sorted[2][0] - p_sorted[1][0]);
    v_12.SetY( p_sorted[2][1] - p_sorted[1][1]);
    v_12.SetZ( p_sorted[2][2] - p_sorted[1][2]);

    orient = normal.Dot( v_10.Cross( v_12));
        //Vector3_dot(normal, Vector3_cross(v_10, v_12));

    if (orient > 0.0) {
        // draw it counter clockwise
        for (int i=0; i<numIntersect; ++i) {
            //glMultiTexCoord3fv(GL_TEXTURE0_ARB, p_sorted[i]);
            glVertex3fv(p_sorted[i]);
        }
    } else {
        // it's clockwise, reverse traversing -> counter clockwise
        for (int i=numIntersect-1; i>=0; --i) {
            //glMultiTexCoord3fv(GL_TEXTURE0_ARB, p_sorted[i]);
            glVertex3fv(p_sorted[i]);
        }
    }
}


void ViewSlicing::drawSlices(GLenum mode, int frontToBack, int maxSlices) {
    maxSlices = (_numSlices < maxSlices) ? _numSlices : maxSlices;

    glPushMatrix();
    //  glTranslatef(_ext[0], _ext[1], _ext[2]);
    if (frontToBack) {
        for (int i=0; i<maxSlices; ++i) {
            glBegin(mode);
            {
                drawSlice(i);
            }
            glEnd();
            //      glFinish();
        }
    } else {
        // back to front
        for (int i=_numSlices; i>_numSlices-maxSlices; --i) {
            glBegin(mode);
            {
                drawSlice(i);
            }
            glEnd();
            //      glFinish();
        }
    }
    glPopMatrix();
}


void ViewSlicing::setupSingleSlice(double *viewVec, float *ext) {
    float xMax, yMax, zMax;
    float dv[7];
    float v[3];
    double len;

    _ext[0] = xMax = 0.5f*ext[0];
    _ext[1] = yMax = 0.5f*ext[1];
    _ext[2] = zMax = 0.5f*ext[2];

    _p[0][1] =  -yMax;  _p[0][2] =  -zMax;
    _p[1][1] =   yMax;  _p[1][2] =  -zMax;
    _p[2][1] =  -yMax;  _p[2][2] =   zMax;
    _p[3][1] =   yMax;  _p[3][2] =   zMax;
    _p[4][0] =  -xMax;  _p[4][2] =  -zMax;
    _p[5][0] =   xMax;  _p[5][2] =  -zMax;
    _p[6][0] =  -xMax;  _p[6][2] =   zMax;
    _p[7][0] =   xMax;  _p[7][2] =   zMax;
    _p[8][0] =  -xMax;  _p[8][1]  = -yMax;
    _p[9][0] =   xMax;  _p[9][1]  = -yMax;
    _p[10][0] = -xMax;  _p[10][1] =  yMax;
    _p[11][0] =  xMax;  _p[11][1] =  yMax;

    // calculate the view vector
    len = sqrt((viewVec[0]*viewVec[0]) + (viewVec[1]*viewVec[1]) + (viewVec[2]*viewVec[2]));
    if (len > VS_EPS)  {
        viewVec[0] /= len;
        viewVec[1] /= len;
        viewVec[2] /= len;
    } else {
        viewVec[0] = viewVec[1] = viewVec[2] = 0.0;
    }

    _v[0] = (float) viewVec[0];
    _v[1] = (float) viewVec[1];
    _v[2] = (float) viewVec[2];

    v[0] = fabs(_v[0]);
    v[1] = fabs(_v[1]);
    v[2] = fabs(_v[2]);

    dv[0] = v[0]*xMax;
    dv[1] = v[1]*yMax;
    dv[2] = v[2]*zMax;
    dv[3] = v[0]*xMax + v[1]*yMax;
    dv[4] = v[0]*xMax + v[2]*zMax;
    dv[5] = v[1]*yMax + v[2]*zMax;
    dv[6] = v[0]*xMax + v[1]*yMax + v[2]*zMax;
}


void ViewSlicing::drawSingleSlice(float dist) {
    float xMax, yMax, zMax;
    char validHit[12] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    char edgeCode[12] = { cubeEdges[0], cubeEdges[1], cubeEdges[2], 
        cubeEdges[3], cubeEdges[4], cubeEdges[5], 
        cubeEdges[6], cubeEdges[7], cubeEdges[8], 
        cubeEdges[9], cubeEdges[10], cubeEdges[11] };
    char actEdge = (char)0xff;
    int numIntersect = 0, oldInt;
    int idx;
    int tmp[6];
    int valid[6] = { 1, 1, 1, 1, 1, 1 };

    float p_sorted[6][3];
    float orient;
    vislib::math::Vector<float, 3> v_10, v_12;
    vislib::math::Vector<float, 3> normal;

    xMax = _ext[0];
    yMax = _ext[1];
    zMax = _ext[2];

    if (fabs(_v[0]) < VS_EPS)
        _v[0] = 0.0;
    if (fabs(_v[1]) < VS_EPS)
        _v[1] = 0.0;
    if (fabs(_v[2]) < VS_EPS)
        _v[2] = 0.0;


    _p[0][0] = (dist + _v[1]*yMax + _v[2]*zMax) / _v[0];
    _p[1][0] = (dist - _v[1]*yMax + _v[2]*zMax) / _v[0];
    _p[2][0] = (dist + _v[1]*yMax - _v[2]*zMax) / _v[0];
    _p[3][0] = (dist - _v[1]*yMax - _v[2]*zMax) / _v[0];

    _p[4][1] = (dist + _v[0]*xMax + _v[2]*zMax) / _v[1];
    _p[5][1] = (dist - _v[0]*xMax + _v[2]*zMax) / _v[1];
    _p[6][1] = (dist + _v[0]*xMax - _v[2]*zMax) / _v[1];
    _p[7][1] = (dist - _v[0]*xMax - _v[2]*zMax) / _v[1];

    _p[8][2] =  (dist + _v[0]*xMax + _v[1]*yMax) / _v[2];
    _p[9][2] =  (dist - _v[0]*xMax + _v[1]*yMax) / _v[2];
    _p[10][2] = (dist + _v[0]*xMax - _v[1]*yMax) / _v[2];
    _p[11][2] = (dist - _v[0]*xMax - _v[1]*yMax) / _v[2];

    for (int i=0; i<4; ++i) {
        if (fabs(_p[i][0]) < (xMax + VS_EPS)) {
            validHit[i] = 1;
            numIntersect++;
        }
    }
    for (int i=4; i<8; ++i) {
        if (fabs(_p[i][1]) < (yMax + VS_EPS)) {
            validHit[i] = 1;
            numIntersect++;
        }
    }
    for (int i=8; i<12; ++i) {
        if (fabs(_p[i][2]) < (zMax + VS_EPS)) {
            validHit[i] = 1;
            numIntersect++;
        }
    }

    oldInt = numIntersect;

    // eliminate double vertizes
    if (numIntersect < 3)
        return;

    for (int i=0; i<12; ++i) {
        for (int j=i+1; j<12; ++j) {
            if (validHit[i] && validHit[j]) {
                if (pointCmp(_p[i], _p[j], (float) VS_EPS)) {
                    validHit[j] = 0;
                    numIntersect--;
                    edgeCode[i] |= edgeCode[j];
                }
            }
        }
    }

    if (numIntersect < 3)
        return;

    // there are only 6 possible intersections
    assert(numIntersect < 7);

    // move hits to smaller array
    idx = 0;
    for (int i=0; i<12; ++i) {
        if (validHit[i]) {
            tmp[idx] = i;
            idx++;
        }
    }

    // sort and move final position to a sorted array
    idx = 0;
    for (int j=0; j<numIntersect; ++j) {
        for (int i=0; i<numIntersect; ++i) {
            if ((edgeCode[tmp[i]] & actEdge) && valid[i]) {
                p_sorted[idx][0] = _p[tmp[i]][0] + _ext[0];
                p_sorted[idx][1] = _p[tmp[i]][1] + _ext[1];
                p_sorted[idx][2] = _p[tmp[i]][2] + _ext[2];
                actEdge = edgeCode[tmp[i]];
                valid[i] = 0;
                idx++;
            }
        }
    }

    // determine orientation (counter clockwise / clockwise)
    normal.SetX( _v[0]);
    normal.SetY( _v[1]);
    normal.SetZ( _v[2]);

    v_10.SetX( p_sorted[0][0] - p_sorted[1][0]);
    v_10.SetY( p_sorted[0][1] - p_sorted[1][1]);
    v_10.SetZ( p_sorted[0][2] - p_sorted[1][2]);

    v_12.SetX( p_sorted[2][0] - p_sorted[1][0]);
    v_12.SetY( p_sorted[2][1] - p_sorted[1][1]);
    v_12.SetZ( p_sorted[2][2] - p_sorted[1][2]);

    orient = normal.Dot( v_10.Cross( v_12));
        //Vector3_dot(normal, Vector3_cross(v_10, v_12));

    if (orient > 0.0) {
        // draw it counter clockwise
        for (int i=0; i<numIntersect; ++i) {
            //glMultiTexCoord3fv(GL_TEXTURE0_ARB, p_sorted[i]);
            glVertex3fv(p_sorted[i]);
        }
    } else {
        // it's clockwise, reverse traversing -> counter clockwise
        for (int i=numIntersect-1; i>=0; --i) {
            //glMultiTexCoord3fv(GL_TEXTURE0_ARB, p_sorted[i]);
            glVertex3fv(p_sorted[i]);
        }
    }
}


inline bool ViewSlicing::pointCmp( float *p1, float *p2, float eps) {
    if ((fabs(p1[0] - p2[0]) < eps)
        && (fabs(p1[1] - p2[1]) < eps)
        && (fabs(p1[2] - p2[2]) < eps))
        return true;
    else
        return false;
}

