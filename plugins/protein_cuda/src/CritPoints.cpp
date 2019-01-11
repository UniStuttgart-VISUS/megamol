/*
 * CritPoints.cpp
 *
 * Copyright (C) 2012 by University of Stuttgart (VISUS).
 * All rights reserved.
 *
 * $Id$
 */

#include "stdafx.h"

#include <cmath>
#include <math.h>

#ifndef M_PI 
#define M_PI    3.14159265358979323846f 
#endif

#include "CritPoints.h"
#include "vislib/sys/Log.h"
#include "vislib/math/Vector.h"
#include "vislib/math/Cuboid.h"

#include "helper_math.h"

using namespace megamol;
using namespace vislib;
using namespace vislib::math;

/*
 * protein:CritPoints::GetCritPoints
 */
vislib::Array<float> protein_cuda::CritPoints::GetCritPoints(UniGrid3D<float3> &uniGrid,
        vislib::math::Vector<float, 3> minCoord,
        vislib::math::Vector<float, 3> maxCoord) {

    using namespace vislib;
    using namespace vislib::sys;
    using namespace vislib::math;

    time_t t = clock();

    Array<float> critPoints;
    critPoints.SetCapacityIncrement(1000);
    UniGrid3D<bool> isCritPoint;
    isCritPoint.Init(
            uniGrid.GetGridDim(),
            uniGrid.GetGridOrg(),
            uniGrid.GetGridStepSize());
//#pragma omp parallel
//{
#pragma omp parallel for
    for(int x = 0; x < static_cast<int>(uniGrid.GetGridDim().X())-1; x++) {
        for(int y = 0; y < static_cast<int>(uniGrid.GetGridDim().Y())-1; y++) {
            for(int z = 0; z < static_cast<int>(uniGrid.GetGridDim().Z())-1; z++) {
                Vector<float, 3> minC(
                        minCoord.X()+x*uniGrid.GetGridStepSize(),
                        minCoord.Y()+y*uniGrid.GetGridStepSize(),
                        minCoord.Z()+z*uniGrid.GetGridStepSize());
                Vector<float, 3> maxC(
                        minC.X() + uniGrid.GetGridStepSize(),
                        minC.Y() + uniGrid.GetGridStepSize(),
                        minC.Z() + uniGrid.GetGridStepSize());
                int degree = CritPoints::calcDegreeOfCell(uniGrid, minC, maxC);
                //printf("Degree %i\n", degree);
                if((degree == 1)||(degree == -1)) {
                    isCritPoint.SetAt(x, y, z, true);
                }
                else {
                    isCritPoint.SetAt(x, y, z, false);
                }
            }
        }
        printf("x-slice %i done\n", x);
    }
//}

    for(int x = 1; x < static_cast<int>(uniGrid.GetGridDim().X())-2; x++) {
        for(int y = 1; y < static_cast<int>(uniGrid.GetGridDim().Y())-2; y++) {
            for(int z = 1; z < static_cast<int>(uniGrid.GetGridDim().Z())-2; z++) {
                if(isCritPoint.GetAt(x,y,z)) {
                    Vector<float, 3> minC(
                            minCoord.X()+x*uniGrid.GetGridStepSize(),
                            minCoord.Y()+y*uniGrid.GetGridStepSize(),
                            minCoord.Z()+z*uniGrid.GetGridStepSize());
                    /*Vector<float, 3> maxC(
                            minC.X() + uniGrid.GetGridStepSize(),
                            minC.Y() + uniGrid.GetGridStepSize(),
                            minC.Z() + uniGrid.GetGridStepSize());*/
                    critPoints.Add(minC.X() + uniGrid.GetGridStepSize()*0.5f);
                    critPoints.Add(minC.Y() + uniGrid.GetGridStepSize()*0.5f);
                    critPoints.Add(minC.Z() + uniGrid.GetGridStepSize()*0.5f);
                }
            }
        }
    }

    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
            "Time for computing critical points %f",
            (double(clock()-t)/double(CLOCKS_PER_SEC) )); // DEBUG

    return critPoints;
}



vislib::Array<float> protein_cuda::CritPoints::GetCritPointsGreene(
        UniGrid3D<float3> &uniGrid,
        vislib::math::Vector<float, 3> minGridCoord,
        vislib::math::Vector<float, 3> maxGridCoord,
        float cellSize) {

    using namespace vislib;
    using namespace vislib::sys;
    using namespace vislib::math;

    Array<float> critPoints;
    critPoints.SetCapacityIncrement(1000);

    for(float x = minGridCoord.X(); x <= maxGridCoord.X(); x += cellSize) {
        for(float y = minGridCoord.Y(); y <= maxGridCoord.Y(); y += cellSize) {
            for(float z = minGridCoord.Z(); z <= maxGridCoord.Z(); z += cellSize) {
                Vector<float, 3> minC(x, y, z);
                Vector<float, 3> maxC(x+cellSize, y+cellSize, z+cellSize);
                int degree = CritPoints::calcDegreeOfCell(uniGrid, minC, maxC);
                //printf("Degree %i\n", degree);
                if((degree == 1)||(degree == -1)) {
                    if(cellSize <= uniGrid.GetGridStepSize()) {
                        critPoints.Add(x + cellSize*0.5f);
                        critPoints.Add(y + cellSize*0.5f);
                        critPoints.Add(z + cellSize*0.5f);
                    }
                    else {

                    }
                }
            }
        }
        printf("x-slice %f done\n", x);
    }

    return critPoints;
}


/*vislib::Array<float> protein_cuda::CritPoints::GetCritPoints(UniGrid3D<float3> &uniGrid,
        vislib::math::Vector<float, 3> minCoord,
        vislib::math::Vector<float, 3> maxCoord) {

    using namespace vislib;
    using namespace vislib::sys;
    using namespace vislib::math;

    Array<float> critPoints;
    critPoints.SetCapacityIncrement(300);

    int degree = CritPoints::calcDegreeOfCell(uniGrid, minCoord, maxCoord);
    printf("==== Subvolume (%f %f %f) (%f %f %f) Topological degree %i\n", minCoord.X(),
    minCoord.Y(), minCoord.Z(), maxCoord.X(), maxCoord.Y(), maxCoord.Z(), degree);

    if(degree != 0) {

        Vector<float, 3> center, step;

        step.Set((maxCoord.X() - minCoord.X())*0.5f,
                 (maxCoord.Y() - minCoord.Y())*0.5f,
                 (maxCoord.Z() - minCoord.Z())*0.5f);

        // Get center of the current
        center = minCoord + step;

        if((maxCoord.X() - minCoord.X()) <= uniGrid.GetGridStepSize()*110) {
            // Append center of cell to critical points
            critPoints.Add(center.X());
            critPoints.Add(center.Y());
            critPoints.Add(center.Z());
        }
        else {
            Array<float> res;

            // Bisect current cell, call recursively and append results

            // #0
            res = CritPoints::GetCritPoints(
                    uniGrid,
                    minCoord,    // Mincoord
                    center);     // Maxcoord
            for(unsigned int cnt = 0; cnt < res.Count(); cnt++) {
                critPoints.Add(res[cnt]);
            }
            res.Clear();

            // #1
            res = CritPoints::GetCritPoints(
                    uniGrid,
                    Vector<float, 3> (minCoord.X(), minCoord.Y(), center.Z()),   // Mincoord
                    Vector<float, 3> (center.X(), center.Y(), maxCoord.Z()));    // Maxcoord
            for(unsigned int cnt = 0; cnt < res.Count(); cnt++) {
                critPoints.Add(res[cnt]);
            }
            res.Clear();

            // #2
            res = CritPoints::GetCritPoints(
                    uniGrid,
                    Vector<float, 3> (minCoord.X(), center.Y(), minCoord.Z()),   // Mincoord
                    Vector<float, 3> (center.X(), maxCoord.Y(), center.Z()));    // Maxcoord
            for(unsigned int cnt = 0; cnt < res.Count(); cnt++) {
                critPoints.Add(res[cnt]);
            }
            res.Clear();

            // #3
            res = CritPoints::GetCritPoints(
                    uniGrid,
                    Vector<float, 3> (minCoord.X(), center.Y(), center.Z()),     // Mincoord
                    Vector<float, 3> (center.X(), maxCoord.Y(), maxCoord.Z()));  // Maxcoord
            for(unsigned int cnt = 0; cnt < res.Count(); cnt++) {
                critPoints.Add(res[cnt]);
            }
            res.Clear();

            // #4
            res = CritPoints::GetCritPoints(
                    uniGrid,
                    Vector<float, 3> (center.X(), minCoord.Y(), minCoord.Z()),   // Mincoord
                    Vector<float, 3> (maxCoord.X(), center.Y(), center.Z()));    // Maxcoord
            for(unsigned int cnt = 0; cnt < res.Count(); cnt++) {
                critPoints.Add(res[cnt]);
            }
            res.Clear();

            // #5
            res = CritPoints::GetCritPoints(
                    uniGrid,
                    Vector<float, 3> (center.X(), minCoord.Y(), center.Z()),     // Mincoord
                    Vector<float, 3> (maxCoord.X(), center.Y(), maxCoord.Z()));  // Maxcoord
            for(unsigned int cnt = 0; cnt < res.Count(); cnt++) {
                critPoints.Add(res[cnt]);
            }
            res.Clear();

            // #6
            res = CritPoints::GetCritPoints(
                    uniGrid,
                    Vector<float, 3> (center.X(), center.Y(), minCoord.Z()),     // Mincoord
                    Vector<float, 3> (maxCoord.X(), maxCoord.Y(), center.Z()));  // Maxcoord
            for(unsigned int cnt = 0; cnt < res.Count(); cnt++) {
                critPoints.Add(res[cnt]);
            }
            res.Clear();

            // #7
            res = CritPoints::GetCritPoints(
                    uniGrid,
                    center,     // Mincoord
                    maxCoord);  // Maxcoord
            for(unsigned int cnt = 0; cnt < res.Count(); cnt++) {
                critPoints.Add(res[cnt]);
            }
            res.Clear();
        }
    }

    critPoints.Trim();
    return critPoints;
}*/


/*
 * protein_cuda::CritPoints::calcDegreeOfCell
 */
int protein_cuda::CritPoints::calcDegreeOfCell(UniGrid3D<float3> &uniGrid,
        vislib::math::Vector<float, 3> minCoord,
        vislib::math::Vector<float, 3> maxCoord) {

    float degree = 0.0f;
    Array<Vector<float, 3> > triangle;
    triangle.SetCount(3);

//#pragma omp parallel
    //{

    //#pragma omp sections
    //{
            // Bottom
    //#pragma omp section nowait
    //{
    // Triangle #0
    triangle[0].Set(minCoord[0], minCoord[1], minCoord[2]);
    triangle[1].Set(maxCoord[0], minCoord[1], minCoord[2]);
    triangle[2].Set(maxCoord[0], minCoord[1], maxCoord[2]);
    degree += CritPoints::calcSolidAngleOfTriangle(uniGrid, triangle);
    //printf("==== Curr degree (bottom) %f\n", degree); // DEBUG
    //}

    //#pragma omp section nowait
    //{
    // Triangle #1
    triangle[0].Set(minCoord[0], minCoord[1], minCoord[2]);
    triangle[1].Set(maxCoord[0], minCoord[1], maxCoord[2]);
    triangle[2].Set(minCoord[0], minCoord[1], maxCoord[2]);
    degree += CritPoints::calcSolidAngleOfTriangle(uniGrid, triangle);
    //printf("==== Curr degree (bottom) %f\n", degree); // DEBUG
    //}

    // Top
    //#pragma omp section nowait
    //{
    // Triangle #2
    triangle[0].Set(minCoord[0], maxCoord[1], minCoord[2]);
    triangle[1].Set(maxCoord[0], maxCoord[1], maxCoord[2]);
    triangle[2].Set(maxCoord[0], maxCoord[1], minCoord[2]);
    degree += CritPoints::calcSolidAngleOfTriangle(uniGrid, triangle);
    //printf("==== Curr degree (top) %f\n", degree); // DEBUG
    //}

    //#pragma omp section nowait
    //{
    // Triangle #3
    triangle[0].Set(minCoord[0], maxCoord[1], minCoord[2]);
    triangle[1].Set(minCoord[0], maxCoord[1], maxCoord[2]);
    triangle[2].Set(maxCoord[0], maxCoord[1], maxCoord[2]);
    degree += CritPoints::calcSolidAngleOfTriangle(uniGrid, triangle);
    //printf("==== Curr degree (top) %f\n", degree); // DEBUG
    //}

// Front
    //#pragma omp section nowait
    //{
    // Triangle #4
    triangle[0].Set(minCoord[0], minCoord[1], maxCoord[2]);
    triangle[1].Set(maxCoord[0], maxCoord[1], maxCoord[2]);
    triangle[2].Set(minCoord[0], maxCoord[1], maxCoord[2]);
    degree += CritPoints::calcSolidAngleOfTriangle(uniGrid, triangle);
    //printf("==== Curr degree (front) %f\n", degree); // DEBUG
    //}

    //#pragma omp section nowait
    //{
    // Triangle #5
    triangle[0].Set(minCoord[0], minCoord[1], maxCoord[2]);
    triangle[1].Set(maxCoord[0], minCoord[1], maxCoord[2]);
    triangle[2].Set(maxCoord[0], maxCoord[1], maxCoord[2]);
    degree += CritPoints::calcSolidAngleOfTriangle(uniGrid, triangle);
    //printf("==== Curr degree (front) %f\n", degree); // DEBUG
    //}

// Back
    //#pragma omp section nowait
    //{
    // Triangle #6
    triangle[0].Set(minCoord[0], minCoord[1], minCoord[2]);
    triangle[1].Set(minCoord[0], maxCoord[1], minCoord[2]);
    triangle[2].Set(maxCoord[0], maxCoord[1], minCoord[2]);
    degree += CritPoints::calcSolidAngleOfTriangle(uniGrid, triangle);
    //printf("==== Curr degree (back) %f\n", degree); // DEBUG
    //}

    //#pragma omp section nowait
    //{
    // Triangle #7
    triangle[0].Set(minCoord[0], minCoord[1], minCoord[2]);
    triangle[1].Set(maxCoord[0], maxCoord[1], minCoord[2]);
    triangle[2].Set(maxCoord[0], minCoord[1], minCoord[2]);
    degree += CritPoints::calcSolidAngleOfTriangle(uniGrid, triangle);
    //printf("==== Curr degree (back) %f\n", degree); // DEBUG
    //}

// Left
    //#pragma omp section nowait
    //{
    // Triangle #8
    triangle[0].Set(minCoord[0], minCoord[1], minCoord[2]);
    triangle[1].Set(minCoord[0], minCoord[1], maxCoord[2]);
    triangle[2].Set(minCoord[0], maxCoord[1], maxCoord[2]);
    degree += CritPoints::calcSolidAngleOfTriangle(uniGrid, triangle);
    //printf("==== Curr degree (left) %f\n", degree); // DEBUG
    //}

    //#pragma omp section nowait
    //{
    // Triangle #9
    triangle[0].Set(minCoord[0], minCoord[1], minCoord[2]);
    triangle[1].Set(minCoord[0], maxCoord[1], maxCoord[2]);
    triangle[2].Set(minCoord[0], maxCoord[1], minCoord[2]);
    degree += CritPoints::calcSolidAngleOfTriangle(uniGrid, triangle);
    //printf("==== Curr degree (left) %f\n", degree); // DEBUG
    //}

// Right
    //#pragma omp section nowait
    //{
    // Triangle #10
    triangle[0].Set(maxCoord[0], minCoord[1], minCoord[2]);
    triangle[1].Set(maxCoord[0], maxCoord[1], minCoord[2]);
    triangle[2].Set(maxCoord[0], maxCoord[1], maxCoord[2]);
    degree += CritPoints::calcSolidAngleOfTriangle(uniGrid, triangle);
    //printf("==== Curr degree (right) %f\n", degree); // DEBUG
    //}

    //#pragma omp section nowait
    //{
    // Triangle #11
    triangle[0].Set(maxCoord[0], minCoord[1], minCoord[2]);
    triangle[1].Set(maxCoord[0], maxCoord[1], maxCoord[2]);
    triangle[2].Set(maxCoord[0], minCoord[1], maxCoord[2]);
    degree += CritPoints::calcSolidAngleOfTriangle(uniGrid, triangle);
    //printf("==== Curr degree (right) %f\n", degree); // DEBUG
    //}
    //}
    //}

    degree /= (4.0f*M_PI);

    if((degree > 10.0f)||(degree < -10.0f)) {
        degree = 0.0f;
    }

    return static_cast<int>(degree);
}



/*
 * protein_cuda::CritPoints::calcSolidAngleOfTriangle
 */
float protein_cuda::CritPoints::calcSolidAngleOfTriangle(UniGrid3D<float3> &uniGrid,
        vislib::Array<vislib::math::Vector<float, 3> > points) {

    float angleSolid = 0.0f, result;
    float theta0, theta1, theta2;

    vislib::Array<vislib::math::Vector<float, 3> > triVecs;
    triVecs.SetCount(3);

    float3 triVecs_f3_0 = uniGrid.SampleNearest(points[0].X(), points[0].Y(), points[0].Z());
    float3 triVecs_f3_1 = uniGrid.SampleNearest(points[1].X(), points[1].Y(), points[1].Z());
    float3 triVecs_f3_2 = uniGrid.SampleNearest(points[2].X(), points[2].Y(), points[2].Z());

    triVecs[0] = Vector<float, 3>(triVecs_f3_0.x, triVecs_f3_0.y, triVecs_f3_0.z);
    triVecs[1] = Vector<float, 3>(triVecs_f3_1.x, triVecs_f3_1.y, triVecs_f3_1.z);
    triVecs[2] = Vector<float, 3>(triVecs_f3_2.x, triVecs_f3_2.y, triVecs_f3_2.z);

    //printf("==== pt %f, %f, %f, vec0 %f, %f, %f\n", points[0].X(), points[0].Y(), points[0].Z(), triVecs[0].X(), triVecs[0].Y(), triVecs[0].Z());
    //printf("==== pt %f, %f, %f, vec1 %f, %f, %f\n", points[1].X(), points[1].Y(), points[1].Z(), triVecs[1].X(), triVecs[1].Y(), triVecs[1].Z());
    //printf("==== pt %f, %f, %f, vec2 %f, %f, %f\n", points[2].X(), points[2].Y(), points[2].Z(), triVecs[2].X(), triVecs[2].Y(), triVecs[2].Z());

    triVecs[0].Normalise();
    triVecs[1].Normalise();
    triVecs[2].Normalise();

    theta0 = acos(triVecs[1].Dot(triVecs[2]));
    theta1 = acos(triVecs[0].Dot(triVecs[2]));
    theta2 = acos(triVecs[0].Dot(triVecs[1]));

    //printf("theta0 %f\n", theta0);
    //printf("theta1 %f\n", theta1);
    //printf("theta2 %f\n", theta2);

    //if(triVecs[0].Length() == 0.0f)
        //printf("Length 0 %f\n", triVecs[0].Length());
    //if(triVecs[1].Length() == 0.0f)
        //printf("Length 1 %f\n", triVecs[1].Length());
    //if(triVecs[2].Length() == 0.0f)
        //printf("Length 2 %f\n", triVecs[2].Length());

    result = tan(( theta0 + theta1 + theta2)/4.0f)
            *tan(( theta0 + theta1 - theta2)/4.0f)
            *tan((-theta0 + theta1 + theta2)/4.0f)
            *tan(( theta0 - theta1 + theta2)/4.0f);

    //printf("result1 %f\n", result);
    result = fabs(result);

    result = sqrt(result);

    //printf("result2 %f\n", result);

    result = atan(result);

    //printf("result3 %f\n", result);

    vislib::math::Vector<float, 3> crossPr = triVecs[1].Cross(triVecs[2]);

    if (triVecs[0].Dot(crossPr) < 0.0f) {
        angleSolid = -result*4.0f;
    }
    else {
        angleSolid = result*4.0f;
    }

    return angleSolid;
}


float protein_cuda::CritPoints::calcSolidAngleOfTriangleAlt(UniGrid3D<float3> &uniGrid,
        vislib::Array<vislib::math::Vector<float, 3> > points) {

    vislib::Array<vislib::math::Vector<float, 3> > triVecs;
    triVecs.SetCount(3);
    /*triVecs[0] = CritPoints::sampleUniGridNearestNeighbour(uniGrid, points[0]);
    triVecs[1] = CritPoints::sampleUniGridNearestNeighbour(uniGrid, points[1]);
    triVecs[2] = CritPoints::sampleUniGridNearestNeighbour(uniGrid, points[2]);*/

    triVecs[0] = vislib::math::Vector<float, 3>(1.0, 0.0, 0.0);
    triVecs[1] = vislib::math::Vector<float, 3>(0.0, 1.0, 0.0);
    triVecs[2] = vislib::math::Vector<float, 3>(0.0, 0.0, 1.0);

    //printf("==== vec0 %f, %f, %f (Alt)\n", triVecs[0].X(), triVecs[0].Y(), triVecs[0].Z());
    //printf("==== vec1 %f, %f, %f (Alt)\n", triVecs[1].X(), triVecs[1].Y(), triVecs[1].Z());
    //printf("==== vec2 %f, %f, %f (Alt)\n", triVecs[2].X(), triVecs[2].Y(), triVecs[2].Z());

    double determ =     triVecs[0].X()*triVecs[1].Y()*triVecs[2].Z()
                      + triVecs[1].X()*triVecs[2].Y()*triVecs[0].Z()
                      + triVecs[2].X()*triVecs[0].Y()*triVecs[1].Z()
                      - triVecs[2].X()*triVecs[1].Y()*triVecs[0].Z()
                      - triVecs[1].X()*triVecs[0].Y()*triVecs[2].Z()
                      - triVecs[0].X()*triVecs[2].Y()*triVecs[1].Z();

    //printf("Determ %e\n", determ);

    float length[3];
    length[0] = triVecs[0].Length();
    length[1] = triVecs[1].Length();
    length[2] = triVecs[2].Length();

    //printf("al %e\n", length[0]);
    //printf("bl %e\n", length[1]);
    //printf("cl %e\n", length[2]);

    float div = length[0]*length[1]*length[2] +
            triVecs[0].Dot(triVecs[1])*length[2] +
            triVecs[0].Dot(triVecs[2])*length[1] +
            triVecs[1].Dot(triVecs[2])*length[0];

    //printf("Div %e\n",div );

    float at = static_cast<float>(atan2(determ, double(div)));
    if(at < 0) {
        at += M_PI; // If det > 0 and div < 0 arctan2 returns < 0, so add pi.
    }

    float omega = 2.0f * at;
    //printf("angle %f (Alt)\n", omega);

    return omega;
}


/*
 * protein_cuda::CritPoints::sampleUniGridNearestNeighbour
 */
vislib::math::Vector<float, 3> protein_cuda::CritPoints::sampleUniGridNearestNeighbour(
        UniGrid3D<float3> &uniGrid,
        vislib::math::Vector<float, 3> pos) {

    /*printf("==== gridDim %u %u %u\n",
            uniGrid.GetGridDim().X(),
            uniGrid.GetGridDim().Y(),
            uniGrid.GetGridDim().Z());

    printf("==== pos %f %f %f\n",
            pos.X() - uniGrid.GetGridOrg().X(),
            pos.Y() - uniGrid.GetGridOrg().Y(),
            pos.Z() - uniGrid.GetGridOrg().Z());

    printf("==== pos %u %u %u\n",
            static_cast<unsigned int>(pos.X() - uniGrid.GetGridOrg().X()),
            static_cast<unsigned int>(pos.Y() - uniGrid.GetGridOrg().Y()),
            static_cast<unsigned int>(pos.Z() - uniGrid.GetGridOrg().Z()));*/

    float3 vecRes =  uniGrid.GetAt(
            static_cast<unsigned int>(pos.X() - uniGrid.GetGridOrg().X()),
            static_cast<unsigned int>(pos.Y() - uniGrid.GetGridOrg().Y()),
            static_cast<unsigned int>(pos.Z() - uniGrid.GetGridOrg().Z()));

    return vislib::math::Vector<float, 3>(vecRes.x, vecRes.y, vecRes.z);
}
