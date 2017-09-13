//
// Streamline.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//

#include "stdafx.h"
#include "Streamline.h"

#include "cuda_runtime.h"
#include "helper_cuda.h"


using namespace megamol;
using namespace megamol::protein_cuda;


/* Streamline::IntegrateRK4 */
void Streamline::IntegrateRK4(Vec3f start, VecField3f &v, unsigned int maxLength,
        float step, float eps, Direction dir) {

    step/=10;

    using namespace vislib::math;
    using namespace vislib;

    this->vertexArr.Clear();
    this->tangentArr.Clear();
    this->texCoordArr.Clear();
    this->vertexArr.SetCapacityIncrement(1000);
    this->tangentArr.SetCapacityIncrement(1000);
    this->texCoordArr.SetCapacityIncrement(1000);

    bool vanishing = false, gridLeft = false;
    unsigned int l0 = 0, l1 = 0;
    Vec3f v0, v1, v2, v3, x0, x1, x2, x3, color, v_test;

    //printf("Bla0\n");

    // 1. Forward
    if((dir == Streamline::FORWARD)||(dir == Streamline::BIDIRECTIONAL)) {

        x0 = start;


        float cx,cy,cz;
        cx = (x0.X() - v.GetOrg().X())/v.GetSpacing().X();
        cy = (x0.Y() - v.GetOrg().Y())/v.GetSpacing().Y();
        cz = (x0.Z() - v.GetOrg().Z())/v.GetSpacing().Z();
        Vector<unsigned int, 3> cellId;
        cellId[0] = static_cast<unsigned int>(cx);
        cellId[1] = static_cast<unsigned int>(cy);
        cellId[2] = static_cast<unsigned int>(cz);

        //printf("Pos %f %f %f\n", posX, posY, posZ); // DEBUG
        //printf("c %f %f %f\n", cx, cy, cz); // DEBUG
        //printf("CellId %u %u %u\n", cellId[0], cellId[1], cellId[2]); // DEBUG

        cx -= cellId[0]; // alpha
        cy -= cellId[1]; // beta
        cz -= cellId[2]; // gamma

        //v_test = v.GetAtTrilin(x0, true);
        //v_test = v.GetAt(cellId[0]+1, cellId[1]+1, cellId[2]+1);

        Vector<unsigned int, 3> posStart = v.GetCellId(x0);
        /*printf("start pos: %f %f %f, org %f %f %f, spacing %f %f %f, abg f: %f %f %f, Start cell: %u %u %u, sample: %f %f %f\n",
                x0.X(), x0.Y(), x0.Z(), v.GetOrg().X(), v.GetOrg().Y(), v.GetOrg().Z(),
                v.GetSpacing().X(), v.GetSpacing().Y(), v.GetSpacing().Z(),
                cx, cy, cz,
                posStart.X(), posStart.Y(), posStart.Z(),
                v_test.X(), v_test.Y(), v_test.Z());*/

        // Test whether the grid has been left
        if(!v.IsValidGridpos(x0)) {
            //printf("Gridleft = true, pos %f %f %f\n", x0.X(), x0.Y(), x0.Z());
            gridLeft = true;
        }

        while (!(vanishing || gridLeft || l0 > maxLength)) {

            v0.Set(0.0f, 0.0f, 0.0f);
            v1.Set(0.0f, 0.0f, 0.0f);
            v2.Set(0.0f, 0.0f, 0.0f);
            v3.Set(0.0f, 0.0f, 0.0f);

            // Find new position using fourth order Runge-Kutta method

            if(v.IsValidGridpos(x0)) {
                v0 = v.GetAtTrilin(x0, true);
                if(v0.Norm() <= eps) vanishing = true;
                v0.Normalise();

                // Add position and tangent to streamline
                this->vertexArr.Add(x0.X());
                this->vertexArr.Add(x0.Y());
                this->vertexArr.Add(x0.Z());

                this->tangentArr.Add(v0.X());
                this->tangentArr.Add(v0.Y());
                this->tangentArr.Add(v0.Z());

                this->texCoordArr.Add((x0.X()-v.GetOrg().X())/((v.GetDim().X()-1)*v.GetSpacing().X()));
                this->texCoordArr.Add((x0.Y()-v.GetOrg().Y())/((v.GetDim().Y()-1)*v.GetSpacing().Y()));
                this->texCoordArr.Add((x0.Z()-v.GetOrg().Z())/((v.GetDim().Z()-1)*v.GetSpacing().Z()));

                v0 *= step;
            }

            x1 = x0 + 0.5f*v0;
            if(v.IsValidGridpos(x1)) {
                v1 = v.GetAtTrilin(x1, true);
                v1.Normalise();
                v1 *= step;
            }

            x2 = x0 + 0.5f*v1;
            if(v.IsValidGridpos(x2)) {
                v2 = v.GetAtTrilin(x2, true);
                v2.Normalise();
                v2 *= step;
            }

            x3 = x0 + v2;
            if(v.IsValidGridpos(x3)) {
                v3 = v.GetAtTrilin(x3, true);
                v3.Normalise();
                v3 *= step;
            }

            v_test = v.GetAtTrilin(x0, true);

            x0 += (1.0f/6.0f)*(v0+2.0f*v1+2.0f*v2+v3);

            // Test whether the grid has been left
            if(!v.IsValidGridpos(x0)) {
                gridLeft = true;
            }

            l0++;
        }
    }

//printf("Bla1\n");

    // 2. Backward
    if((dir == Streamline::BACKWARD)||(dir == Streamline::BIDIRECTIONAL)) {

        x0 = start;
        vanishing = false;
        gridLeft = false;

        // Test whether the grid has been left
        if(!v.IsValidGridpos(x0)) {
            //printf("Gridleft = true, pos %f %f %f\n", x0.X(), x0.Y(), x0.Z());
            gridLeft = true;
        }

        while (!(vanishing || gridLeft || l1 >= maxLength)) {

            v0.Set(0.0f, 0.0f, 0.0f);
            v1.Set(0.0f, 0.0f, 0.0f);
            v2.Set(0.0f, 0.0f, 0.0f);
            v3.Set(0.0f, 0.0f, 0.0f);

            // Find new position using fourth order Runge-Kutta method

            if(v.IsValidGridpos(x0)) {
                v0 = v.GetAtTrilin(x0, true);
                if(v0.Norm() <= eps) vanishing = true;
                v0.Normalise();

                // Add position and tangent to streamline

                this->vertexArr.Insert(0, x0.Z());
                this->vertexArr.Insert(0, x0.Y());
                this->vertexArr.Insert(0, x0.X());

                this->tangentArr.Insert(0, v0.Z());
                this->tangentArr.Insert(0, v0.Y());
                this->tangentArr.Insert(0, v0.X());

                this->texCoordArr.Insert(0, (x0.Z()-v.GetOrg().Z())/((v.GetDim().Z()-1)*v.GetSpacing().Z()));
                this->texCoordArr.Insert(0, (x0.Y()-v.GetOrg().Y())/((v.GetDim().Y()-1)*v.GetSpacing().Y()));
                this->texCoordArr.Insert(0, (x0.X()-v.GetOrg().X())/((v.GetDim().X()-1)*v.GetSpacing().X()));

                /*printf("tc %f %f %f\n",
                        (x0.X()-v.GetOrg().X())/(v.GetDim().X()*v.GetSpacing().X()),
                        (x0.Y()-v.GetOrg().Y())/(v.GetDim().Y()*v.GetSpacing().Y()),
                        (x0.Z()-v.GetOrg().Z())/(v.GetDim().Z()*v.GetSpacing().Z()));*/

                v0 *= step;
            }

            x1 = x0 - 0.5f*v0;
            if(v.IsValidGridpos(x1)) {
                v1 = v.GetAtTrilin(x1, true);
                v1.Normalise();
                v1 *= step;
            }

            x2 = x0 - 0.5f*v1;
            if(v.IsValidGridpos(x2)) {
                v2 = v.GetAtTrilin(x2, true);
                v2.Normalise();
                v2 *= step;
            }

            x3 = x0 - v2;
            if(v.IsValidGridpos(x3)) {
                v3 = v.GetAtTrilin(x3, true);
                v3.Normalise();
                v3 *= step;
            }


            x0 -= (1.0f/6.0f)*(v0+2.0f*v1+2.0f*v2+v3);

            // Test whether the grid has been left
            if(!v.IsValidGridpos(x0)) {
                gridLeft = true;
            }

            l1++;
        }
    }

   // printf("Bla2, count vertexArr: %u\n", this->vertexArr.Count());

    if(this->vertexArr.Count() > 0) {
        // Set start and end point


        this->startPos.Set(this->vertexArr[0], this->vertexArr[1],
                this->vertexArr[2]);

        //printf("Bla4\n");
        this->endPos.Set(this->vertexArr[this->vertexArr.Count()-3],
                this->vertexArr[this->vertexArr.Count()-2],
                this->vertexArr[this->vertexArr.Count()-1]);
    }

   // printf("Bla3\n");
}




