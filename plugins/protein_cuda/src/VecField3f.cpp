//
// VecField3D.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//




#include "stdafx.h"
#define _USE_MATH_DEFINES 1

#include "VecField3f.h"
#include "protein_calls/Interpol.h"
#include "helper_math.h"



using namespace megamol::protein_cuda;

typedef unsigned int uint;

/* VecField3f::GetAt */
Vec3f VecField3f::GetAt(unsigned int posX, unsigned int posY, unsigned int posZ) {

    using namespace vislib;
    using namespace vislib::math;

    ASSERT(this->data != NULL);

    //printf("POS %u %u %u\n", posX, posY, posZ);

    ASSERT(posX < this->dimX);
    ASSERT(posY < this->dimY);
    ASSERT(posZ < this->dimZ);

    ASSERT(posX >= 0);
    ASSERT(posY >= 0);
    ASSERT(posZ >= 0);

    return Vec3f (this->data[3*(this->dimX*(this->dimY*posZ+posY)+posX)+0],
                         this->data[3*(this->dimX*(this->dimY*posZ+posY)+posX)+1],
                         this->data[3*(this->dimX*(this->dimY*posZ+posY)+posX)+2]);
}


/* VecField3f::GetAtTrilin */
Vec3f VecField3f::GetAtTrilin(float posX, float posY, float posZ,
        bool normalize) {

    float cx,cy,cz;
    cx = (posX - this->orgX)/this->spacingX;
    cy = (posY - this->orgY)/this->spacingY;
    cz = (posZ - this->orgZ)/this->spacingZ;

    Vec3u cellId;
    cellId[0] = static_cast<unsigned int>(cx);
    cellId[1] = static_cast<unsigned int>(cy);
    cellId[2] = static_cast<unsigned int>(cz);

    //printf("Pos %f %f %f\n", posX, posY, posZ); // DEBUG
    //printf("c %f %f %f\n", cx, cy, cz); // DEBUG
    //printf("CellId %u %u %u\n", cellId[0], cellId[1], cellId[2]); // DEBUG

    cx -= cellId[0]; // alpha
    cy -= cellId[1]; // beta
    cz -= cellId[2]; // gamma

    //printf("alpha beta gamma %f %f %f\n", cx, cy, cz); // DEBUG

    // Get neighbour vecs
    Vec3f  n[8];
    n[0] = this->GetAt(cellId.X()+0, cellId.Y()+0, cellId.Z()+0);
    n[1] = this->GetAt(cellId.X()+1, cellId.Y()+0, cellId.Z()+0);
    n[2] = this->GetAt(cellId.X()+0, cellId.Y()+1, cellId.Z()+0);
    n[3] = this->GetAt(cellId.X()+1, cellId.Y()+1, cellId.Z()+0);
    n[4] = this->GetAt(cellId.X()+0, cellId.Y()+0, cellId.Z()+1);
    n[5] = this->GetAt(cellId.X()+1, cellId.Y()+0, cellId.Z()+1);
    n[6] = this->GetAt(cellId.X()+0, cellId.Y()+1, cellId.Z()+1);
    n[7] = this->GetAt(cellId.X()+1, cellId.Y()+1, cellId.Z()+1);
    if(normalize) {
        for(int i = 0; i < 8; i++) n[i].Normalise();
    }

    // DEBUG
    /*printf("alpha=%f, beta=%f, gamma=%f\n", cx, cy, cz);
    printf("n0 (%.16f %.16f %.16f)\n", n[0].X(), n[0].Y(), n[0].Z());
    printf("n1 (%.16f %.16f %.16f)\n", n[1].X(), n[1].Y(), n[1].Z());
    printf("n2 (%.16f %.16f %.16f)\n", n[2].X(), n[2].Y(), n[2].Z());
    printf("n3 (%.16f %.16f %.16f)\n", n[3].X(), n[3].Y(), n[3].Z());
    printf("n4 (%.16f %.16f %.16f)\n", n[4].X(), n[4].Y(), n[4].Z());
    printf("n5 (%.16f %.16f %.16f)\n", n[5].X(), n[5].Y(), n[5].Z());
    printf("n6 (%.16f %.16f %.16f)\n", n[6].X(), n[6].Y(), n[6].Z());
    printf("n7 (%.16f %.16f %.16f)\n", n[7].X(), n[7].Y(), n[7].Z());*/

    // Interpolate
	Vec3f v = protein_calls::Interpol::Trilin<Vec3f>(n[0], n[1], n[2], n[3],
            n[4], n[5], n[6], n[7], cx, cy, cz);

    return v;
}


/* VecField3f::getJacobianAt */
Mat3f VecField3f::GetJacobianAt(unsigned int x, unsigned int y, unsigned int z,
        bool normalize) {

    ASSERT(x < this->dimX-1); // Because we are using central differences
    ASSERT(y < this->dimY-1);
    ASSERT(z < this->dimZ-1);
    ASSERT(x >= 1);
    ASSERT(y >= 1);
    ASSERT(z >= 1);

    Vec3f v[6];
    v[0] = this->GetAt(x+1, y, z);
    v[1] = this->GetAt(x-1, y, z);
    v[2] = this->GetAt(x, y+1, z);
    v[3] = this->GetAt(x, y-1, z);
    v[4] = this->GetAt(x, y, z+1);
    v[5] = this->GetAt(x, y, z-1);

    if(normalize) {
        for(int i = 0; i < 6; i++) v[i].Normalise();
    }

    // Central differences
    Mat3f j(
            (v[0].X()-v[1].X())/(2.0f*this->spacingX), // vx_dx
            (v[2].X()-v[3].X())/(2.0f*this->spacingY), // vx_dy
            (v[4].X()-v[5].X())/(2.0f*this->spacingZ), // vx_dz
            (v[0].Y()-v[1].Y())/(2.0f*this->spacingX), // vy_dx
            (v[2].Y()-v[3].Y())/(2.0f*this->spacingY), // vy_dy
            (v[4].Y()-v[5].Y())/(2.0f*this->spacingZ), // vy_dz
            (v[0].Z()-v[1].Z())/(2.0f*this->spacingX), // vz_dx
            (v[2].Z()-v[3].Z())/(2.0f*this->spacingY), // vz_dy
            (v[4].Z()-v[5].Z())/(2.0f*this->spacingZ)  // vz_dz
    );

    return j;
}


/* VecField3f::IsPosInCell */
bool VecField3f::IsPosInCell(Vec3u cellId, Vec3f pos) {
    float posXf = (pos.X() - this->orgX)/this->spacingX;
    float posYf = (pos.Y() - this->orgY)/this->spacingY;
    float posZf = (pos.Z() - this->orgZ)/this->spacingZ;
    unsigned int posCellX = static_cast<unsigned int>(posXf);
    unsigned int posCellY = static_cast<unsigned int>(posYf);
    unsigned int posCellZ = static_cast<unsigned int>(posZf);

    return (cellId.X() == posCellX)&&(cellId.Y() == posCellY)&&
            (cellId.Z() == posCellZ);
}


/* VecField3f::IsValidGridpos */
bool VecField3f::IsValidGridpos(Vec3f pos) {
    /*printf("IsValidGridPos: %f %f %f\n", pos.X(), pos.Y(), pos.Z());
    printf("Min %f %f %f\n", this->orgX, this->orgY, this->orgZ);
    printf("Max %f %f %f\n", (this->orgX + (this->dimX-1)*this->spacingX),
            (this->orgY + (this->dimY-1)*this->spacingY),
            (this->orgZ + (this->dimZ-1)*this->spacingZ));*/
    if(pos.X() < this->orgX) return false;
    if(pos.Y() < this->orgY) return false;
    if(pos.Z() < this->orgZ) return false;
    if(pos.X() >= (this->orgX + (this->dimX-1)*this->spacingX)) return false;
    if(pos.Y() >= (this->orgY + (this->dimY-1)*this->spacingY)) return false;
    if(pos.Z() >= (this->orgZ + (this->dimZ-1)*this->spacingZ)) return false;
    return true;
}


/* VecField3f::SearchCritPoints */
void VecField3f::SearchCritPoints(unsigned int maxBisections,
        unsigned int maxItNewton, float stepNewton, float epsNewton) {

    this->critPoints.Clear();

//#pragma omp parallel for
    for(int x = 1; x < static_cast<int>(this->dimX-2); x++) {
        for(int y = 1; y < static_cast<int>(this->dimY-2); y++) {
            for(int z = 1; z < static_cast<int>(this->dimZ-2); z++) {
                // Get cell id
                Vec3u cellId(x, y, z);

                // Get corner vals
                vislib::Array<Vec3f> n;
                n.SetCount(8);
                n[0] = this->GetAt(cellId.X()+0, cellId.Y()+0, cellId.Z()+0);
                n[1] = this->GetAt(cellId.X()+1, cellId.Y()+0, cellId.Z()+0);
                n[2] = this->GetAt(cellId.X()+0, cellId.Y()+1, cellId.Z()+0);
                n[3] = this->GetAt(cellId.X()+1, cellId.Y()+1, cellId.Z()+0);
                n[4] = this->GetAt(cellId.X()+0, cellId.Y()+0, cellId.Z()+1);
                n[5] = this->GetAt(cellId.X()+1, cellId.Y()+0, cellId.Z()+1);
                n[6] = this->GetAt(cellId.X()+0, cellId.Y()+1, cellId.Z()+1);
                n[7] = this->GetAt(cellId.X()+1, cellId.Y()+1, cellId.Z()+1);

                for(int i = 0; i < 8; i++) n[i].Normalise();

                /*printf("%u %u %u\n", x,y,z);
                printf("n0 (%.16f %.16f %.16f)\n", n[0].X(), n[0].Y(), n[0].Z());
                printf("n1 (%.16f %.16f %.16f)\n", n[1].X(), n[1].Y(), n[1].Z());
                printf("n2 (%.16f %.16f %.16f)\n", n[2].X(), n[2].Y(), n[2].Z());
                printf("n3 (%.16f %.16f %.16f)\n", n[3].X(), n[3].Y(), n[3].Z());
                printf("n4 (%.16f %.16f %.16f)\n", n[4].X(), n[4].Y(), n[4].Z());
                printf("n5 (%.16f %.16f %.16f)\n", n[5].X(), n[5].Y(), n[5].Z());
                printf("n6 (%.16f %.16f %.16f)\n", n[6].X(), n[6].Y(), n[6].Z());
                printf("n7 (%.16f %.16f %.16f)\n", n[7].X(), n[7].Y(), n[7].Z());*/

                /*if((cellId.X() == 5)&&(cellId.Y() == 16)&&(cellId.Z() == 23))
                    printf("cell %u %u %u, coords %f %f %f, vecfield at center (%f %f %f)\n", x, y, z, 0.5f,
                        0.5f, 0.5f, Interpol::Trilin(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], 0.5, 0.5, 0.5).X(),
                        Interpol::Trilin(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], 0.5, 0.5, 0.5).Y(),
                        Interpol::Trilin(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], 0.5, 0.5, 0.5).Z());
                    printf("cell %u %u %u, coords %f %f %f, vecfield at corner (%f %f %f)\n", x, y, z, 0.5f,
                                                0.5f, 0.5f, n[0].X(), n[0].Y(), n[0].Z());*/

                if(this->isFieldVanishingInCellBisectionRec(1, maxBisections, n)) {
                    Vec3f pos;
                    pos.SetX(this->orgX + (cellId.X() + 0.5f)*this->spacingX);
                    pos.SetY(this->orgY + (cellId.Y() + 0.5f)*this->spacingY);
                    pos.SetZ(this->orgZ + (cellId.Z() + 0.5f)*this->spacingZ);
                    //Vector<float, 3> posNewton = this->searchNullPointNewton(maxItNewton, pos, cellId, stepNewton, epsNewton);
                    Vec3f posNewton = this->searchNullPointNewton(maxItNewton, pos, cellId, stepNewton, 0.0001f);
                    this->critPoints.Add(CritPoint(posNewton, cellId,
                            this->classifyCritPoint(cellId, posNewton)));
                    /*if(posNewton.Z() > 18.0f) printf("posnewton %f %f %f\n",
                            posNewton.X(), posNewton.Y(), posNewton.Z());*/
                    //this->critPoints.Add(CritPoint(pos, cellId, this->classifyCritPoint(cellId, pos)));

                }
            }
        }
    }
    //printf("Null points found: %u\n", static_cast<unsigned int>(this->critPoints.Count())); // DEBUG
}

void VecField3f::SearchCritPointsCUDA(unsigned int maxItNewton,
        float stepNewton, float epsNewton) {
    using namespace vislib;
    using namespace vislib::math;

    this->critPoints.Clear();

    unsigned int nCells = (this->dimX-1)*(this->dimY-1)*(this->dimZ-1);
    float *cellCoords = new float[nCells*3]; // TODO Do not allocate every time

    // Set parameters in constant memory
    SetGridParams(make_uint3(this->dimX, this->dimY, this->dimZ),
                  make_float3(this->orgX, this->orgY, this->orgZ),
                  make_float3(this->orgX + (this->dimX-1)*this->spacingX,
                              this->orgY + (this->dimY-1)*this->spacingY,
                              this->orgZ + (this->dimZ-1)*this->spacingZ),
                  make_float3(this->spacingX, this->spacingY, this->spacingZ));

    // TODO What happens if a double pointer is beeing cast to a float pointer?
    // Call CUDA function
    SearchNullPoints((const float*)this->data,
            make_uint3(this->dimX, this->dimY, this->dimZ),
            make_float3(this->orgX, this->orgY, this->orgZ),
            make_float3(this->spacingX, this->spacingY, this->spacingZ),
            cellCoords, 0);

    for(unsigned int cnt = 0; cnt < nCells; cnt++) {
        if(cellCoords[3*cnt] != -1.0) {
            Vector<unsigned int, 3> cellId;
            cellId.SetX(cnt%(this->dimX-1));
            cellId.SetY((cnt/(this->dimX-1))%(this->dimY-1));
            cellId.SetZ((cnt/(this->dimX-1))/(this->dimY-1));

            if((cellId.X() >= 1)&&(cellId.Y() >= 1)&&(cellId.Z() >= 1)&&
                    (cellId.X() < this->dimX-2)&&(cellId.Y() < this->dimY-2)&&(cellId.Z() < this->dimZ-2)) {

                /*if((cellId.X() == 5)&&(cellId.Y() == 16)&&(cellId.Z() == 23)) {
                printf("cell %u %u %u, coords %f %f %f\n", cellId.X(),
                        cellId.Y(), cellId.Z(), cellCoords[3*cnt],
                        cellCoords[3*cnt+1], cellCoords[3*cnt+2]);
                }*/

                Vec3f pos;
                pos.SetX(this->orgX + (cellId.X()+ cellCoords[3*cnt+0]) * this->spacingX);
                pos.SetY(this->orgY + (cellId.Y()+ cellCoords[3*cnt+1]) * this->spacingY);
                pos.SetZ(this->orgZ + (cellId.Z()+ cellCoords[3*cnt+2]) * this->spacingZ);

                Vec3f posNewton = this->searchNullPointNewton(maxItNewton, pos, cellId, stepNewton, epsNewton);
                this->critPoints.Add(CritPoint(posNewton, cellId,
                        this->classifyCritPoint(cellId, posNewton)));

                //this->critPoints.Add(CritPoint(pos, cellId, this->classifyCritPoint(cellId, pos)));
            }
        }
    }

    delete[] cellCoords;
}


/* VecField3f::SetDim */
void VecField3f::SetData(const float *data, unsigned int dX, unsigned int dY,
        unsigned int dZ, float sx, float sy, float sz, float orgX, float orgY,
        float orgZ) {

    // TODO allocating memory every time --> slow?

    // Get rid of possible previous data
    if(this->data != NULL) delete[] this->data;

    this->dimX = dX;
    this->dimY = dY;
    this->dimZ = dZ;

    this->spacingX = sx;
    this->spacingY = sy;
    this->spacingZ = sz;

    this->orgX = orgX;
    this->orgY = orgY;
    this->orgZ = orgZ;

    this->data = new float[this->dimX*this->dimY*this->dimZ*3];
    memcpy(this->data, data, sizeof(float)*this->dimX*this->dimY*this->dimZ*3);
}


/* VecField3f::classifyCritPoint */
VecField3f::CritPoint::Type VecField3f::classifyCritPoint(Vec3u cellId,
        Vec3f pos) {

    //printf("Dims %u %u %u\n", this->dimX, this->dimY, this->dimZ);

    vislib::Array<Vec3f> n;
    n.SetCount(8);
    n[0] = this->GetAt(cellId.X()+0, cellId.Y()+0, cellId.Z()+0);
    n[1] = this->GetAt(cellId.X()+1, cellId.Y()+0, cellId.Z()+0);
    n[2] = this->GetAt(cellId.X()+0, cellId.Y()+1, cellId.Z()+0);
    n[3] = this->GetAt(cellId.X()+1, cellId.Y()+1, cellId.Z()+0);
    n[4] = this->GetAt(cellId.X()+0, cellId.Y()+0, cellId.Z()+1);
    n[5] = this->GetAt(cellId.X()+1, cellId.Y()+0, cellId.Z()+1);
    n[6] = this->GetAt(cellId.X()+0, cellId.Y()+1, cellId.Z()+1);
    n[7] = this->GetAt(cellId.X()+1, cellId.Y()+1, cellId.Z()+1);

    // Calc jacobian for all eight neighbours
    // Compute jacobian at cell corners
    Mat3f j[8];
    /*j[0] = this->GetJacobianAt(cellId.X()+0, cellId.Y()+0, cellId.Z()+0, true);
    j[1] = this->GetJacobianAt(cellId.X()+1, cellId.Y()+0, cellId.Z()+0, true);
    j[2] = this->GetJacobianAt(cellId.X()+0, cellId.Y()+1, cellId.Z()+0, true);
    j[3] = this->GetJacobianAt(cellId.X()+1, cellId.Y()+1, cellId.Z()+0, true);
    j[4] = this->GetJacobianAt(cellId.X()+0, cellId.Y()+0, cellId.Z()+1, true);
    j[5] = this->GetJacobianAt(cellId.X()+1, cellId.Y()+0, cellId.Z()+1, true);
    j[6] = this->GetJacobianAt(cellId.X()+0, cellId.Y()+1, cellId.Z()+1, true);
    j[7] = this->GetJacobianAt(cellId.X()+1, cellId.Y()+1, cellId.Z()+1, true);*/

    j[0] = Mat3f (
                (n[1].X()-n[0].X())/this->spacingX, // vx_dx
                (n[2].X()-n[0].X())/this->spacingY, // vx_dy
                (n[4].X()-n[0].X())/this->spacingZ, // vx_dz
                (n[1].Y()-n[0].Y())/this->spacingX, // vy_dx
                (n[2].Y()-n[0].Y())/this->spacingY, // vy_dy
                (n[4].Y()-n[0].Y())/this->spacingZ, // vy_dz
                (n[1].Z()-n[0].Z())/this->spacingX, // vz_dx
                (n[2].Z()-n[0].Z())/this->spacingY, // vz_dy
                (n[4].Z()-n[0].Z())/this->spacingZ  // vz_dz
                );

    j[1] = Mat3f (
                (n[1].X()-n[0].X())/this->spacingX, // vx_dx
                (n[3].X()-n[1].X())/this->spacingY, // vx_dy
                (n[5].X()-n[1].X())/this->spacingZ, // vx_dz
                (n[1].Y()-n[0].Y())/this->spacingX, // vy_dx
                (n[3].Y()-n[1].Y())/this->spacingY, // vy_dy
                (n[5].Y()-n[1].Y())/this->spacingZ, // vy_dz
                (n[1].Z()-n[0].Z())/this->spacingX, // vz_dx
                (n[3].Z()-n[1].Z())/this->spacingY, // vz_dy
                (n[5].Z()-n[1].Z())/this->spacingZ  // vz_dz
                );

    j[2] = Mat3f (
                (n[3].X()-n[2].X())/this->spacingX, // vx_dx
                (n[2].X()-n[0].X())/this->spacingY, // vx_dy
                (n[6].X()-n[2].X())/this->spacingZ, // vx_dz
                (n[3].Y()-n[2].Y())/this->spacingX, // vy_dx
                (n[2].Y()-n[0].Y())/this->spacingY, // vy_dy
                (n[6].Y()-n[2].Y())/this->spacingZ, // vy_dz
                (n[3].Z()-n[2].Z())/this->spacingX, // vz_dx
                (n[2].Z()-n[0].Z())/this->spacingY, // vz_dy
                (n[6].Z()-n[2].Z())/this->spacingZ  // vz_dz
                );

    j[3] = Mat3f (
                (n[3].X()-n[2].X())/this->spacingX, // vx_dx
                (n[3].X()-n[1].X())/this->spacingY, // vx_dy
                (n[7].X()-n[3].X())/this->spacingZ, // vx_dz
                (n[3].Y()-n[2].Y())/this->spacingX, // vy_dx
                (n[3].Y()-n[1].Y())/this->spacingY, // vy_dy
                (n[7].Y()-n[3].Y())/this->spacingZ, // vy_dz
                (n[3].Z()-n[2].Z())/this->spacingX, // vz_dx
                (n[3].Z()-n[1].Z())/this->spacingY, // vz_dy
                (n[7].Z()-n[3].Z())/this->spacingZ  // vz_dz
                );

    j[4] = Mat3f (
                (n[5].X()-n[4].X())/this->spacingX, // vx_dx
                (n[6].X()-n[4].X())/this->spacingY, // vx_dy
                (n[4].X()-n[0].X())/this->spacingZ, // vx_dz
                (n[5].Y()-n[4].Y())/this->spacingX, // vy_dx
                (n[6].Y()-n[4].Y())/this->spacingY, // vy_dy
                (n[4].Y()-n[0].Y())/this->spacingZ, // vy_dz
                (n[5].Z()-n[4].Z())/this->spacingX, // vz_dx
                (n[6].Z()-n[4].Z())/this->spacingY, // vz_dy
                (n[4].Z()-n[0].Z())/this->spacingZ  // vz_dz
                );

    j[5] = Mat3f (
                (n[5].X()-n[4].X())/this->spacingX, // vx_dx
                (n[7].X()-n[5].X())/this->spacingY, // vx_dy
                (n[5].X()-n[1].X())/this->spacingZ, // vx_dz
                (n[5].Y()-n[4].Y())/this->spacingX, // vy_dx
                (n[7].Y()-n[5].Y())/this->spacingY, // vy_dy
                (n[5].Y()-n[1].Y())/this->spacingZ, // vy_dz
                (n[5].Z()-n[4].Z())/this->spacingX, // vz_dx
                (n[7].Z()-n[5].Z())/this->spacingY, // vz_dy
                (n[5].Z()-n[1].Z())/this->spacingZ  // vz_dz
                );

    j[6] = Mat3f (
                (n[7].X()-n[6].X())/this->spacingX, // vx_dx
                (n[6].X()-n[4].X())/this->spacingY, // vx_dy
                (n[6].X()-n[2].X())/this->spacingZ, // vx_dz
                (n[7].Y()-n[6].Y())/this->spacingX, // vy_dx
                (n[6].Y()-n[4].Y())/this->spacingY, // vy_dy
                (n[6].Y()-n[2].Y())/this->spacingZ, // vy_dz
                (n[7].Z()-n[6].Z())/this->spacingX, // vz_dx
                (n[6].Z()-n[4].Z())/this->spacingY, // vz_dy
                (n[6].Z()-n[2].Z())/this->spacingZ  // vz_dz
                );

    j[7] = Mat3f (
                (n[7].X()-n[6].X())/this->spacingX, // vx_dx
                (n[7].X()-n[5].X())/this->spacingY, // vx_dy
                (n[7].X()-n[3].X())/this->spacingZ, // vx_dz
                (n[7].Y()-n[6].Y())/this->spacingX, // vy_dx
                (n[7].Y()-n[5].Y())/this->spacingY, // vy_dy
                (n[7].Y()-n[3].Y())/this->spacingZ, // vy_dz
                (n[7].Z()-n[6].Z())/this->spacingX, // vz_dx
                (n[7].Z()-n[5].Z())/this->spacingY, // vz_dy
                (n[7].Z()-n[3].Z())/this->spacingZ  // vz_dz
                );


    // Interpolate
    float alpha = (pos.X()-this->orgX)/this->spacingX;
    float beta  = (pos.Y()-this->orgY)/this->spacingY;
    float gamma = (pos.Z()-this->orgZ)/this->spacingZ;
    alpha = alpha - static_cast<unsigned int>(alpha);
    beta  = beta  - static_cast<unsigned int>(beta);
    gamma = gamma - static_cast<unsigned int>(gamma);
    /*Mat3f jac =
            UtilsNumerics::TrilinInterp<Mat3f >(j[0],
            j[1], j[2], j[3], j[4], j[5], j[6], j[7], alpha, beta, gamma);*/

    Mat3f jac;

    // Row #0
	jac.SetAt(0, 0, protein_calls::Interpol::Trilin<float>(j[0].GetAt(0, 0),
            j[1].GetAt(0, 0),j[2].GetAt(0, 0),j[3].GetAt(0, 0),
            j[4].GetAt(0, 0),j[5].GetAt(0, 0),j[6].GetAt(0, 0),
            j[7].GetAt(0, 0), alpha, beta, gamma));
	jac.SetAt(0, 1, protein_calls::Interpol::Trilin<float>(j[0].GetAt(0, 1),
            j[1].GetAt(0, 1),j[2].GetAt(0, 1),j[3].GetAt(0, 1),
            j[4].GetAt(0, 1),j[5].GetAt(0, 1),j[6].GetAt(0, 1),
            j[7].GetAt(0, 1), alpha, beta, gamma));
	jac.SetAt(0, 2, protein_calls::Interpol::Trilin<float>(j[0].GetAt(0, 2),
            j[1].GetAt(0, 2),j[2].GetAt(0, 2),j[3].GetAt(0, 2),
            j[4].GetAt(0, 2),j[5].GetAt(0, 2),j[6].GetAt(0, 2),
            j[7].GetAt(0, 2), alpha, beta, gamma));

    // Row #1
	jac.SetAt(1, 0, protein_calls::Interpol::Trilin<float>(j[0].GetAt(1, 0),
            j[1].GetAt(1, 0),j[2].GetAt(1, 0),j[3].GetAt(1, 0),
            j[4].GetAt(1, 0),j[5].GetAt(1, 0),j[6].GetAt(1, 0),
            j[7].GetAt(1, 0), alpha, beta, gamma));
	jac.SetAt(1, 1, protein_calls::Interpol::Trilin<float>(j[0].GetAt(1, 1),
            j[1].GetAt(1, 1),j[2].GetAt(1, 1),j[3].GetAt(1, 1),
            j[4].GetAt(1, 1),j[5].GetAt(1, 1),j[6].GetAt(1, 1),
            j[7].GetAt(1, 1), alpha, beta, gamma));
	jac.SetAt(1, 2, protein_calls::Interpol::Trilin<float>(j[0].GetAt(1, 2),
            j[1].GetAt(1, 2),j[2].GetAt(1, 2),j[3].GetAt(1, 2),
            j[4].GetAt(1, 2),j[5].GetAt(1, 2),j[6].GetAt(1, 2),
            j[7].GetAt(1, 2), alpha, beta, gamma));

    // Row #2
	jac.SetAt(2, 0, protein_calls::Interpol::Trilin<float>(j[0].GetAt(2, 0),
            j[1].GetAt(2, 0),j[2].GetAt(2, 0),j[3].GetAt(2, 0),
            j[4].GetAt(2, 0),j[5].GetAt(2, 0),j[6].GetAt(2, 0),
            j[7].GetAt(2, 0), alpha, beta, gamma));
	jac.SetAt(2, 1, protein_calls::Interpol::Trilin<float>(j[0].GetAt(2, 1),
            j[1].GetAt(2, 1),j[2].GetAt(2, 1),j[3].GetAt(2, 1),
            j[4].GetAt(2, 1),j[5].GetAt(2, 1),j[6].GetAt(2, 1),
            j[7].GetAt(2, 1), alpha, beta, gamma));
	jac.SetAt(2, 2, protein_calls::Interpol::Trilin<float>(j[0].GetAt(2, 2),
            j[1].GetAt(2, 2),j[2].GetAt(2, 2),j[3].GetAt(2, 2),
            j[4].GetAt(2, 2),j[5].GetAt(2, 2),j[6].GetAt(2, 2),
            j[7].GetAt(2, 2), alpha, beta, gamma));

    /*printf("-----------\n");
    printf("(%f %f %f)\n", jac.GetAt(0, 0), jac.GetAt(0, 1), jac.GetAt(0, 2));
    printf("(%f %f %f)\n", jac.GetAt(1, 0), jac.GetAt(1, 1), jac.GetAt(1, 2));
    printf("(%f %f %f)\n", jac.GetAt(2, 0), jac.GetAt(2, 1), jac.GetAt(2, 2));*/

    Mat3f jacSym, jacTrans;
    jacTrans = jac;
    jacTrans.Transpose();
    jacSym = (jac + jacTrans)/2.0f;

    /*printf("Symmetric part\n");
    printf("(%f %f %f)\n", jacSym.GetAt(0, 0), jacSym.GetAt(0, 1), jacSym.GetAt(0, 2));
    printf("(%f %f %f)\n", jacSym.GetAt(1, 0), jacSym.GetAt(1, 1), jacSym.GetAt(1, 2));
    printf("(%f %f %f)\n", jacSym.GetAt(2, 0), jacSym.GetAt(2, 1), jacSym.GetAt(2, 2));*/


    float *eigenvalues = new float[3];
    Vec3f *eigenvectors = new Vec3f[3];
    jacSym.FindEigenvalues(eigenvalues, eigenvectors, 3);
   // printf("Eigenvalues: %f %f %f\n", eigenvalues[0], eigenvalues[1], eigenvalues[2]);

    // Count how many positive negative eigenvalues there are
    unsigned int nPos=0, nNeg=0;
    for(int i = 0; i < 3; i++) {
        // TODO What happens if ev == 0 ?
        if(eigenvalues[i] >= 0) nPos++;
        else nNeg++;
    }

    delete[] eigenvalues;
    delete[] eigenvectors;

    if(nPos == 3) {
        //printf("(%f %f %f) SOURCE\n", crit.GetPosX(), crit.GetPosY(), crit.GetPosZ());
        return CritPoint::SOURCE;
    }
    if(nNeg == 3) {
        //printf("(%f %f %f) SINK\n", crit.GetPosX(), crit.GetPosY(), crit.GetPosZ());
        return CritPoint::SINK;
    }
    if((nPos == 2)&&(nNeg == 1)) {
        //printf("(%f %f %f) REPELLING_SADDLE\n", crit.GetPosX(), crit.GetPosY(), crit.GetPosZ());
        return CritPoint::REPELLING_SADDLE;
    }
    if((nNeg == 2)&&(nPos == 1)) {
        //printf("(%f %f %f) ATTRACTING_SADDLE\n", crit.GetPosX(), crit.GetPosY(), crit.GetPosZ());
        return CritPoint::ATTRACTING_SADDLE;
    }

    return CritPoint::UNKNOWN;
}


/* VecField3f::isFieldVanishingInCellBisectionRec */
bool VecField3f::isFieldVanishingInCellBisectionRec(unsigned int currDepth,
        unsigned int maxDepth, vislib::Array<Vec3f> n) {

    /*if(currDepth > 4) {
        printf("--Depth %u --------------------------\n", currDepth);
        printf("n0 (%.16f %.16f %.16f)\n", n[0].X(), n[0].Y(), n[0].Z());
        printf("n1 (%.16f %.16f %.16f)\n", n[1].X(), n[1].Y(), n[1].Z());
        printf("n2 (%.16f %.16f %.16f)\n", n[2].X(), n[2].Y(), n[2].Z());
        printf("n3 (%.16f %.16f %.16f)\n", n[3].X(), n[3].Y(), n[3].Z());
        printf("n4 (%.16f %.16f %.16f)\n", n[4].X(), n[4].Y(), n[4].Z());
        printf("n5 (%.16f %.16f %.16f)\n", n[5].X(), n[5].Y(), n[5].Z());
        printf("n6 (%.16f %.16f %.16f)\n", n[6].X(), n[6].Y(), n[6].Z());
        printf("n7 (%.16f %.16f %.16f)\n", n[7].X(), n[7].Y(), n[7].Z());
    }*/

    // Check whether one of the vector components has the same sign at all
    // eight corners
    bool flag = true;
    if((n[0].X() > 0)&&(n[1].X() > 0)&&(n[2].X() > 0)&&
            (n[3].X() > 0)&&(n[4].X() > 0)&&(n[5].X() > 0)&&
            (n[6].X() > 0)&&(n[7].X() > 0)) {

        flag = false;
    }
    if((n[0].X() < 0)&&(n[1].X() < 0)&&(n[2].X() < 0)&&
            (n[3].X() < 0)&&(n[4].X() < 0)&&(n[5].X() < 0)&&
            (n[6].X() < 0)&&(n[7].X() < 0)) {

        flag = false;
    }

    if((n[0].Y() > 0)&&(n[1].Y() > 0)&&(n[2].Y() > 0)&&
            (n[3].Y() > 0)&&(n[4].Y() > 0)&&(n[5].Y() > 0)&&
            (n[6].Y() > 0)&&(n[7].Y() > 0)) {

        flag = false;
    }
    if((n[0].Y() < 0)&&(n[1].Y() < 0)&&(n[2].Y() < 0)&&
            (n[3].Y() < 0)&&(n[4].Y() < 0)&&(n[5].Y() < 0)&&
            (n[6].Y() < 0)&&(n[7].Y() < 0)) {

        flag = false;
    }

    if((n[0].Z() > 0)&&(n[1].Z() > 0)&&(n[2].Z() > 0)&&
            (n[3].Z() > 0)&&(n[4].Z() > 0)&&(n[5].Z() > 0)&&
            (n[6].Z() > 0)&&(n[7].Z() > 0)) {
        flag = false;
    }
    if((n[0].Z() < 0)&&(n[1].Z() < 0)&&(n[2].Z() < 0)&&
            (n[3].Z() < 0)&&(n[4].Z() < 0)&&(n[5].Z() < 0)&&
            (n[6].Z() < 0)&&(n[7].Z() < 0)) {
        flag = false;
    }

    if(flag) {

        if(currDepth >= maxDepth) {
            //printf("Depth %u --> accepted\n", currDepth);
            return true;
        }

        // Bisect cell and recursively call function for all eight sub-cells

        vislib::Array<Vec3f> s, c;

        // Interpolate missing values
        s.SetCount(19);
        // Back
		s[0] = protein_calls::Interpol::Lin<Vec3f>(n[0], n[1], 0.5f);
		s[1] = protein_calls::Interpol::Lin<Vec3f>(n[0], n[2], 0.5f);
		s[2] = protein_calls::Interpol::Bilin<Vec3f>(n[0], n[1], n[2], n[3], 0.5f, 0.5f);
		s[3] = protein_calls::Interpol::Lin<Vec3f>(n[1], n[3], 0.5f);
		s[4] = protein_calls::Interpol::Lin<Vec3f>(n[2], n[3], 0.5f);
        // Middle
		s[5] = protein_calls::Interpol::Lin<Vec3f>(n[0], n[4], 0.5f);
		s[6] = protein_calls::Interpol::Bilin<Vec3f>(n[0], n[1], n[4], n[5], 0.5f, 0.5f);
		s[7] = protein_calls::Interpol::Lin<Vec3f>(n[1], n[5], 0.5f);
		s[8] = protein_calls::Interpol::Bilin<Vec3f>(n[0], n[2], n[4], n[6], 0.5f, 0.5f);
		s[9] = protein_calls::Interpol::Trilin<Vec3f>(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], 0.5f, 0.5f, 0.5f);
		s[10] = protein_calls::Interpol::Bilin<Vec3f>(n[1], n[3], n[5], n[7], 0.5f, 0.5f);
		s[11] = protein_calls::Interpol::Lin<Vec3f>(n[2], n[6], 0.5f);
		s[12] = protein_calls::Interpol::Bilin<Vec3f >(n[2], n[3], n[6], n[7], 0.5f, 0.5f);
		s[13] = protein_calls::Interpol::Lin<Vec3f>(n[3], n[7], 0.5f);
        // Front
		s[14] = protein_calls::Interpol::Lin<Vec3f>(n[4], n[5], 0.5f);
		s[15] = protein_calls::Interpol::Lin<Vec3f>(n[4], n[6], 0.5f);
		s[16] = protein_calls::Interpol::Bilin<Vec3f>(n[4], n[5], n[6], n[7], 0.5f, 0.5f);
		s[17] = protein_calls::Interpol::Lin<Vec3f>(n[5], n[7], 0.5f);
		s[18] = protein_calls::Interpol::Lin<Vec3f>(n[6], n[7], 0.5f);

        /*for(int i = 0; i < 19; i++) {
            printf("s[%i].Z(): %f\n", i, s[i].Z());
        }*/

        // test sub-cells
        Vec3f resCoord;
        c.SetCount(8);

        bool subFlags[8];
        for(int i = 0; i < 8; i++) subFlags[i] = false;

        //#0
        //printf("Depth %u --> subcell 0\n", currDepth);
        c[0] = n[0];
        c[1] = s[0];
        c[2] = s[1];
        c[3] = s[2];
        c[4] = s[5];
        c[5] = s[6];
        c[6] = s[8];
        c[7] = s[9];
        subFlags[0] = this->isFieldVanishingInCellBisectionRec(currDepth+1, maxDepth, c);

        //#1
        //printf("Depth %u --> subcell 1\n", currDepth);
        c[0] = s[0];
        c[1] = n[1];
        c[2] = s[2];
        c[3] = s[3];
        c[4] = s[6];
        c[5] = s[7];
        c[6] = s[9];
        c[7] = s[10];
        subFlags[1] = this->isFieldVanishingInCellBisectionRec(currDepth+1, maxDepth, c);

        //#2
        //printf("Depth %u --> subcell 2\n", currDepth);
        c[0] = s[1];
        c[1] = s[2];
        c[2] = n[2];
        c[3] = s[4];
        c[4] = s[8];
        c[5] = s[9];
        c[6] = s[11];
        c[7] = s[12];
        subFlags[2] = this->isFieldVanishingInCellBisectionRec(currDepth+1, maxDepth, c);

        //#3
        //printf("Depth %u --> subcell 3\n", currDepth);
        c[0] = s[2];
        c[1] = s[3];
        c[2] = s[4];
        c[3] = n[3];
        c[4] = s[9];
        c[5] = s[10];
        c[6] = s[12];
        c[7] = s[13];
        subFlags[3] = this->isFieldVanishingInCellBisectionRec(currDepth+1, maxDepth, c);

        //#4
        //printf("Depth %u --> subcell 4\n", currDepth);
        c[0] = s[5];
        c[1] = s[6];
        c[2] = s[8];
        c[3] = s[9];
        c[4] = n[4];
        c[5] = s[14];
        c[6] = s[15];
        c[7] = s[16];
        subFlags[4] = this->isFieldVanishingInCellBisectionRec(currDepth+1, maxDepth, c);

        //#5
        //printf("Depth %u --> subcell 5\n", currDepth);
        c[0] = s[6];
        c[1] = s[7];
        c[2] = s[9];
        c[3] = s[10];
        c[4] = s[14];
        c[5] = n[5];
        c[6] = s[16];
        c[7] = s[17];
        subFlags[5] = this->isFieldVanishingInCellBisectionRec(currDepth+1, maxDepth, c);

        //#6
        //printf("Depth %u --> subcell 6\n", currDepth);
        c[0] = s[8];
        c[1] = s[9];
        c[2] = s[11];
        c[3] = s[12];
        c[4] = s[15];
        c[5] = s[16];
        c[6] = n[6];
        c[7] = s[18];
        subFlags[6] = this->isFieldVanishingInCellBisectionRec(currDepth+1, maxDepth, c);

        //#7
        //printf("Depth %u --> subcell 7\n", currDepth);
        c[0] = s[9];
        c[1] = s[10];
        c[2] = s[12];
        c[3] = s[13];
        c[4] = s[16];
        c[5] = s[17];
        c[6] = s[18];
        c[7] = n[7];
        subFlags[7] = this->isFieldVanishingInCellBisectionRec(currDepth+1, maxDepth, c);

        return (subFlags[0])||(subFlags[1])||(subFlags[2])||(subFlags[3])||
                (subFlags[4])||(subFlags[5])||(subFlags[6])||(subFlags[7]);
    }

    // This cell does not contain a null point
    return false;
}


/* VecField3f::searchNullPointNewton */
Vec3f VecField3f::searchNullPointNewton(
        unsigned int maxIt,
        Vec3f startPos,
        Vec3u cellId,
        float step, float eps) {

    //printf("NEWTON\n");

    bool debug = false;
    //if((cellId.X() == 19)&&(cellId.Y() == 10)&&(cellId.Z() == 28)) {
    //    debug = true;
    //}

    // Get minimum spacing
    float minSpacing;
    if(this->spacingX < this->spacingY) {
        minSpacing = this->spacingX;
    }
    else {
        minSpacing = this->spacingY;
    }
    if(this->spacingZ < minSpacing) {
        minSpacing = this->spacingZ;
    }

    if (debug) printf("--> Neighbour values\n");
    // Get corner vals
    vislib::Array<Vec3f> n;
    n.SetCount(8);
    n[0] = this->GetAt(cellId.X()+0, cellId.Y()+0, cellId.Z()+0);
    n[1] = this->GetAt(cellId.X()+1, cellId.Y()+0, cellId.Z()+0);
    n[2] = this->GetAt(cellId.X()+0, cellId.Y()+1, cellId.Z()+0);
    n[3] = this->GetAt(cellId.X()+1, cellId.Y()+1, cellId.Z()+0);
    n[4] = this->GetAt(cellId.X()+0, cellId.Y()+0, cellId.Z()+1);
    n[5] = this->GetAt(cellId.X()+1, cellId.Y()+0, cellId.Z()+1);
    n[6] = this->GetAt(cellId.X()+0, cellId.Y()+1, cellId.Z()+1);
    n[7] = this->GetAt(cellId.X()+1, cellId.Y()+1, cellId.Z()+1);
    for(int i = 0; i < 8; i++) {
        if (debug) printf("(%f %f %f)\n", n[i].X(), n[i].Y(), n[i].Z());
    }
    for (int i = 0; i < 8; i++) n[i].Normalise();


    // Compute jacobian at cell corners
    Mat3f j[8];
    /*j[0] = this->GetJacobianAt(cellId.X()+0, cellId.Y()+0, cellId.Z()+0, true);
    j[1] = this->GetJacobianAt(cellId.X()+1, cellId.Y()+0, cellId.Z()+0, true);
    j[2] = this->GetJacobianAt(cellId.X()+0, cellId.Y()+1, cellId.Z()+0, true);
    j[3] = this->GetJacobianAt(cellId.X()+1, cellId.Y()+1, cellId.Z()+0, true);
    j[4] = this->GetJacobianAt(cellId.X()+0, cellId.Y()+0, cellId.Z()+1, true);
    j[5] = this->GetJacobianAt(cellId.X()+1, cellId.Y()+0, cellId.Z()+1, true);
    j[6] = this->GetJacobianAt(cellId.X()+0, cellId.Y()+1, cellId.Z()+1, true);
    j[7] = this->GetJacobianAt(cellId.X()+1, cellId.Y()+1, cellId.Z()+1, true);*/

    j[0] = Mat3f (
                (n[1].X()-n[0].X())/this->spacingX, // vx_dx
                (n[2].X()-n[0].X())/this->spacingY, // vx_dy
                (n[4].X()-n[0].X())/this->spacingZ, // vx_dz
                (n[1].Y()-n[0].Y())/this->spacingX, // vy_dx
                (n[2].Y()-n[0].Y())/this->spacingY, // vy_dy
                (n[4].Y()-n[0].Y())/this->spacingZ, // vy_dz
                (n[1].Z()-n[0].Z())/this->spacingX, // vz_dx
                (n[2].Z()-n[0].Z())/this->spacingY, // vz_dy
                (n[4].Z()-n[0].Z())/this->spacingZ  // vz_dz
                );

    j[1] = Mat3f (
                (n[1].X()-n[0].X())/this->spacingX, // vx_dx
                (n[3].X()-n[1].X())/this->spacingY, // vx_dy
                (n[5].X()-n[1].X())/this->spacingZ, // vx_dz
                (n[1].Y()-n[0].Y())/this->spacingX, // vy_dx
                (n[3].Y()-n[1].Y())/this->spacingY, // vy_dy
                (n[5].Y()-n[1].Y())/this->spacingZ, // vy_dz
                (n[1].Z()-n[0].Z())/this->spacingX, // vz_dx
                (n[3].Z()-n[1].Z())/this->spacingY, // vz_dy
                (n[5].Z()-n[1].Z())/this->spacingZ  // vz_dz
                );

    j[2] = Mat3f (
                (n[3].X()-n[2].X())/this->spacingX, // vx_dx
                (n[2].X()-n[0].X())/this->spacingY, // vx_dy
                (n[6].X()-n[2].X())/this->spacingZ, // vx_dz
                (n[3].Y()-n[2].Y())/this->spacingX, // vy_dx
                (n[2].Y()-n[0].Y())/this->spacingY, // vy_dy
                (n[6].Y()-n[2].Y())/this->spacingZ, // vy_dz
                (n[3].Z()-n[2].Z())/this->spacingX, // vz_dx
                (n[2].Z()-n[0].Z())/this->spacingY, // vz_dy
                (n[6].Z()-n[2].Z())/this->spacingZ  // vz_dz
                );

    j[3] = Mat3f (
                (n[3].X()-n[2].X())/this->spacingX, // vx_dx
                (n[3].X()-n[1].X())/this->spacingY, // vx_dy
                (n[7].X()-n[3].X())/this->spacingZ, // vx_dz
                (n[3].Y()-n[2].Y())/this->spacingX, // vy_dx
                (n[3].Y()-n[1].Y())/this->spacingY, // vy_dy
                (n[7].Y()-n[3].Y())/this->spacingZ, // vy_dz
                (n[3].Z()-n[2].Z())/this->spacingX, // vz_dx
                (n[3].Z()-n[1].Z())/this->spacingY, // vz_dy
                (n[7].Z()-n[3].Z())/this->spacingZ  // vz_dz
                );

    j[4] = Mat3f (
                (n[5].X()-n[4].X())/this->spacingX, // vx_dx
                (n[6].X()-n[4].X())/this->spacingY, // vx_dy
                (n[4].X()-n[0].X())/this->spacingZ, // vx_dz
                (n[5].Y()-n[4].Y())/this->spacingX, // vy_dx
                (n[6].Y()-n[4].Y())/this->spacingY, // vy_dy
                (n[4].Y()-n[0].Y())/this->spacingZ, // vy_dz
                (n[5].Z()-n[4].Z())/this->spacingX, // vz_dx
                (n[6].Z()-n[4].Z())/this->spacingY, // vz_dy
                (n[4].Z()-n[0].Z())/this->spacingZ  // vz_dz
                );

    j[5] = Mat3f (
                (n[5].X()-n[4].X())/this->spacingX, // vx_dx
                (n[7].X()-n[5].X())/this->spacingY, // vx_dy
                (n[5].X()-n[1].X())/this->spacingZ, // vx_dz
                (n[5].Y()-n[4].Y())/this->spacingX, // vy_dx
                (n[7].Y()-n[5].Y())/this->spacingY, // vy_dy
                (n[5].Y()-n[1].Y())/this->spacingZ, // vy_dz
                (n[5].Z()-n[4].Z())/this->spacingX, // vz_dx
                (n[7].Z()-n[5].Z())/this->spacingY, // vz_dy
                (n[5].Z()-n[1].Z())/this->spacingZ  // vz_dz
                );

    j[6] = Mat3f (
                (n[7].X()-n[6].X())/this->spacingX, // vx_dx
                (n[6].X()-n[4].X())/this->spacingY, // vx_dy
                (n[6].X()-n[2].X())/this->spacingZ, // vx_dz
                (n[7].Y()-n[6].Y())/this->spacingX, // vy_dx
                (n[6].Y()-n[4].Y())/this->spacingY, // vy_dy
                (n[6].Y()-n[2].Y())/this->spacingZ, // vy_dz
                (n[7].Z()-n[6].Z())/this->spacingX, // vz_dx
                (n[6].Z()-n[4].Z())/this->spacingY, // vz_dy
                (n[6].Z()-n[2].Z())/this->spacingZ  // vz_dz
                );

    j[7] = Mat3f (
                (n[7].X()-n[6].X())/this->spacingX, // vx_dx
                (n[7].X()-n[5].X())/this->spacingY, // vx_dy
                (n[7].X()-n[3].X())/this->spacingZ, // vx_dz
                (n[7].Y()-n[6].Y())/this->spacingX, // vy_dx
                (n[7].Y()-n[5].Y())/this->spacingY, // vy_dy
                (n[7].Y()-n[3].Y())/this->spacingZ, // vy_dz
                (n[7].Z()-n[6].Z())/this->spacingX, // vz_dx
                (n[7].Z()-n[5].Z())/this->spacingY, // vz_dy
                (n[7].Z()-n[3].Z())/this->spacingZ  // vz_dz
                );


    /*if (debug) printf("--> Jacobi corners\n");
    for(int i = 0; i < 8; i++) {
        if (debug) printf("(%f %f %f)\n", j[i].GetAt(0, 0), j[i].GetAt(0, 1), j[i].GetAt(0, 2));
        if (debug) printf("(%f %f %f)\n", j[i].GetAt(1, 0), j[i].GetAt(1, 1), j[i].GetAt(1, 2));
        if (debug) printf("(%f %f %f)\n", j[i].GetAt(2, 0), j[i].GetAt(2, 1), j[i].GetAt(2, 2));
        if (debug) printf("\n");
    }*/

    /*j[0] = this->GetJacobianAt(cellId.X()+0, cellId.Y()+0, cellId.Z()+0);
    j[1] = this->GetJacobianAt(cellId.X()+1, cellId.Y()+0, cellId.Z()+0);
    j[2] = this->GetJacobianAt(cellId.X()+0, cellId.Y()+1, cellId.Z()+0);
    j[3] = this->GetJacobianAt(cellId.X()+1, cellId.Y()+1, cellId.Z()+0);
    j[4] = this->GetJacobianAt(cellId.X()+0, cellId.Y()+0, cellId.Z()+1);
    j[5] = this->GetJacobianAt(cellId.X()+1, cellId.Y()+0, cellId.Z()+1);
    j[6] = this->GetJacobianAt(cellId.X()+0, cellId.Y()+1, cellId.Z()+1);
    j[7] = this->GetJacobianAt(cellId.X()+1, cellId.Y()+1, cellId.Z()+1);*/

    Vec3f startPosArr[9];
    Vec3f dPos, pos;
    Vec3f v, dv;
    Mat3f jac;

    // Possible starting positions
    float offs = minSpacing*0.2f; // We do not want to start exactly in the cell corners
    startPosArr[0] = startPos;
    startPosArr[1] = Vec3f(
            this->orgX + static_cast<float>(cellId.X()+1-offs)*this->spacingX,
            this->orgY + static_cast<float>(cellId.Y()+1-offs)*this->spacingY,
            this->orgZ + static_cast<float>(cellId.Z()+1-offs)*this->spacingZ);
    startPosArr[2] = Vec3f(
            this->orgX + static_cast<float>(cellId.X()+1-offs)*this->spacingX,
            this->orgY + static_cast<float>(cellId.Y()+1-offs)*this->spacingY,
            this->orgZ + static_cast<float>(cellId.Z()+offs)*this->spacingZ);
    startPosArr[3] = Vec3f(
            this->orgX + static_cast<float>(cellId.X()+1-offs)*this->spacingX,
            this->orgY + static_cast<float>(cellId.Y()+offs)*this->spacingY,
            this->orgZ + static_cast<float>(cellId.Z()+1-offs)*this->spacingZ);
    startPosArr[4] = Vec3f(
            this->orgX + static_cast<float>(cellId.X()+1-offs)*this->spacingX,
            this->orgY + static_cast<float>(cellId.Y()+offs)*this->spacingY,
            this->orgZ + static_cast<float>(cellId.Z()+offs)*this->spacingZ);
    startPosArr[5] = Vec3f(
            this->orgX + static_cast<float>(cellId.X()+offs)*this->spacingX,
            this->orgY + static_cast<float>(cellId.Y()+1-offs)*this->spacingY,
            this->orgZ + static_cast<float>(cellId.Z()+1-offs)*this->spacingZ);
    startPosArr[6] = Vec3f(
            this->orgX + static_cast<float>(cellId.X()+offs)*this->spacingX,
            this->orgY + static_cast<float>(cellId.Y()+1-offs)*this->spacingY,
            this->orgZ + static_cast<float>(cellId.Z()+offs)*this->spacingZ);
    startPosArr[7] = Vec3f(
            this->orgX + static_cast<float>(cellId.X()+offs)*this->spacingX,
            this->orgY + static_cast<float>(cellId.Y()+offs)*this->spacingY,
            this->orgZ + static_cast<float>(cellId.Z()+1-offs)*this->spacingZ);
    startPosArr[8] = Vec3f(
            this->orgX + static_cast<float>(cellId.X()+offs)*this->spacingX,
            this->orgY + static_cast<float>(cellId.Y()+offs)*this->spacingY,
            this->orgZ + static_cast<float>(cellId.Z()+offs)*this->spacingZ);


    // Loop through all startpoints until the null point is approximated
    //for(int i = 0; i < 1; i++) {
        unsigned int it = 0, posCnt = 1;
        pos = startPosArr[0];
        v = this->GetAtTrilin(pos.X(), pos.Y(), pos.Z(), true);
        //while((v.Norm() > eps)&&(it < maxIt)) {
        while(it < maxIt) {

            if (debug) printf("------------------------------\n");

            //Vec3f v0 = this->GetAtTrilin(pos.X()+offs, pos.Y(), pos.Z(), true);
            //Vec3f v1 = this->GetAtTrilin(pos.X()-offs, pos.Y(), pos.Z(), true);
            //if (debug)  printf("vxdx: %f\n", (v0.X()-v1.X())/(2.0*offs));

             if (debug)
                 printf("it = %u, sampled vec: %f %f %f, norm %f, eps %f\n",
                    it, v.X(), v.Y(), v.Z(), v.Norm(), eps);

            // Sample jacobian at pos using trilinear interpolation
            float alpha = (pos.X()-this->orgX)/this->spacingX;
            float beta  = (pos.Y()-this->orgY)/this->spacingY;
            float gamma = (pos.Z()-this->orgZ)/this->spacingZ;
            alpha = alpha - static_cast<unsigned int>(alpha);
            beta  = beta  - static_cast<unsigned int>(beta);
            gamma = gamma - static_cast<unsigned int>(gamma);
            /*jac = Interpol::Trilin<Mat3f >(
                    j[0], j[1], j[2], j[3], j[4], j[5],
                    j[6], j[7], alpha, beta, gamma);*/

            // Row #0
			jac.SetAt(0, 0, protein_calls::Interpol::Trilin<float>(j[0].GetAt(0, 0),
                    j[1].GetAt(0, 0),j[2].GetAt(0, 0),j[3].GetAt(0, 0),
                    j[4].GetAt(0, 0),j[5].GetAt(0, 0),j[6].GetAt(0, 0),
                    j[7].GetAt(0, 0), alpha, beta, gamma));
			jac.SetAt(0, 1, protein_calls::Interpol::Trilin<float>(j[0].GetAt(0, 1),
                    j[1].GetAt(0, 1),j[2].GetAt(0, 1),j[3].GetAt(0, 1),
                    j[4].GetAt(0, 1),j[5].GetAt(0, 1),j[6].GetAt(0, 1),
                    j[7].GetAt(0, 1), alpha, beta, gamma));
			jac.SetAt(0, 2, protein_calls::Interpol::Trilin<float>(j[0].GetAt(0, 2),
                    j[1].GetAt(0, 2),j[2].GetAt(0, 2),j[3].GetAt(0, 2),
                    j[4].GetAt(0, 2),j[5].GetAt(0, 2),j[6].GetAt(0, 2),
                    j[7].GetAt(0, 2), alpha, beta, gamma));

            // Row #1
			jac.SetAt(1, 0, protein_calls::Interpol::Trilin<float>(j[0].GetAt(1, 0),
                    j[1].GetAt(1, 0),j[2].GetAt(1, 0),j[3].GetAt(1, 0),
                    j[4].GetAt(1, 0),j[5].GetAt(1, 0),j[6].GetAt(1, 0),
                    j[7].GetAt(1, 0), alpha, beta, gamma));
			jac.SetAt(1, 1, protein_calls::Interpol::Trilin<float>(j[0].GetAt(1, 1),
                    j[1].GetAt(1, 1),j[2].GetAt(1, 1),j[3].GetAt(1, 1),
                    j[4].GetAt(1, 1),j[5].GetAt(1, 1),j[6].GetAt(1, 1),
                    j[7].GetAt(1, 1), alpha, beta, gamma));
			jac.SetAt(1, 2, protein_calls::Interpol::Trilin<float>(j[0].GetAt(1, 2),
                    j[1].GetAt(1, 2),j[2].GetAt(1, 2),j[3].GetAt(1, 2),
                    j[4].GetAt(1, 2),j[5].GetAt(1, 2),j[6].GetAt(1, 2),
                    j[7].GetAt(1, 2), alpha, beta, gamma));

            // Row #2
			jac.SetAt(2, 0, protein_calls::Interpol::Trilin<float>(j[0].GetAt(2, 0),
                    j[1].GetAt(2, 0),j[2].GetAt(2, 0),j[3].GetAt(2, 0),
                    j[4].GetAt(2, 0),j[5].GetAt(2, 0),j[6].GetAt(2, 0),
                    j[7].GetAt(2, 0), alpha, beta, gamma));
			jac.SetAt(2, 1, protein_calls::Interpol::Trilin<float>(j[0].GetAt(2, 1),
                    j[1].GetAt(2, 1),j[2].GetAt(2, 1),j[3].GetAt(2, 1),
                    j[4].GetAt(2, 1),j[5].GetAt(2, 1),j[6].GetAt(2, 1),
                    j[7].GetAt(2, 1), alpha, beta, gamma));
			jac.SetAt(2, 2, protein_calls::Interpol::Trilin<float>(j[0].GetAt(2, 2),
                    j[1].GetAt(2, 2),j[2].GetAt(2, 2),j[3].GetAt(2, 2),
                    j[4].GetAt(2, 2),j[5].GetAt(2, 2),j[6].GetAt(2, 2),
                    j[7].GetAt(2, 2), alpha, beta, gamma));


            if (debug) printf("NORMAL\n");
            if (debug) printf("(%f %f %f)\n", jac.GetAt(0, 0), jac.GetAt(0, 1), jac.GetAt(0, 2));
            if (debug) printf("(%f %f %f)\n", jac.GetAt(1, 0), jac.GetAt(1, 1), jac.GetAt(1, 2));
            if (debug) printf("(%f %f %f)\n", jac.GetAt(2, 0), jac.GetAt(2, 1), jac.GetAt(2, 2));

            if(!jac.Invert()) {
                printf("Matrix invert failed!\n");
            }

            dPos = jac*v;
            //dPos.Normalise();

            if (debug) printf("INVERSE cellID %u %u %u, dPos norm %f, alpha %f , beta %f, gamma %f\n",
                    cellId.X(), cellId.Y(), cellId.Z(), dPos.Norm(), alpha, beta, gamma);
            if (debug) printf("(%f %f %f)\n", jac.GetAt(0, 0), jac.GetAt(0, 1), jac.GetAt(0, 2));
            if (debug) printf("(%f %f %f)\n", jac.GetAt(1, 0), jac.GetAt(1, 1), jac.GetAt(1, 2));
            if (debug) printf("(%f %f %f)\n", jac.GetAt(2, 0), jac.GetAt(2, 1), jac.GetAt(2, 2));

            // Update pos using the jacobian and the sampled vector field
            pos = pos - dPos*minSpacing*step;

            // Check whether we went outside the cell
            if(!this->IsPosInCell(cellId, pos)) {
                if(posCnt > 8) break;
                if(posCnt == 8) {
                    printf("acht\n");
                }
                pos = startPosArr[posCnt];
                posCnt++;
                it = 0;
            }
            else {
                it++;
            }

            // Sample field at the new position
            v = this->GetAtTrilin(pos.X(), pos.Y(), pos.Z(), true);

        }
       // if((v.Norm() <= eps)&&(this->IsPosInCell(cellId, pos))) break;
    //}

    if(this->IsValidGridpos(pos)) {
        if (v.Norm() > eps) {
            printf("Newton not ready in cell %u %u %u (eps=%f, norm=%f)\n",
                    cellId.X(), cellId.Y(),cellId.Z(), eps, v.Norm());
        }
        return pos;
    }
    else {
        printf("Newton not converging in cell %u %u %u\n", cellId.X(),
                cellId.Y(),cellId.Z());
        return startPos;
    }
}




