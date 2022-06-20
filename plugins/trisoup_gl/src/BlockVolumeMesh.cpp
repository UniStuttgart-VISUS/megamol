/*
 * BlockVolumeMesh.h
 *
 * Copyright (C) 2010 by Sebastian Grottel
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "BlockVolumeMesh.h"
#include "mmcore/utility/log/Log.h"
#include "trisoup/CallBinaryVolumeData.h"
#include "vislib/assert.h"

using namespace megamol;
using namespace megamol::trisoup_gl;


/*
 * BlockVolumeMesh::BlockVolumeMesh
 */
BlockVolumeMesh::BlockVolumeMesh(void)
        : AbstractTriMeshDataSource()
        , inDataSlot("indata", "Slot fetching binary volume data")
        , inDataHash(0) {

    this->inDataSlot.SetCompatibleCall<trisoup::CallBinaryVolumeDataDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);
}


/*
 * BlockVolumeMesh::~BlockVolumeMesh
 */
BlockVolumeMesh::~BlockVolumeMesh(void) {
    this->Release();
    ASSERT(this->objs.IsEmpty());
    ASSERT(this->mats.IsEmpty());
}


/*
 * BlockVolumeMesh::assertData
 */
void BlockVolumeMesh::assertData(void) {

    trisoup::CallBinaryVolumeData* cbvd = this->inDataSlot.CallAs<trisoup::CallBinaryVolumeData>();
    if (cbvd == NULL)
        return;
    if (!(*cbvd)(0))
        return;

    if ((this->inDataHash == cbvd->DataHash()) && (this->inDataHash != 0)) {
        cbvd->Unlock(); // data has not changed
        return;
    }

    // new data!
    this->inDataHash = cbvd->DataHash();
    const bool* volume = cbvd->GetVolume();
    unsigned int cntX = cbvd->GetSizeX(), cntY = cbvd->GetSizeY(), cntZ = cbvd->GetSizeZ();
    float sizeX = cbvd->GetVoxelSizeX(), sizeY = cbvd->GetVoxelSizeY(), sizeZ = cbvd->GetVoxelSizeZ();

    this->bbox.Set(0.0f, 0.0f, 0.0f, sizeX * cntX, sizeY * cntY, sizeZ * cntZ);
    this->mats.Clear();
    this->objs.Clear();

    unsigned int fCnt = 0;
    for (unsigned int z = 0; z < cntZ; z++) {
        for (unsigned int y = 0; y < cntY; y++) {
            for (unsigned int x = 0; x < cntX; x++) {
                if (!volume[x + (y + z * cntY) * cntZ])
                    continue;

                if ((x == 0) || !volume[(x - 1) + (y + z * cntY) * cntZ])
                    fCnt++;
                if ((x + 1 == cntX) || !volume[(x + 1) + (y + z * cntY) * cntZ])
                    fCnt++;
                if ((y == 0) || !volume[x + ((y - 1) + z * cntY) * cntZ])
                    fCnt++;
                if ((y + 1 == cntY) || !volume[x + ((y + 1) + z * cntY) * cntZ])
                    fCnt++;
                if ((z == 0) || !volume[x + (y + (z - 1) * cntY) * cntZ])
                    fCnt++;
                if ((z + 1 == cntZ) || !volume[x + (y + (z + 1) * cntY) * cntZ])
                    fCnt++;
            }
        }
    }

    float* v = new float[3 * 4 * fCnt];
    float* n = new float[3 * 4 * fCnt];
    fCnt = 0;
    for (unsigned int z = 0; z < cntZ; z++) {
        for (unsigned int y = 0; y < cntY; y++) {
            for (unsigned int x = 0; x < cntX; x++) {
                if (!volume[x + (y + z * cntY) * cntZ])
                    continue;

                if ((x == 0) || !volume[(x - 1) + (y + z * cntY) * cntZ]) {
                    v[(fCnt * 4 + 0) * 3 + 0] = x * sizeX;
                    v[(fCnt * 4 + 0) * 3 + 1] = y * sizeY;
                    v[(fCnt * 4 + 0) * 3 + 2] = z * sizeZ;
                    v[(fCnt * 4 + 1) * 3 + 0] = x * sizeX;
                    v[(fCnt * 4 + 1) * 3 + 1] = y * sizeY;
                    v[(fCnt * 4 + 1) * 3 + 2] = (z + 1) * sizeZ;
                    v[(fCnt * 4 + 2) * 3 + 0] = x * sizeX;
                    v[(fCnt * 4 + 2) * 3 + 1] = (y + 1) * sizeY;
                    v[(fCnt * 4 + 2) * 3 + 2] = z * sizeZ;
                    v[(fCnt * 4 + 3) * 3 + 0] = x * sizeX;
                    v[(fCnt * 4 + 3) * 3 + 1] = (y + 1) * sizeY;
                    v[(fCnt * 4 + 3) * 3 + 2] = (z + 1) * sizeZ;
                    n[(fCnt * 4 + 0) * 3 + 0] = -1.0f;
                    n[(fCnt * 4 + 0) * 3 + 1] = 0.0f;
                    n[(fCnt * 4 + 0) * 3 + 2] = 0.0f;
                    n[(fCnt * 4 + 1) * 3 + 0] = n[(fCnt * 4 + 0) * 3 + 0];
                    n[(fCnt * 4 + 1) * 3 + 1] = n[(fCnt * 4 + 0) * 3 + 1];
                    n[(fCnt * 4 + 1) * 3 + 2] = n[(fCnt * 4 + 0) * 3 + 2];
                    n[(fCnt * 4 + 2) * 3 + 0] = n[(fCnt * 4 + 0) * 3 + 0];
                    n[(fCnt * 4 + 2) * 3 + 1] = n[(fCnt * 4 + 0) * 3 + 1];
                    n[(fCnt * 4 + 2) * 3 + 2] = n[(fCnt * 4 + 0) * 3 + 2];
                    n[(fCnt * 4 + 3) * 3 + 0] = n[(fCnt * 4 + 0) * 3 + 0];
                    n[(fCnt * 4 + 3) * 3 + 1] = n[(fCnt * 4 + 0) * 3 + 1];
                    n[(fCnt * 4 + 3) * 3 + 2] = n[(fCnt * 4 + 0) * 3 + 2];
                    fCnt++;
                }
                if ((x + 1 == cntX) || !volume[(x + 1) + (y + z * cntY) * cntZ]) {
                    v[(fCnt * 4 + 0) * 3 + 0] = (x + 1) * sizeX;
                    v[(fCnt * 4 + 0) * 3 + 1] = y * sizeY;
                    v[(fCnt * 4 + 0) * 3 + 2] = z * sizeZ;
                    v[(fCnt * 4 + 1) * 3 + 0] = (x + 1) * sizeX;
                    v[(fCnt * 4 + 1) * 3 + 1] = (y + 1) * sizeY;
                    v[(fCnt * 4 + 1) * 3 + 2] = z * sizeZ;
                    v[(fCnt * 4 + 2) * 3 + 0] = (x + 1) * sizeX;
                    v[(fCnt * 4 + 2) * 3 + 1] = y * sizeY;
                    v[(fCnt * 4 + 2) * 3 + 2] = (z + 1) * sizeZ;
                    v[(fCnt * 4 + 3) * 3 + 0] = (x + 1) * sizeX;
                    v[(fCnt * 4 + 3) * 3 + 1] = (y + 1) * sizeY;
                    v[(fCnt * 4 + 3) * 3 + 2] = (z + 1) * sizeZ;
                    n[(fCnt * 4 + 0) * 3 + 0] = 1.0f;
                    n[(fCnt * 4 + 0) * 3 + 1] = 0.0f;
                    n[(fCnt * 4 + 0) * 3 + 2] = 0.0f;
                    n[(fCnt * 4 + 1) * 3 + 0] = n[(fCnt * 4 + 0) * 3 + 0];
                    n[(fCnt * 4 + 1) * 3 + 1] = n[(fCnt * 4 + 0) * 3 + 1];
                    n[(fCnt * 4 + 1) * 3 + 2] = n[(fCnt * 4 + 0) * 3 + 2];
                    n[(fCnt * 4 + 2) * 3 + 0] = n[(fCnt * 4 + 0) * 3 + 0];
                    n[(fCnt * 4 + 2) * 3 + 1] = n[(fCnt * 4 + 0) * 3 + 1];
                    n[(fCnt * 4 + 2) * 3 + 2] = n[(fCnt * 4 + 0) * 3 + 2];
                    n[(fCnt * 4 + 3) * 3 + 0] = n[(fCnt * 4 + 0) * 3 + 0];
                    n[(fCnt * 4 + 3) * 3 + 1] = n[(fCnt * 4 + 0) * 3 + 1];
                    n[(fCnt * 4 + 3) * 3 + 2] = n[(fCnt * 4 + 0) * 3 + 2];
                    fCnt++;
                }
                if ((y == 0) || !volume[x + ((y - 1) + z * cntY) * cntZ]) {
                    v[(fCnt * 4 + 0) * 3 + 0] = x * sizeX;
                    v[(fCnt * 4 + 0) * 3 + 1] = y * sizeY;
                    v[(fCnt * 4 + 0) * 3 + 2] = z * sizeZ;
                    v[(fCnt * 4 + 1) * 3 + 0] = (x + 1) * sizeX;
                    v[(fCnt * 4 + 1) * 3 + 1] = y * sizeY;
                    v[(fCnt * 4 + 1) * 3 + 2] = z * sizeZ;
                    v[(fCnt * 4 + 2) * 3 + 0] = x * sizeX;
                    v[(fCnt * 4 + 2) * 3 + 1] = y * sizeY;
                    v[(fCnt * 4 + 2) * 3 + 2] = (z + 1) * sizeZ;
                    v[(fCnt * 4 + 3) * 3 + 0] = (x + 1) * sizeX;
                    v[(fCnt * 4 + 3) * 3 + 1] = y * sizeY;
                    v[(fCnt * 4 + 3) * 3 + 2] = (z + 1) * sizeZ;
                    n[(fCnt * 4 + 0) * 3 + 0] = 0.0f;
                    n[(fCnt * 4 + 0) * 3 + 1] = -1.0f;
                    n[(fCnt * 4 + 0) * 3 + 2] = 0.0f;
                    n[(fCnt * 4 + 1) * 3 + 0] = n[(fCnt * 4 + 0) * 3 + 0];
                    n[(fCnt * 4 + 1) * 3 + 1] = n[(fCnt * 4 + 0) * 3 + 1];
                    n[(fCnt * 4 + 1) * 3 + 2] = n[(fCnt * 4 + 0) * 3 + 2];
                    n[(fCnt * 4 + 2) * 3 + 0] = n[(fCnt * 4 + 0) * 3 + 0];
                    n[(fCnt * 4 + 2) * 3 + 1] = n[(fCnt * 4 + 0) * 3 + 1];
                    n[(fCnt * 4 + 2) * 3 + 2] = n[(fCnt * 4 + 0) * 3 + 2];
                    n[(fCnt * 4 + 3) * 3 + 0] = n[(fCnt * 4 + 0) * 3 + 0];
                    n[(fCnt * 4 + 3) * 3 + 1] = n[(fCnt * 4 + 0) * 3 + 1];
                    n[(fCnt * 4 + 3) * 3 + 2] = n[(fCnt * 4 + 0) * 3 + 2];
                    fCnt++;
                }
                if ((y + 1 == cntY) || !volume[x + ((y + 1) + z * cntY) * cntZ]) {
                    v[(fCnt * 4 + 0) * 3 + 0] = x * sizeX;
                    v[(fCnt * 4 + 0) * 3 + 1] = (y + 1) * sizeY;
                    v[(fCnt * 4 + 0) * 3 + 2] = z * sizeZ;
                    v[(fCnt * 4 + 1) * 3 + 0] = x * sizeX;
                    v[(fCnt * 4 + 1) * 3 + 1] = (y + 1) * sizeY;
                    v[(fCnt * 4 + 1) * 3 + 2] = (z + 1) * sizeZ;
                    v[(fCnt * 4 + 2) * 3 + 0] = (x + 1) * sizeX;
                    v[(fCnt * 4 + 2) * 3 + 1] = (y + 1) * sizeY;
                    v[(fCnt * 4 + 2) * 3 + 2] = z * sizeZ;
                    v[(fCnt * 4 + 3) * 3 + 0] = (x + 1) * sizeX;
                    v[(fCnt * 4 + 3) * 3 + 1] = (y + 1) * sizeY;
                    v[(fCnt * 4 + 3) * 3 + 2] = (z + 1) * sizeZ;
                    n[(fCnt * 4 + 0) * 3 + 0] = 0.0f;
                    n[(fCnt * 4 + 0) * 3 + 1] = 1.0f;
                    n[(fCnt * 4 + 0) * 3 + 2] = 0.0f;
                    n[(fCnt * 4 + 1) * 3 + 0] = n[(fCnt * 4 + 0) * 3 + 0];
                    n[(fCnt * 4 + 1) * 3 + 1] = n[(fCnt * 4 + 0) * 3 + 1];
                    n[(fCnt * 4 + 1) * 3 + 2] = n[(fCnt * 4 + 0) * 3 + 2];
                    n[(fCnt * 4 + 2) * 3 + 0] = n[(fCnt * 4 + 0) * 3 + 0];
                    n[(fCnt * 4 + 2) * 3 + 1] = n[(fCnt * 4 + 0) * 3 + 1];
                    n[(fCnt * 4 + 2) * 3 + 2] = n[(fCnt * 4 + 0) * 3 + 2];
                    n[(fCnt * 4 + 3) * 3 + 0] = n[(fCnt * 4 + 0) * 3 + 0];
                    n[(fCnt * 4 + 3) * 3 + 1] = n[(fCnt * 4 + 0) * 3 + 1];
                    n[(fCnt * 4 + 3) * 3 + 2] = n[(fCnt * 4 + 0) * 3 + 2];
                    fCnt++;
                }
                if ((z == 0) || !volume[x + (y + (z - 1) * cntY) * cntZ]) {
                    v[(fCnt * 4 + 0) * 3 + 0] = x * sizeX;
                    v[(fCnt * 4 + 0) * 3 + 1] = y * sizeY;
                    v[(fCnt * 4 + 0) * 3 + 2] = z * sizeZ;
                    v[(fCnt * 4 + 1) * 3 + 0] = x * sizeX;
                    v[(fCnt * 4 + 1) * 3 + 1] = (y + 1) * sizeY;
                    v[(fCnt * 4 + 1) * 3 + 2] = z * sizeZ;
                    v[(fCnt * 4 + 2) * 3 + 0] = (x + 1) * sizeX;
                    v[(fCnt * 4 + 2) * 3 + 1] = y * sizeY;
                    v[(fCnt * 4 + 2) * 3 + 2] = z * sizeZ;
                    v[(fCnt * 4 + 3) * 3 + 0] = (x + 1) * sizeX;
                    v[(fCnt * 4 + 3) * 3 + 1] = (y + 1) * sizeY;
                    v[(fCnt * 4 + 3) * 3 + 2] = z * sizeZ;
                    n[(fCnt * 4 + 0) * 3 + 0] = 0.0f;
                    n[(fCnt * 4 + 0) * 3 + 1] = 0.0f;
                    n[(fCnt * 4 + 0) * 3 + 2] = -1.0f;
                    n[(fCnt * 4 + 1) * 3 + 0] = n[(fCnt * 4 + 0) * 3 + 0];
                    n[(fCnt * 4 + 1) * 3 + 1] = n[(fCnt * 4 + 0) * 3 + 1];
                    n[(fCnt * 4 + 1) * 3 + 2] = n[(fCnt * 4 + 0) * 3 + 2];
                    n[(fCnt * 4 + 2) * 3 + 0] = n[(fCnt * 4 + 0) * 3 + 0];
                    n[(fCnt * 4 + 2) * 3 + 1] = n[(fCnt * 4 + 0) * 3 + 1];
                    n[(fCnt * 4 + 2) * 3 + 2] = n[(fCnt * 4 + 0) * 3 + 2];
                    n[(fCnt * 4 + 3) * 3 + 0] = n[(fCnt * 4 + 0) * 3 + 0];
                    n[(fCnt * 4 + 3) * 3 + 1] = n[(fCnt * 4 + 0) * 3 + 1];
                    n[(fCnt * 4 + 3) * 3 + 2] = n[(fCnt * 4 + 0) * 3 + 2];
                    fCnt++;
                }
                if ((z + 1 == cntZ) || !volume[x + (y + (z + 1) * cntY) * cntZ]) {
                    v[(fCnt * 4 + 0) * 3 + 0] = x * sizeX;
                    v[(fCnt * 4 + 0) * 3 + 1] = y * sizeY;
                    v[(fCnt * 4 + 0) * 3 + 2] = (z + 1) * sizeZ;
                    v[(fCnt * 4 + 1) * 3 + 0] = (x + 1) * sizeX;
                    v[(fCnt * 4 + 1) * 3 + 1] = y * sizeY;
                    v[(fCnt * 4 + 1) * 3 + 2] = (z + 1) * sizeZ;
                    v[(fCnt * 4 + 2) * 3 + 0] = x * sizeX;
                    v[(fCnt * 4 + 2) * 3 + 1] = (y + 1) * sizeY;
                    v[(fCnt * 4 + 2) * 3 + 2] = (z + 1) * sizeZ;
                    v[(fCnt * 4 + 3) * 3 + 0] = (x + 1) * sizeX;
                    v[(fCnt * 4 + 3) * 3 + 1] = (y + 1) * sizeY;
                    v[(fCnt * 4 + 3) * 3 + 2] = (z + 1) * sizeZ;
                    n[(fCnt * 4 + 0) * 3 + 0] = 0.0f;
                    n[(fCnt * 4 + 0) * 3 + 1] = 0.0f;
                    n[(fCnt * 4 + 0) * 3 + 2] = 1.0f;
                    n[(fCnt * 4 + 1) * 3 + 0] = n[(fCnt * 4 + 0) * 3 + 0];
                    n[(fCnt * 4 + 1) * 3 + 1] = n[(fCnt * 4 + 0) * 3 + 1];
                    n[(fCnt * 4 + 1) * 3 + 2] = n[(fCnt * 4 + 0) * 3 + 2];
                    n[(fCnt * 4 + 2) * 3 + 0] = n[(fCnt * 4 + 0) * 3 + 0];
                    n[(fCnt * 4 + 2) * 3 + 1] = n[(fCnt * 4 + 0) * 3 + 1];
                    n[(fCnt * 4 + 2) * 3 + 2] = n[(fCnt * 4 + 0) * 3 + 2];
                    n[(fCnt * 4 + 3) * 3 + 0] = n[(fCnt * 4 + 0) * 3 + 0];
                    n[(fCnt * 4 + 3) * 3 + 1] = n[(fCnt * 4 + 0) * 3 + 1];
                    n[(fCnt * 4 + 3) * 3 + 2] = n[(fCnt * 4 + 0) * 3 + 2];
                    fCnt++;
                }
            }
        }
    }

    this->objs.Append(Mesh());
    Mesh& mesh = this->objs.Last();
    mesh.SetVertexData(4 * fCnt, v, n, NULL, NULL, true);
    unsigned int* t = new unsigned int[6 * fCnt];
    for (unsigned int i = 0; i < fCnt; i++) {
        t[i * 6 + 0] = i * 4 + 0;
        t[i * 6 + 1] = i * 4 + 1;
        t[i * 6 + 2] = i * 4 + 2;
        t[i * 6 + 3] = i * 4 + 2;
        t[i * 6 + 4] = i * 4 + 1;
        t[i * 6 + 5] = i * 4 + 3;
    }
    mesh.SetTriangleData(2 * fCnt, t, true);
    this->datahash++;
}
