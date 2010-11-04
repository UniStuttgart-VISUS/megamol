#include "stdafx.h"
#include "Voxelizer.h"
#include "JobStructures.h"
#include "vislib/Log.h"
#include "vislib/ShallowPoint.h"
#include "MarchingCubeTables.h"

using namespace megamol;
using namespace megamol::trisoup;
using namespace megamol::trisoup::volumetrics;

Voxelizer::Voxelizer(void) : terminate(false), sjd(NULL) {
    //triangleSoup.SetCapacityIncrement(90); // AKA 10 triangles?
}


Voxelizer::~Voxelizer(void) {
}

vislib::math::Point<signed char, 3> neighbors[] = {
    vislib::math::Point<signed char, 3>(-1, 0, 0), vislib::math::Point<signed char, 3>(1, 0, 0),
    vislib::math::Point<signed char, 3>(0, -1, 0), vislib::math::Point<signed char, 3>(0, 1, 0),
    vislib::math::Point<signed char, 3>(0, 0, -1), vislib::math::Point<signed char, 3>(0, 0, 1)
};

void Voxelizer::growSurfaceFromTriangle(FatVoxel *theVolume, unsigned int x, unsigned int y, unsigned int z,
                             unsigned char triIndex, 
                             vislib::Array<float> &surf, vislib::Array<BorderVoxel *> &border) {

    FatVoxel *f = &theVolume[(z * sjd->resY + y) * sjd->resX + x];
    int currSurfID = MarchingCubeTables::a2ucTriangleSurfaceID[f->mcCase][triIndex];

    vislib::math::ShallowShallowTriangle<float, 3> sst(f->triangles + 3 * 3 * triIndex);
    vislib::math::ShallowShallowTriangle<float, 3> sst2(f->triangles + 3 * 3 * triIndex);
    vislib::math::ShallowShallowTriangle<float, 3> sstTemp(f->triangles + 3 * 3 * triIndex);
    vislib::math::ShallowShallowTriangle<float, 3> sstI(f->triangles + 3 * 3 * triIndex);

    // we come here only once per surface and cell. that is why I can do this.
    //if (isBorder(x, y, z)) {
    //    unsigned int numTris = 0;
    //    for (unsigned char c = 0; c < MarchingCubeTables::a2ucTriangleConnectionCount[f->mcCase]; c++) {
    //        if (MarchingCubeTables::a2ucTriangleSurfaceID[f->mcCase][c] == currSurfID) {
    //            numTris++;
    //        }
    //    }
    //    bv = new BorderVoxel;
    //    bv->triangles.SetCount(numTris * 3 * 3);
    //    border.Append(bv);
    //    isBorder = true;
    //}

    for (unsigned char c = 0; c < MarchingCubeTables::a2ucTriangleConnectionCount[f->mcCase]; c++) {
        sstI.SetPointer(f->triangles + 3 * 3 * c);
        // if c != triIndex, the question is whether THAT one has a common edge with any other
        // in the same cell which already belongs to this surface
        // IN ADDITION to the 'base triangle' at index c (2nd level in-cell neighbor!)
        //if (c == triIndex || sst.HasCommonEdge(sstI)) {
        if (MarchingCubeTables::a2ucTriangleSurfaceID[f->mcCase][c] == currSurfID) {
            if ((f->consumedTriangles & (1 << c)) == 0) {
                //surf.Add(sstI.GetPointer());
                surf.SetCount(surf.Count() + 9);
                sstTemp.SetPointer(const_cast<float *>(surf.PeekElements() + surf.Count() - 9));
                sstTemp = sstI;
                if (isBorder(x, y, z)) {
                    if (f->borderVoxel == NULL) {
                        f->borderVoxel = new BorderVoxel();
                        f->borderVoxel->x = x;
                        f->borderVoxel->y = y;
                        f->borderVoxel->z = z;
                        f->borderVoxel->triangles.AssertCapacity(5 * 9);
                        border.Add(f->borderVoxel);
                    }
                    f->borderVoxel->triangles.SetCount(f->borderVoxel->triangles.Count() + 9);
                    sstTemp.SetPointer(const_cast<float *>(f->borderVoxel->triangles.PeekElements()
                        + f->borderVoxel->triangles.Count() - 9));
                    sstTemp = sstI;
                }
                f->consumedTriangles |= (1 << c);
                //vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
                //    "[%08u] consuming  (%04u, %04u, %04u)[%u/%u]"
                //    " (%03.3f, %03.3f, %03.3f), (%03.3f, %03.3f, %03.3f), (%03.3f, %03.3f, %03.3f)\n",
                //    vislib::sys::Thread::CurrentID(), x, y, z, c,
                //    MarchingCubeTables::a2ucTriangleConnectionCount[f->mcCase],
                //    sstI.PeekCoordinates()[0][0], sstI.PeekCoordinates()[0][1], sstI.PeekCoordinates()[0][2],
                //    sstI.PeekCoordinates()[1][0], sstI.PeekCoordinates()[1][1], sstI.PeekCoordinates()[1][2],
                //    sstI.PeekCoordinates()[2][0], sstI.PeekCoordinates()[2][1], sstI.PeekCoordinates()[2][2]);
            }
            for (int ni = 0; ni < 6; ni++) {
                if ((((neighbors[ni].X() < 0) && (x > 0)) || (neighbors[ni].X() == 0) || ((neighbors[ni].X() > 0) && (x < sjd->resX - 2))) &&
                    (((neighbors[ni].Y() < 0) && (y > 0)) || (neighbors[ni].Y() == 0) || ((neighbors[ni].Y() > 0) && (y < sjd->resY - 2))) &&
                    (((neighbors[ni].Z() < 0) && (z > 0)) || (neighbors[ni].Z() == 0) || ((neighbors[ni].Z() > 0) && (z < sjd->resZ - 2)))) {
                        FatVoxel *n = &theVolume[((z + neighbors[ni].Z()) * sjd->resY
                            + y + neighbors[ni].Y()) * sjd->resX + x + neighbors[ni].X()];
                        for (unsigned int m = 0; m < MarchingCubeTables::a2ucTriangleConnectionCount[n->mcCase]; m++) {
                            if ((n->consumedTriangles & (1 << m)) == 0) {
                                sst2.SetPointer(n->triangles + 3 * 3 * m);
                                if (sst2.HasCommonEdge(sstI)) {
                                    n->consumedTriangles |= (1 << m);
                                    //surf.Add(sst2.GetPointer());
                                    surf.SetCount(surf.Count() + 9);
                                    sstTemp.SetPointer(const_cast<float *>(surf.PeekElements() + surf.Count() - 9));
                                    sstTemp = sst2;

                                    if (isBorder(x + neighbors[ni].X(), y + neighbors[ni].Y(), z + neighbors[ni].Z())) {
                                        if (n->borderVoxel == NULL) {
                                            n->borderVoxel = new BorderVoxel();
                                            n->borderVoxel->x = x + neighbors[ni].X();
                                            n->borderVoxel->y = y + neighbors[ni].Y();
                                            n->borderVoxel->z = z + neighbors[ni].Z();
                                            n->borderVoxel->triangles.AssertCapacity(5 * 9);
                                            border.Add(n->borderVoxel);
                                        }
                                        n->borderVoxel->triangles.SetCount(n->borderVoxel->triangles.Count() + 9);
                                        sstTemp.SetPointer(const_cast<float *>(n->borderVoxel->triangles.PeekElements()
                                            + n->borderVoxel->triangles.Count() - 9));
                                        sstTemp = sst2;
                                    }

                                    //vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
                                    //    "[%08u] consuming  (%04u, %04u, %04u)[%u/%u]"
                                    //    " (%03.3f, %03.3f, %03.3f), (%03.3f, %03.3f, %03.3f), (%03.3f, %03.3f, %03.3f)\n",
                                    //    vislib::sys::Thread::CurrentID(),
                                    //    x + neighbors[ni].X(), y + neighbors[ni].Y(),
                                    //    z + neighbors[ni].Z(), m,
                                    //    MarchingCubeTables::a2ucTriangleConnectionCount[n->mcCase],
                                    //    sst2.PeekCoordinates()[0][0], sst2.PeekCoordinates()[0][1], sst2.PeekCoordinates()[0][2],
                                    //    sst2.PeekCoordinates()[1][0], sst2.PeekCoordinates()[1][1], sst2.PeekCoordinates()[1][2],
                                    //    sst2.PeekCoordinates()[2][0], sst2.PeekCoordinates()[2][1], sst2.PeekCoordinates()[2][2]);
                                    cellFIFO.Append(vislib::math::Point<unsigned int, 4>(
                                        x + neighbors[ni].X(),
                                        y + neighbors[ni].Y(),
                                        z + neighbors[ni].Z(), m));
                                }
                            }
                        }       
                }
            }
        }
    }
}

void Voxelizer::collectCell(FatVoxel *theVolume, unsigned int x, unsigned int y, unsigned int z) {
    if (CellEmpty(theVolume, x, y, z)) {
        return;
    }

    FatVoxel *f = &theVolume[(z * sjd->resY + y) * sjd->resX + x];
    vislib::math::ShallowShallowTriangle<float, 3> sst(f->triangles);
    //vislib::math::ShallowShallowTriangle<float, 3> sst2(f->triangles);
    //vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
    //    "[%08u] collecting (%04u, %04u, %04u)\n", vislib::sys::Thread::CurrentID(), x, y, z);
    for (unsigned int l = 0; l < MarchingCubeTables::a2ucTriangleConnectionCount[f->mcCase]; l++) {
        if ((f->consumedTriangles & (1 << l)) == 0) {
            // this is a new surface
            vislib::Array<float> surf;
            vislib::Array<BorderVoxel *> border;
            for (SIZE_T idx = 0; idx < sjd->resX * sjd->resY * sjd->resZ; idx++) {
                theVolume[idx].borderVoxel = NULL;
            }
            surf.SetCapacityIncrement(90);
            border.SetCapacityIncrement(10);
            cellFIFO.Append(vislib::math::Point<unsigned int, 4>(x, y, z, l));
            //vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
            //    "[%08u] appending  (%04u, %04u, %04u)[%u]\n", vislib::sys::Thread::CurrentID(), x, y, z, l);
            while(cellFIFO.Count() > 0) {
                vislib::math::Point<unsigned int, 4> p = cellFIFO.First();
                cellFIFO.RemoveFirst();
                //vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
                //    "[%08u] growing    (%04u, %04u, %04u)[%u]\n", vislib::sys::Thread::CurrentID(),
                //    p.X(), p.Y(), p.Z(), p.W());
                growSurfaceFromTriangle(theVolume, p.X(), p.Y(), p.Z(), p.W(), surf, border);
            }
            sjd->Result.surfaces.Append(surf);
            sjd->Result.borderVoxels.Append(border);
        }
    }
}


void Voxelizer::marchCell(FatVoxel *theVolume, unsigned int x, unsigned int y, unsigned int z) {
    if (CellEmpty(theVolume, x, y, z)) {
        //theVolume[(z * sjd->resY + y) * sjd->resX + x].numTriangles = 0;
        theVolume[(z * sjd->resY + y) * sjd->resX + x].mcCase = 0;
        return;
    }
    theVolume[(z * sjd->resY + y) * sjd->resX + x].consumedTriangles = 0;

    unsigned int i;
    float CubeValues[8];
    int flagIndex, edgeFlags, edge, triangle;
    int corner, vertex;
    float offset;
    vislib::math::Point<float, 3> EdgeVertex[12];

    //Make a local copy of the values at the cube's corners
    for (i = 0; i < 8; i++) {
        CubeValues[i] = theVolume[
            ((z + MarchingCubeTables::a2fVertexOffset[i][2]) * sjd->resY
                + y + MarchingCubeTables::a2fVertexOffset[i][1]) * sjd->resX
                + x + MarchingCubeTables::a2fVertexOffset[i][0]
        ].distField;
    }

    //Find which vertices are inside of the surface and which are outside
    flagIndex = 0;
    for (i = 0; i < 8; i++) {
        if (CubeValues[i] < 0.0f) {
            flagIndex |= 1 << i;
        }
    }

    //Find which edges are intersected by the surface
    edgeFlags = MarchingCubeTables::aiCubeEdgeFlags[flagIndex];

    //If the cube is entirely inside or outside of the surface, then there will be no intersections
    if (edgeFlags == 0) {
        return;
    }

    //Find the point of intersection of the surface with each edge
    //Then find the normal to the surface at those points
//
//- im aktuellen Eintrag der a2iTriangleConnectionTable die Edges paarweise nehmen
//- Edge-Richtung in a2fEdgeDirection nachschlagen
//- Cross Product gibt Richtung der Nachbarzelle, die auf den Stack gelegt werden muss.

    for (edge = 0; edge < 12; edge++) {
        //if there is an intersection on this edge
        if (edgeFlags & (1 << edge)) {
            offset = getOffset(CubeValues[MarchingCubeTables::a2iEdgeConnection[edge][0]],
                CubeValues[MarchingCubeTables::a2iEdgeConnection[edge][1]], 0.0f);
            vislib::math::Point<float, 3> p(sjd->Bounds.Left() + x * sjd->CellSize,
                sjd->Bounds.Bottom() + y * sjd->CellSize,
                sjd->Bounds.Back() + z * sjd->CellSize);

            EdgeVertex[edge].SetX(p.X() + (MarchingCubeTables::a2fVertexOffset[MarchingCubeTables::a2iEdgeConnection[edge][0]][0]
            + offset * MarchingCubeTables::a2fEdgeDirection[edge][0]) * sjd->CellSize);
            EdgeVertex[edge].SetY(p.Y() + (MarchingCubeTables::a2fVertexOffset[MarchingCubeTables::a2iEdgeConnection[edge][0]][1]
            + offset * MarchingCubeTables::a2fEdgeDirection[edge][1]) * sjd->CellSize);
            EdgeVertex[edge].SetZ(p.Z() + (MarchingCubeTables::a2fVertexOffset[MarchingCubeTables::a2iEdgeConnection[edge][0]][2]
            + offset * MarchingCubeTables::a2fEdgeDirection[edge][2]) * sjd->CellSize);

            //for (i = 0; i < 2; i++) {
            //    int lx = int(x) + MarchingCubeTables::neighbourTable[edge][i][0];
            //    int ly = int(y) + MarchingCubeTables::neighbourTable[edge][i][1];
            //    int lz = int(z) + MarchingCubeTables::neighbourTable[edge][i][2];
            //    if ((lx < 0) || (ly < 0) || (lz < 0)
            //        || (lx >= int(sjd->resX - 1)) || (ly >= int(sjd->resY - 1)) || (lz >= int(sjd->resZ - 1)))
            //        continue;
            //    if (!markedCells.IsTagged(lx, ly, lz)) {
            //        markedCells.Tag(lx, ly, lz);

            //        cellFIFO[fifoEnd++].Set(lx, ly, lz);
            //        if (fifoEnd >= fifoLen) fifoEnd = 0;
            //    }
            //}
        }
    }

    // make triangles
    vislib::math::Vector<float, 3> normal;
    vislib::math::Vector<float, 3> a, b;

    int triCnt = MarchingCubeTables::a2ucTriangleConnectionCount[flagIndex];
    theVolume[(z * sjd->resY + y) * sjd->resX + x].triangles = new float[triCnt * 3 * 3];
    //theVolume[(z * sjd->resY + y) * sjd->resX + x].numTriangles = triCnt;
    theVolume[(z * sjd->resY + y) * sjd->resX + x].mcCase = flagIndex;
    vislib::math::ShallowShallowTriangle<float, 3> tri(theVolume[(z * sjd->resY + y) * sjd->resX + x].triangles);

    for (triangle = 0; triangle < triCnt; triangle++) {
        //if (MarchingCubeTables::a2iTriangleConnectionTable[flagIndex][3*triangle] < 0) {
        //    break;
        //}

        a = EdgeVertex[MarchingCubeTables::a2iTriangleConnectionTable[flagIndex][3*triangle + 0]] 
        - EdgeVertex[MarchingCubeTables::a2iTriangleConnectionTable[flagIndex][3*triangle + 1]];
        b = EdgeVertex[MarchingCubeTables::a2iTriangleConnectionTable[flagIndex][3*triangle + 0]] 
        - EdgeVertex[MarchingCubeTables::a2iTriangleConnectionTable[flagIndex][3*triangle + 2]];
        normal = a.Cross(b);

        sjd->Result.surface += normal.Length() / 2.0f;
        normal.Normalise();
        tri.SetPointer(theVolume[(z * sjd->resY + y) * sjd->resX + x].triangles + 3 * 3 * triangle);
        for (corner = 0; corner < 3; corner++) {
            vertex = MarchingCubeTables::a2iTriangleConnectionTable[flagIndex][3*triangle + corner];
            tri[2 - corner] = EdgeVertex[vertex];
            //sjd->Result.vertices.Append(EdgeVertex[vertex]);
            //sjd->Result.normals.Append(normal);
            //sjd->Result.indices.Append(sjd->Result.vertices.Count() - 1);
        }
    }
}

// fGetOffset finds the approximate point of intersection of the surface
// between two points with the values fValue1 and fValue2
// TODO intersect with the sphere we know is nearest! code below makes no sense for our situation
float Voxelizer::getOffset(float fValue1, float fValue2, float fValueDesired) {
        double fDelta = fValue2 - fValue1;
        if(fDelta == 0.0) {
                return 0.5;
        }
        return (float)((fValueDesired - fValue1)/fDelta);
}

bool Voxelizer::CellEmpty(FatVoxel *theVolume, unsigned x, unsigned y, unsigned z) {
    unsigned int i;
    bool neg = false, pos = false;
    float f;

    //Make a local copy of the values at the cube's corners
    for (i = 0; i < 8; i++) {
        f = theVolume[
            ((z + MarchingCubeTables::a2fVertexOffset[i][2]) * sjd->resY
                + y + MarchingCubeTables::a2fVertexOffset[i][1]) * sjd->resX
                + x + MarchingCubeTables::a2fVertexOffset[i][0]
        ].distField;
        neg = neg | (f < 0.0f);
        pos = pos | (f >= 0.0f);
    }
    return !(neg && pos);
}

DWORD Voxelizer::Run(void *userData) {
    using vislib::sys::Log;

    int x, y, z;
    unsigned int vertFloatSize = 0;
    float currRad = 0.f;
    //, maxRad = -FLT_MAX;
    float currDist;
    vislib::math::Point<unsigned int, 3> pStart, pEnd;
    vislib::math::Point<float, 3> p;
    sjd = static_cast<SubJobData*>(userData);

    //TagVolume markedCells(sjd->resX - 1, sjd->resY - 1, sjd->resZ - 1);
    //fifoLen = (sjd->resX - 1) * (sjd->resY - 1) * (sjd->resZ - 1) + 1;
    //cellFIFO = new vislib::math::Point<unsigned int, 3>[fifoLen];
    unsigned int fifoEnd = 0, fifoCur = 0;
    FatVoxel *volume = new FatVoxel[sjd->resX * sjd->resY * sjd->resZ];
    for (SIZE_T i = 0; i < sjd->resX * sjd->resY * sjd->resZ; i++) {
        volume[i].distField = FLT_MAX;
        volume[i].borderVoxel = NULL;
    }

    unsigned int partListCnt = sjd->datacall->GetParticleListCount();
    for (unsigned int partListI = 0; partListI < partListCnt; partListI++) {
        core::moldyn::MultiParticleDataCall::Particles ps = sjd->datacall->AccessParticles(partListI);
        UINT64 numParticles = ps.GetCount();
        unsigned int stride = ps.GetVertexDataStride();
        core::moldyn::MultiParticleDataCall::Particles::VertexDataType dataType =
            ps.GetVertexDataType();
        unsigned char *vertexData = (unsigned char*)ps.GetVertexData();
        switch (dataType) {
            case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE:
                continue;
            case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                vertFloatSize = 3 * sizeof(float);
                break;
            case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                vertFloatSize = 4 * sizeof(float);
                break;
            case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
                Log::DefaultLog.WriteError("This module does not yet like quantized data");
                return -2;
        }
    }

    // sample everything into our temporary volume
    vislib::math::Cuboid<float> bx(sjd->Bounds);
    bx.Grow(2 * sjd->MaxRad * sjd->RadMult);

    for (unsigned int partListI = 0; partListI < partListCnt; partListI++) {
        core::moldyn::MultiParticleDataCall::Particles ps = sjd->datacall->AccessParticles(partListI);
        currRad = ps.GetGlobalRadius() * sjd->RadMult;
        UINT64 numParticles = ps.GetCount();
        unsigned int stride = ps.GetVertexDataStride();
        core::moldyn::MultiParticleDataCall::Particles::VertexDataType dataType =
            ps.GetVertexDataType();
        unsigned char *vertexData = (unsigned char*)ps.GetVertexData();
        switch (dataType) {
            case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE:
                continue;
            case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                vertFloatSize = 3 * sizeof(float);
                break;
            case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                vertFloatSize = 4 * sizeof(float);
                break;
            case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
                Log::DefaultLog.WriteError("This module does not yet like quantized data");
                return -2;
        }
        for (UINT64 l = 0; l < numParticles; l++) {
            vislib::math::ShallowPoint<float, 3> sp((float*)&vertexData[(vertFloatSize + stride) * l]);
            if (dataType == core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR) {
                currRad = (float)vertexData[(vertFloatSize + stride) * l + 3 * sizeof(float)];
                currRad *= sjd->RadMult;
            }
            if (!bx.Contains(sp, vislib::math::Cuboid<float>::FACE_ALL)) {
                continue;
            }
            x = static_cast<int>((sp.X() - currRad - sjd->Bounds.Left()) / sjd->CellSize);
            if (x > 0) x--;
            if (x < 0) x = 0;
            y = static_cast<int>((sp.Y() - currRad - sjd->Bounds.Bottom()) / sjd->CellSize);
            if (y > 0) y--;
            if (y < 0) y = 0;
            z = static_cast<int>((sp.Z() - currRad - sjd->Bounds.Back()) / sjd->CellSize);
            if (z > 0) z--;
            if (z < 0) z = 0;
            pStart.Set(x, y, z);

            x = static_cast<int>((sp.X() + currRad - sjd->Bounds.Left()) / sjd->CellSize);
            if (x + 1 < static_cast<int>(sjd->resX)) x++;
            if (x + 1 >= static_cast<int>(sjd->resX)) x = sjd->resX - 1;
            y = static_cast<int>((sp.Y() + currRad - sjd->Bounds.Bottom()) / sjd->CellSize);
            if (y + 1 < static_cast<int>(sjd->resY)) y++;
            if (y + 1 >= static_cast<int>(sjd->resY)) y = sjd->resY - 1;
            z = static_cast<int>((sp.Z() + currRad - sjd->Bounds.Back()) / sjd->CellSize);
            if (z + 1 < static_cast<int>(sjd->resZ)) z++;
            if (z + 1 >= static_cast<int>(sjd->resZ)) z = sjd->resZ - 1;
            pEnd.Set(x, y, z);

            for (z = pStart.Z(); z <= static_cast<int>(pEnd.Z()); z++) {
                for (y = pStart.Y(); y <= static_cast<int>(pEnd.Y()); y++) {
                    for (x = pStart.X(); x <= static_cast<int>(pEnd.X()); x++) {
                        // TODO think about this. here the voxel content is determined by a corner
                        p.Set(
                        sjd->Bounds.Left() + x * sjd->CellSize,
                        sjd->Bounds.Bottom() + y * sjd->CellSize,
                        sjd->Bounds.Back() + z * sjd->CellSize);

                        // and here it is the center!
                        //p.Set(
                        //	sjd->Bounds.Left() + x * sjd->CellSize + sjd->CellSize * 0.5f,
                        //	sjd->Bounds.Bottom() + y * sjd->CellSize + sjd->CellSize * 0.5f,
                        //	sjd->Bounds.Back() + z * sjd->CellSize + sjd->CellSize * 0.5f);
                        currDist = sp.Distance(p) - currRad;
                        SIZE_T i = (z * sjd->resY + y) * sjd->resX + x;
                        volume[i].distField = vislib::math::Min(volume[i].distField, currDist);
                    }
                }
            }
        }
    }

    // march it
    for (x = 0; x < static_cast<int>(sjd->resX) - 1; x++) {
        for (y = 0; y < static_cast<int>(sjd->resY) - 1; y++) {
            for (z = 0; z < static_cast<int>(sjd->resZ) - 1; z++) {
                marchCell(volume, x, y, z);
            }
        }
    }

    // collect the surfaces
    for (x = 0; x < static_cast<int>(sjd->resX) - 1; x++) {
        for (y = 0; y < static_cast<int>(sjd->resY) - 1; y++) {
            for (z = 0; z < static_cast<int>(sjd->resZ) - 1; z++) {
                collectCell(volume, x, y, z);
            }
        }
    }

    // pass on border(unemptied) fatvoxels,
    
    //for (unsigned int l = 0; l < sjd->Result.surfaces.Count(); l++) {
    //    vislib::graphics::ColourRGBAu8 c(rand() * 255, rand()*255, rand() * 255, 255);
    //    vislib::math::ShallowShallowTriangle<float, 3> sst(sjd->Result.surfaces[l][0]);
    //    for (unsigned int m = 0; m < sjd->Result.surfaces[l].Count(); m++) {
    //        sjd->Result.vertices.Append(vislib::math::Point<float, 3>(sjd->Result.surfaces[l][m]));
    //        sjd->Result.indices.Append(static_cast<unsigned int>(sjd->Result.vertices.Count() - 1));
    //        sjd->Result.vertices.Append(vislib::math::Point<float, 3>(sjd->Result.surfaces[l][m] + 3));
    //        sjd->Result.indices.Append(static_cast<unsigned int>(sjd->Result.vertices.Count() - 1));
    //        sjd->Result.vertices.Append(vislib::math::Point<float, 3>(sjd->Result.surfaces[l][m] + 6));
    //        sjd->Result.indices.Append(static_cast<unsigned int>(sjd->Result.vertices.Count() - 1));
    //        vislib::math::Vector<float, 3> norm;
    //        sst.SetPointer(sjd->Result.surfaces[l][m]);
    //        (sst.Normal(norm));
    //        sjd->Result.normals.Append(norm);
    //        sjd->Result.normals.Append(norm);
    //        sjd->Result.normals.Append(norm);
    //        sjd->Result.colors.Append(c);
    //        sjd->Result.colors.Append(c);
    //        sjd->Result.colors.Append(c);
    //    }
    //}

    // dealloc stuff in volume
    // dealloc volume as a whole etc.
    for (x = 0; x < static_cast<int>(sjd->resX) - 1; x++) {
        for (y = 0; y < static_cast<int>(sjd->resY) - 1; y++) {
            for (z = 0; z < static_cast<int>(sjd->resZ) - 1; z++) {
                if (MarchingCubeTables::a2ucTriangleConnectionCount[volume[(z * sjd->resY + y) * sjd->resX + x].mcCase] > 0) {
                    SAFE_DELETE(volume[(z * sjd->resY + y) * sjd->resX + x].triangles);
                }
            }
        }
    }
    ARY_SAFE_DELETE(volume);

    sjd->Result.done = true;

    return 0;
}

bool Voxelizer::Terminate(void) {
    terminate = true;
    return true;
}