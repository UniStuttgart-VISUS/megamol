#include "stdafx.h"
#include "TetraVoxelizer.h"
#include "JobStructures.h"
#include "vislib/Log.h"
#include "vislib/ShallowPoint.h"
#include "MarchingCubeTables.h"
#include "vislib/Vector.h"

using namespace megamol;
using namespace megamol::trisoup;
using namespace megamol::trisoup::volumetrics;

unsigned char tets[6][4] = {{0, 2, 3, 7}, {0, 2, 6, 7},
                            {0, 4, 6 ,7}, {0, 6, 1, 2},
                            {0, 6, 1, 4}, {5, 6, 1, 4}};

//#define ULTRADEBUG

vislib::math::Point<signed char, 3> TetraVoxelizer::cornerNeighbors[8][7] = {
    {vislib::math::Point<signed char, 3>(-1, 0, 0), vislib::math::Point<signed char, 3>(-1, 0, -1),
     vislib::math::Point<signed char, 3>(-1, -1, 0), vislib::math::Point<signed char, 3>(-1, -1, -1),
     vislib::math::Point<signed char, 3>(0, 0, -1), vislib::math::Point<signed char, 3>(0, -1, -1),
     vislib::math::Point<signed char, 3>(0, -1, 0)},

    {vislib::math::Point<signed char, 3>(1, 0, 0), vislib::math::Point<signed char, 3>(1, 0, -1),
     vislib::math::Point<signed char, 3>(1, -1, 0), vislib::math::Point<signed char, 3>(1, -1, -1),
     vislib::math::Point<signed char, 3>(0, 0, -1), vislib::math::Point<signed char, 3>(0, -1, -1),
     vislib::math::Point<signed char, 3>(0, -1, 0)},

    {vislib::math::Point<signed char, 3>(1, 1, 0), vislib::math::Point<signed char, 3>(1, 1, -1),
     vislib::math::Point<signed char, 3>(1, 0, 0), vislib::math::Point<signed char, 3>(1, 0, -1),
     vislib::math::Point<signed char, 3>(0, 1, -1), vislib::math::Point<signed char, 3>(0, 0, -1),
     vislib::math::Point<signed char, 3>(0, 1, 0)},

    {vislib::math::Point<signed char, 3>(-1, 1, 0), vislib::math::Point<signed char, 3>(-1, 1, -1),
     vislib::math::Point<signed char, 3>(-1, 0, 0), vislib::math::Point<signed char, 3>(-1, 0, -1),
     vislib::math::Point<signed char, 3>(0, 1, -1), vislib::math::Point<signed char, 3>(0, 0, -1),
     vislib::math::Point<signed char, 3>(0, 1, 0)},

     // front
    {vislib::math::Point<signed char, 3>(-1, 0, 1), vislib::math::Point<signed char, 3>(-1, 0, 0),
     vislib::math::Point<signed char, 3>(-1, -1, 1), vislib::math::Point<signed char, 3>(-1, -1, 0),
     vislib::math::Point<signed char, 3>(0, 0, 1), vislib::math::Point<signed char, 3>(0, -1, 1),
     vislib::math::Point<signed char, 3>(0, -1, 0)},

    {vislib::math::Point<signed char, 3>(1, 0, 1), vislib::math::Point<signed char, 3>(1, 0, 0),
     vislib::math::Point<signed char, 3>(1, -1, 1), vislib::math::Point<signed char, 3>(1, -1, 0),
     vislib::math::Point<signed char, 3>(0, 0, 1), vislib::math::Point<signed char, 3>(0, -1, 1),
     vislib::math::Point<signed char, 3>(0, -1, 0)},

    {vislib::math::Point<signed char, 3>(1, 1, 1), vislib::math::Point<signed char, 3>(1, 1, 0),
     vislib::math::Point<signed char, 3>(1, 0, 1), vislib::math::Point<signed char, 3>(1, 0, 0),
     vislib::math::Point<signed char, 3>(0, 1, 1), vislib::math::Point<signed char, 3>(0, 0, 1),
     vislib::math::Point<signed char, 3>(0, 1, 0)},

    {vislib::math::Point<signed char, 3>(-1, 1, 1), vislib::math::Point<signed char, 3>(-1, 1, 0),
     vislib::math::Point<signed char, 3>(-1, 0, 1), vislib::math::Point<signed char, 3>(-1, 0, 0),
     vislib::math::Point<signed char, 3>(0, 1, 1), vislib::math::Point<signed char, 3>(0, 0, 1),
     vislib::math::Point<signed char, 3>(0, 1, 0)}
};

vislib::math::Point<signed char, 3> TetraVoxelizer::moreNeighbors[6]  = {
    vislib::math::Point<signed char, 3>(-1, 0, 0), vislib::math::Point<signed char, 3>(1, 0, 0),
    vislib::math::Point<signed char, 3>(0, -1, 0), vislib::math::Point<signed char, 3>(0, 1, 0),
    vislib::math::Point<signed char, 3>(0, 0, -1), vislib::math::Point<signed char, 3>(0, 0, 1),
};

void TetraVoxelizer::debugPrintTriangle(vislib::math::ShallowShallowTriangle<float, 3> &tri) {
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
        "[%08u] (%03.3f, %03.3f, %03.3f), (%03.3f, %03.3f, %03.3f), (%03.3f, %03.3f, %03.3f)",
        vislib::sys::Thread::CurrentID(),
        tri.PeekCoordinates()[0][0], tri.PeekCoordinates()[0][1], tri.PeekCoordinates()[0][2],
        tri.PeekCoordinates()[1][0], tri.PeekCoordinates()[1][1], tri.PeekCoordinates()[1][2],
        tri.PeekCoordinates()[2][0], tri.PeekCoordinates()[2][1], tri.PeekCoordinates()[2][2]);
}

void TetraVoxelizer::debugPrintTriangle(vislib::math::ShallowShallowTriangle<double, 3> &tri) {
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
        "[%08u] (%03.3lf, %03.3lf, %03.3lf), (%03.3lf, %03.3lf, %03.3lf), (%03.3lf, %03.3lf, %03.3lf)",
        vislib::sys::Thread::CurrentID(),
        tri.PeekCoordinates()[0][0], tri.PeekCoordinates()[0][1], tri.PeekCoordinates()[0][2],
        tri.PeekCoordinates()[1][0], tri.PeekCoordinates()[1][1], tri.PeekCoordinates()[1][2],
        tri.PeekCoordinates()[2][0], tri.PeekCoordinates()[2][1], tri.PeekCoordinates()[2][2]);
}

TetraVoxelizer::TetraVoxelizer(void) : terminate(false), sjd(NULL) {
    //triangleSoup.SetCapacityIncrement(90); // AKA 10 triangles?
}


TetraVoxelizer::~TetraVoxelizer(void) {
}

bool TetraVoxelizer::CellHasNoGeometry(FatVoxel *theVolume, unsigned x, unsigned y, unsigned z) {
    unsigned int i;
    bool neg = false, pos = false;
    VoxelizerFloat f;

    for (i = 0; i < 8; i++) {
        f = theVolume[
            ((z + MarchingCubeTables::a2fVertexOffset[i][2]) * sjd->resY
                + y + MarchingCubeTables::a2fVertexOffset[i][1]) * sjd->resX
                + x + MarchingCubeTables::a2fVertexOffset[i][0]
        ].distField;
        neg = neg | (f < 0.0);
        pos = pos | (f >= 0.0);
    }
    return !(neg && pos);
}

bool TetraVoxelizer::CellFull(FatVoxel *theVolume, unsigned x, unsigned y, unsigned z) {
    //unsigned int i;
    //bool neg = true;
    //float f;

    //for (i = 0; i < 8; i++) {
    //    f = theVolume[
    //        ((z + MarchingCubeTables::a2fVertexOffset[i][2]) * sjd->resY
    //            + y + MarchingCubeTables::a2fVertexOffset[i][1]) * sjd->resX
    //            + x + MarchingCubeTables::a2fVertexOffset[i][0]
    //    ].distField;
    //    neg = neg && (f < 0.0f);
    //}
    //ASSERT (neg == (theVolume[(z * sjd->resY + y) * sjd->resX + x].mcCase == 255));
    //return neg;
    return theVolume[(z * sjd->resY + y) * sjd->resX + x].mcCase == 255;
}

void TetraVoxelizer::CollectCell(FatVoxel *theVolume, unsigned int x, unsigned int y, unsigned int z) {
    if (CellHasNoGeometry(theVolume, x, y, z)) {
        return;
    }

    FatVoxel &f = theVolume[(z * sjd->resY + y) * sjd->resX + x];

    if (f.numTriangles > 0) {
        vislib::math::ShallowShallowTriangle<VoxelizerFloat, 3> sst(f.triangles);
    }
    //vislib::math::ShallowShallowTriangle<float, 3> sst2(f.triangles);
#ifdef ULTRADEBUG
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
        "[%08u] collecting (%04u, %04u, %04u)\n", vislib::sys::Thread::CurrentID(), x, y, z);
#endif /* ULTRADEBUG */
    for (unsigned int l = 0; l < f.numTriangles; l++) {
        if ((f.consumedTriangles & (1 << l)) == 0) {
            // this is a new surface
            Surface surf;
            surf.border->SetCapacityIncrement(10);
            surf.mesh.SetCapacityIncrement(90);
            surf.surface = static_cast<VoxelizerFloat>(0.0);
            surf.volume = static_cast<VoxelizerFloat>(0.0);
            surf.fullFaces = 0;
            surf.globalID = UINT_MAX;
            for (SIZE_T idx = 0; idx < sjd->resX * sjd->resY * sjd->resZ; idx++) {
                theVolume[idx].borderVoxel = NULL;
            }
            cellFIFO.Append(vislib::math::Point<unsigned int, 4>(x, y, z, l));
#ifdef ULTRADEBUG
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
                "[%08u] appending  (%04u, %04u, %04u)[%u]\n", vislib::sys::Thread::CurrentID(), x, y, z, l);
#endif /* ULTRADEBUG */
            while(cellFIFO.Count() > 0) {
                vislib::math::Point<unsigned int, 4> p = cellFIFO.First();
                cellFIFO.RemoveFirst();
#ifdef ULTRADEBUG
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
                    "[%08u] growing    (%04u, %04u, %04u)[%u]\n", vislib::sys::Thread::CurrentID(),
                    p.X(), p.Y(), p.Z(), p.W());
#endif /* ULTRADEBUG */
                growSurfaceFromTriangle(theVolume, p.X(), p.Y(), p.Z(), p.W(), surf);
            }
            sjd->Result.surfaces.Append(surf);
        }
    }
}

VoxelizerFloat TetraVoxelizer::GetOffset(VoxelizerFloat fValue1, VoxelizerFloat fValue2, VoxelizerFloat fValueDesired) {
    VoxelizerFloat fDelta = fValue2 - fValue1;
    ASSERT(fDelta != static_cast<VoxelizerFloat>(0));
    VoxelizerFloat res = (fValueDesired - fValue1) / fDelta;
    ASSERT(res <= static_cast<VoxelizerFloat>(1) && res >= static_cast<VoxelizerFloat>(0));
    return res;
}

VoxelizerFloat TetraVoxelizer::growVolume(FatVoxel *theVolume, unsigned char &fullFaces,
                                          unsigned int x, unsigned int y, unsigned int z) {
    SIZE_T cells = 0;
    vislib::math::Point<int, 3> p;
    vislib::Array<vislib::math::Point<int, 3> > queue;
    queue.SetCapacityIncrement(32);
    queue.Add(vislib::math::Point<int, 3>(x, y, z));
    while (queue.Count() > 0) {
        p = queue.Last();
        queue.RemoveLast();
        FatVoxel &f = theVolume[(p.Z() * sjd->resY + p.Y()) * sjd->resX + p.X()];
        if (f.mcCase == 255 && f.consumedTriangles == 0) {
            cells++;
            f.consumedTriangles = 1;
            if (p.X() == 0) {
                fullFaces |= 1;
            }
            if (p.Y() == 0) {
                fullFaces |= 4;
            }
            if (p.Z() == 0) {
                fullFaces |= 16;
            }
            if (p.X() == sjd->resX - 2) {
                fullFaces |= 2;
            }
            if (p.Y() == sjd->resY - 2) {
                fullFaces |= 8;
            }
            if (p.Z() == sjd->resZ - 2) {
                fullFaces |= 32;
            }

#ifdef ULTRADEBUG
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
                "[%08u] grew to (%04u, %04u, %04u)",
                vislib::sys::Thread::CurrentID(), p.X(), p.Y(), p.Z());
#endif /* ULTRADEBUG */

            for (unsigned int ni = 0; ni < 6; ni++) {
                if ((((moreNeighbors[ni].X() < 0) && (p.X() > 0)) || (moreNeighbors[ni].X() == 0) || ((moreNeighbors[ni].X() > 0) && (p.X() < sjd->resX - 2))) &&
                    (((moreNeighbors[ni].Y() < 0) && (p.Y() > 0)) || (moreNeighbors[ni].Y() == 0) || ((moreNeighbors[ni].Y() > 0) && (p.Y() < sjd->resY - 2))) &&
                    (((moreNeighbors[ni].Z() < 0) && (p.Z() > 0)) || (moreNeighbors[ni].Z() == 0) || ((moreNeighbors[ni].Z() > 0) && (p.Z() < sjd->resZ - 2)))) {
                        queue.Add(vislib::math::Point<int, 3>(p.X() + moreNeighbors[ni].X(),
                            p.Y() + moreNeighbors[ni].Y(), p.Z() + moreNeighbors[ni].Z()));
                }
            }
        }
    }
#ifdef ULTRADEBUG
    if (cells > 0) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
            "[%08u] grew volume from (%04u, %04u, %04u) yielding %u cells and a volume of %f",
            vislib::sys::Thread::CurrentID(), x, y, z, cells, cells * sjd->CellSize * sjd->CellSize * sjd->CellSize);
    }
#endif /* ULTRADEBUG */
    return cells * sjd->CellSize * sjd->CellSize * sjd->CellSize;
}

void TetraVoxelizer::growSurfaceFromTriangle(FatVoxel *theVolume, unsigned int x, unsigned int y, unsigned int z,
                             unsigned char triIndex, Surface &surf) {

    FatVoxel &f = theVolume[(z * sjd->resY + y) * sjd->resX + x];
    //int currSurfID = MarchingCubeTables::a2ucTriangleSurfaceID[f->mcCase][triIndex];

    // first, grow the full neighbors
    for (unsigned int a = 0; a < 8; a++) {
        if (f.mcCase & (1 << a)) {
            for (unsigned int b = 0; b < 7; b++) {
                if ((((cornerNeighbors[a][b].X() < 0) && (x > 0)) 
                        || (cornerNeighbors[a][b].X() == 0) || ((cornerNeighbors[a][b].X() > 0) && (x < sjd->resX - 2))) &&
                    (((cornerNeighbors[a][b].Y() < 0) && (y > 0))
                        || (cornerNeighbors[a][b].Y() == 0) || ((cornerNeighbors[a][b].Y() > 0) && (y < sjd->resY - 2))) &&
                    (((cornerNeighbors[a][b].Z() < 0) && (z > 0))
                        || (cornerNeighbors[a][b].Z() == 0) || ((cornerNeighbors[a][b].Z() > 0) && (z < sjd->resZ - 2)))) {
                            surf.volume +=
                                growVolume(theVolume, surf.fullFaces, x + cornerNeighbors[a][b].X(),
                                y + cornerNeighbors[a][b].Y(), z + cornerNeighbors[a][b].Z());
                }
            }
        }
    }

    vislib::math::ShallowShallowTriangle<VoxelizerFloat, 3> sst(f.triangles + 3 * 3 * triIndex);
    vislib::math::ShallowShallowTriangle<VoxelizerFloat, 3> sst2(f.triangles + 3 * 3 * triIndex);
    vislib::math::ShallowShallowTriangle<VoxelizerFloat, 3> sstTemp(f.triangles + 3 * 3 * triIndex);
    vislib::math::ShallowShallowTriangle<VoxelizerFloat, 3> sstI(f.triangles + 3 * 3 * triIndex);

    // seed triangle
    unsigned short inCellSurf = 1 << triIndex;
    bool foundNew;

    // find all in-cell triangles connected with the seed triangle
    // TODO this is slow and very expensive
    do {
        foundNew = false;
        for (unsigned char c = 0; c < f.numTriangles; c++) {
            if ((inCellSurf & (1 << c)) == 0) {
                // we haven't been here before, or it did not fit.
                sstI.SetPointer(f.triangles + 3 * 3 * c);
                // does it fit to any of the collected triangles?
                for (unsigned char d = 0; d < f.numTriangles; d++) {
                    if (inCellSurf & (1 << d)) {
                        sst2.SetPointer(f.triangles + 3 * 3 * d);
                        //if (sstI.HasCommonEdge(sst2)) {
                        if (Dowel::HaveCommonEdge(sstI, sst2)) {
                            inCellSurf |= (1 << c);
                            foundNew = true;
                        }
                    }
                }
            }
        }
    } while (foundNew);

    for (unsigned char c = 0; c < f.numTriangles; c++) {
        // is this part of the in-cell surface?
        if (inCellSurf & (1 << c)) {
            sstI.SetPointer(f.triangles + 3 * 3 * c);
            //vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
            //    "[%08u] -> has common edge", vislib::sys::Thread::CurrentID());
            if ((f.consumedTriangles & (1 << c)) == 0) {
                //surf.Add(sstI.GetPointer());
                if (sjd->storeMesh) {
                    surf.mesh.SetCount(surf.mesh.Count() + 9);
                    sstTemp.SetPointer(const_cast<VoxelizerFloat *>(surf.mesh.PeekElements() + surf.mesh.Count() - 9));
                    sstTemp = sstI;
                }
                surf.surface += sstI.Area<VoxelizerFloat>();
                surf.volume += f.volumes[c];
                if (isBorder(x, y, z)) {
                    if (f.borderVoxel == NULL) {
                        f.borderVoxel = new BorderVoxel();
                        f.borderVoxel->x = x + sjd->offsetX;
                        f.borderVoxel->y = y + sjd->offsetY;
                        f.borderVoxel->z = z + sjd->offsetZ;
                        f.borderVoxel->triangles.AssertCapacity(f.numTriangles * 9);
                        surf.border->Add(f.borderVoxel);
                    }
                    f.borderVoxel->triangles.SetCount(f.borderVoxel->triangles.Count() + 9);
                    sstTemp.SetPointer(const_cast<VoxelizerFloat *>(f.borderVoxel->triangles.PeekElements()
                        + f.borderVoxel->triangles.Count() - 9));
                    sstTemp = sstI;
                }
                f.consumedTriangles |= (1 << c);
#ifdef ULTRADEBUG
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
                    "[%08u] consuming  (%04u, %04u, %04u)[%u/%u]",
                    vislib::sys::Thread::CurrentID(), x, y, z, c, f.numTriangles);
                debugPrintTriangle(sstI);
#endif /* ULTRADEBUG */
            }
            for (int ni = 0; ni < 6; ni++) {
                if ((((moreNeighbors[ni].X() < 0) && (x > 0)) || (moreNeighbors[ni].X() == 0) || ((moreNeighbors[ni].X() > 0) && (x < sjd->resX - 2))) &&
                    (((moreNeighbors[ni].Y() < 0) && (y > 0)) || (moreNeighbors[ni].Y() == 0) || ((moreNeighbors[ni].Y() > 0) && (y < sjd->resY - 2))) &&
                    (((moreNeighbors[ni].Z() < 0) && (z > 0)) || (moreNeighbors[ni].Z() == 0) || ((moreNeighbors[ni].Z() > 0) && (z < sjd->resZ - 2)))) {
                        FatVoxel &n = theVolume[((z + moreNeighbors[ni].Z()) * sjd->resY
                            + y + moreNeighbors[ni].Y()) * sjd->resX + x + moreNeighbors[ni].X()];
                        for (unsigned int m = 0; m < n.numTriangles; m++) {
                            if ((n.consumedTriangles & (1 << m)) == 0) {
                                sst2.SetPointer(n.triangles + 3 * 3 * m);
#ifdef ULTRADEBUG
                                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
                                    "[%08u] comparing with (%04u, %04u, %04u)[%u/%u]", vislib::sys::Thread::CurrentID(),
                                    x + moreNeighbors[ni].X(),
                                    y + moreNeighbors[ni].Y(), z + moreNeighbors[ni].Z(), m, n.numTriangles);
                                debugPrintTriangle(sst2);
                                debugPrintTriangle(sstI);
#endif /* ULTRADEBUG */
                                //if (sst2.HasCommonEdge(sstI)) {
                                if (Dowel::HaveCommonEdge(sst2, sstI)) {
                                    n.consumedTriangles |= (1 << m);
#ifdef ULTRADEBUG
                                    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
                                        "[%08u] -> has common edge", vislib::sys::Thread::CurrentID());
                                    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
                                        "[%08u] consuming  (%04u, %04u, %04u)[%u/%u]",
                                        vislib::sys::Thread::CurrentID(), x + moreNeighbors[ni].X(),
                                        y + moreNeighbors[ni].Y(), z + moreNeighbors[ni].Z(), m, n.numTriangles);
#endif /* ULTRADEBUG */
                                    //surf.Add(sst2.GetPointer());
                                    if (sjd->storeMesh) {
                                        surf.mesh.SetCount(surf.mesh.Count() + 9);
                                        sstTemp.SetPointer(const_cast<VoxelizerFloat *>(surf.mesh.PeekElements() 
                                            + surf.mesh.Count() - 9));
                                        sstTemp = sst2;
                                    }
                                    surf.surface += sst2.Area<VoxelizerFloat>();
                                    surf.volume += n.volumes[m];
                                    if (isBorder(x + moreNeighbors[ni].X(), y + moreNeighbors[ni].Y(), z + moreNeighbors[ni].Z())) {
                                        if (n.borderVoxel == NULL) {
                                            n.borderVoxel = new BorderVoxel();
                                            n.borderVoxel->x = x + moreNeighbors[ni].X() + sjd->offsetX;
                                            n.borderVoxel->y = y + moreNeighbors[ni].Y() + sjd->offsetY;
                                            n.borderVoxel->z = z + moreNeighbors[ni].Z() + sjd->offsetZ;
                                            n.borderVoxel->triangles.AssertCapacity(n.numTriangles * 9);
                                            surf.border->Add(n.borderVoxel);
                                        }
                                        n.borderVoxel->triangles.SetCount(n.borderVoxel->triangles.Count() + 9);
                                        sstTemp.SetPointer(const_cast<VoxelizerFloat *>(n.borderVoxel->triangles.PeekElements()
                                            + n.borderVoxel->triangles.Count() - 9));
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
                                        x + moreNeighbors[ni].X(),
                                        y + moreNeighbors[ni].Y(),
                                        z + moreNeighbors[ni].Z(), m));
                                }
                            }
                        }       
                }
            }
        }
    }
}

void TetraVoxelizer::MarchCell(FatVoxel *theVolume, unsigned int x, unsigned int y, unsigned int z) {

    FatVoxel &currVoxel = theVolume[(z * sjd->resY + y) * sjd->resX + x];
    currVoxel.consumedTriangles = 0;
    currVoxel.numTriangles = 0;

    unsigned int i;
    VoxelizerFloat CubeValues[8];
    vislib::math::Point<VoxelizerFloat, 3> EdgeVertex[12];

    currVoxel.mcCase = 0;
    //Make a local copy of the values at the cube's corners
    for (i = 0; i < 8; i++) {
        CubeValues[i] = theVolume[
            ((z + MarchingCubeTables::a2fVertexOffset[i][2]) * sjd->resY
                + y + MarchingCubeTables::a2fVertexOffset[i][1]) * sjd->resX
                + x + MarchingCubeTables::a2fVertexOffset[i][0]
        ].distField;
        if (CubeValues[i] < 0.0f) {
            currVoxel.mcCase |= 1 << i;
        }
    }
    //CellFull(theVolume, x, y, z);
    if (CellHasNoGeometry(theVolume, x, y, z)) {// || !((x==6) && (y==7) && (z==6))) {
        currVoxel.consumedTriangles = 0;
        currVoxel.triangles = NULL;
        currVoxel.volumes = NULL;
        currVoxel.numTriangles = 0;
        return;
    }

    // reference corner of this cell
    vislib::math::Point<VoxelizerFloat, 3> p(sjd->Bounds.Left() + x * sjd->CellSize,
        sjd->Bounds.Bottom() + y * sjd->CellSize,
        sjd->Bounds.Back() + z * sjd->CellSize);

    // how many triangles will we get?
    for (unsigned char tetIdx = 0; tetIdx < 6; tetIdx++) {
        unsigned char triIdx = 0;
        if (CubeValues[tets[tetIdx][0]] < 0.0f)
            triIdx |= 1;
        if (CubeValues[tets[tetIdx][1]] < 0.0f)
            triIdx |= 2;
        if (CubeValues[tets[tetIdx][2]] < 0.0f)
            triIdx |= 4;
        if (CubeValues[tets[tetIdx][3]] < 0.0f)
            triIdx |= 8;

        switch(triIdx) {
            case 0x00:
            case 0x0F:
                break;
            case 0x0E:
            case 0x01:
                currVoxel.numTriangles++;
                break;
            case 0x0D:
            case 0x02:
                currVoxel.numTriangles++;
                break;
            case 0x0C:
            case 0x03:
                currVoxel.numTriangles += 2;
                break;
            case 0x0B:
            case 0x04:
                currVoxel.numTriangles++;
                break;
            case 0x0A:
            case 0x05:
                currVoxel.numTriangles += 2;
                break;
            case 0x09:
            case 0x06:
                currVoxel.numTriangles += 2;
                break;
            case 0x07:
            case 0x08:
                currVoxel.numTriangles++;
                break;
        }
    }

    currVoxel.triangles = new VoxelizerFloat[currVoxel.numTriangles * 3 * 3];
    currVoxel.volumes = new VoxelizerFloat[currVoxel.numTriangles];
    vislib::math::ShallowShallowTriangle<VoxelizerFloat, 3> tri(currVoxel.triangles);
    vislib::math::ShallowShallowTriangle<VoxelizerFloat, 3> tri2(currVoxel.triangles);
    vislib::math::Point<VoxelizerFloat, 3> temp;
    VoxelizerFloat *vol = NULL;
    VoxelizerFloat *vol2 = NULL;
    int triOffset = 0;

    // now we repeat this for all six sub-tetrahedra
    for (unsigned char tetIdx = 0; tetIdx < 6; tetIdx++) {
        unsigned char triIdx = 0;
        if (CubeValues[tets[tetIdx][0]] < 0.0f)
            triIdx |= 1;
        if (CubeValues[tets[tetIdx][1]] < 0.0f)
            triIdx |= 2;
        if (CubeValues[tets[tetIdx][2]] < 0.0f)
            triIdx |= 4;
        if (CubeValues[tets[tetIdx][3]] < 0.0f)
            triIdx |= 8;

        vislib::math::Point<VoxelizerFloat, 3> p0(
            p.X() + (VoxelizerFloat)MarchingCubeTables::a2fVertexOffset[tets[tetIdx][0]][0] * sjd->CellSize,
            p.Y() + (VoxelizerFloat)MarchingCubeTables::a2fVertexOffset[tets[tetIdx][0]][1] * sjd->CellSize,
            p.Z() + (VoxelizerFloat)MarchingCubeTables::a2fVertexOffset[tets[tetIdx][0]][2] * sjd->CellSize);
        vislib::math::Point<VoxelizerFloat, 3> p1(
            p.X() + (VoxelizerFloat)MarchingCubeTables::a2fVertexOffset[tets[tetIdx][1]][0] * sjd->CellSize,
            p.Y() + (VoxelizerFloat)MarchingCubeTables::a2fVertexOffset[tets[tetIdx][1]][1] * sjd->CellSize,
            p.Z() + (VoxelizerFloat)MarchingCubeTables::a2fVertexOffset[tets[tetIdx][1]][2] * sjd->CellSize);
        vislib::math::Point<VoxelizerFloat, 3> p2(
            p.X() + (VoxelizerFloat)MarchingCubeTables::a2fVertexOffset[tets[tetIdx][2]][0] * sjd->CellSize,
            p.Y() + (VoxelizerFloat)MarchingCubeTables::a2fVertexOffset[tets[tetIdx][2]][1] * sjd->CellSize,
            p.Z() + (VoxelizerFloat)MarchingCubeTables::a2fVertexOffset[tets[tetIdx][2]][2] * sjd->CellSize);
        vislib::math::Point<VoxelizerFloat, 3> p3(
            p.X() + (VoxelizerFloat)MarchingCubeTables::a2fVertexOffset[tets[tetIdx][3]][0] * sjd->CellSize,
            p.Y() + (VoxelizerFloat)MarchingCubeTables::a2fVertexOffset[tets[tetIdx][3]][1] * sjd->CellSize,
            p.Z() + (VoxelizerFloat)MarchingCubeTables::a2fVertexOffset[tets[tetIdx][3]][2] * sjd->CellSize);

        VoxelizerFloat fullVol = vislib::math::Abs((p1 - p0).Dot((p2 - p0).Cross(p3 - p0))) 
            / static_cast<VoxelizerFloat>(6.0);

        tri.SetPointer(currVoxel.triangles + 3 * 3 * triOffset);
        vol = currVoxel.volumes + triOffset;
        switch(triIdx) {
            case 0x00:
            case 0x0F:
                break;
            case 0x0E:
            case 0x01:
                //if (CubeValues[tets[tetIdx][0]] > 0.0f) {
                //} else {
                
                    tri[0] = p0.Interpolate(p1, GetOffset(CubeValues[tets[tetIdx][0]],
                        CubeValues[tets[tetIdx][1]], static_cast<VoxelizerFloat>(0)));
                    tri[2] = p0.Interpolate(p2, GetOffset(CubeValues[tets[tetIdx][0]],
                        CubeValues[tets[tetIdx][2]], static_cast<VoxelizerFloat>(0)));
                    tri[1] = p0.Interpolate(p3, GetOffset(CubeValues[tets[tetIdx][0]],
                        CubeValues[tets[tetIdx][3]], static_cast<VoxelizerFloat>(0)));

                    *vol = vislib::math::Abs((tri[0] - p0).Dot((tri[2] - p0).Cross(tri[1] - p0)))
                        / static_cast<VoxelizerFloat>(6.0);
                    if (CubeValues[tets[tetIdx][0]] > 0.0) {
                        *vol = fullVol - *vol;
                    }
                //}
                //tri[0] = p;
                //tri[1] = p;
                //tri[2] = p;
                //tri[0].p[0] = VertexInterp(iso,g.p[v0],g.p[v1],g.val[v0],g.val[v1]);
                //tri[0].p[1] = VertexInterp(iso,g.p[v0],g.p[v2],g.val[v0],g.val[v2]);
                //tri[0].p[2] = VertexInterp(iso,g.p[v0],g.p[v3],g.val[v0],g.val[v3]);
                triOffset++;
                break;
            case 0x0D:
            case 0x02:
                //if (CubeValues[tets[tetIdx][1]] > 0.0f) {
                //    tri[0] = p1.Interpolate(p0, GetOffset(CubeValues[tets[tetIdx][1]], CubeValues[tets[tetIdx][0]], 0.0f));
                //    tri[2] = p1.Interpolate(p3, GetOffset(CubeValues[tets[tetIdx][1]], CubeValues[tets[tetIdx][3]], 0.0f));
                //    tri[1] = p1.Interpolate(p2, GetOffset(CubeValues[tets[tetIdx][1]], CubeValues[tets[tetIdx][2]], 0.0f));
                //} else {
                    tri[0] = p1.Interpolate(p0, GetOffset(CubeValues[tets[tetIdx][1]], 
                        CubeValues[tets[tetIdx][0]], static_cast<VoxelizerFloat>(0)));
                    tri[1] = p1.Interpolate(p3, GetOffset(CubeValues[tets[tetIdx][1]],
                        CubeValues[tets[tetIdx][3]], static_cast<VoxelizerFloat>(0)));
                    tri[2] = p1.Interpolate(p2, GetOffset(CubeValues[tets[tetIdx][1]],
                        CubeValues[tets[tetIdx][2]], static_cast<VoxelizerFloat>(0)));

                    *vol = vislib::math::Abs((tri[0] - p1).Dot((tri[1] - p1).Cross(tri[2] - p1)))
                        / static_cast<VoxelizerFloat>(6.0);
                    if (CubeValues[tets[tetIdx][1]] > 0.0) {
                        *vol = fullVol - *vol;
                    }
                //}
                //tri[0] = p;
                //tri[1] = p;
                //tri[2] = p;
                //tri[0].p[0] = VertexInterp(iso,g.p[v1],g.p[v0],g.val[v1],g.val[v0]);
                //tri[0].p[1] = VertexInterp(iso,g.p[v1],g.p[v3],g.val[v1],g.val[v3]);
                //tri[0].p[2] = VertexInterp(iso,g.p[v1],g.p[v2],g.val[v1],g.val[v2]);
                triOffset++;
                break;
            case 0x0C:
            case 0x03:
                // tetrahedron 1: around p1: p1->p2, p0->p2, p1->p3
                // tetrahedron 2: around p0: p0->p3, p1->p3, p0->p2
                // tetrahedron 3: around p1: p0, p1->p3, p0->p2
                //if (CubeValues[tets[tetIdx][0]] > 0.0f) {
                //    tri[0] = p0.Interpolate(p3, GetOffset(CubeValues[tets[tetIdx][0]], CubeValues[tets[tetIdx][3]], 0.0f));
                //    tri[1] = p0.Interpolate(p2, GetOffset(CubeValues[tets[tetIdx][0]], CubeValues[tets[tetIdx][2]], 0.0f));
                //    tri[2] = p1.Interpolate(p3, GetOffset(CubeValues[tets[tetIdx][1]], CubeValues[tets[tetIdx][3]], 0.0f));
                //} else {
                    tri[0] = p0.Interpolate(p3, GetOffset(CubeValues[tets[tetIdx][0]],
                        CubeValues[tets[tetIdx][3]], static_cast<VoxelizerFloat>(0)));
                    tri[1] = p0.Interpolate(p2, GetOffset(CubeValues[tets[tetIdx][0]],
                        CubeValues[tets[tetIdx][2]], static_cast<VoxelizerFloat>(0)));
                    tri[2] = p1.Interpolate(p3, GetOffset(CubeValues[tets[tetIdx][1]],
                        CubeValues[tets[tetIdx][3]], static_cast<VoxelizerFloat>(0)));

                    // tet3
                    *vol = vislib::math::Abs((p0 - p1).Dot((tri[2] - p1).Cross(tri[1] - p1)))
                        / static_cast<VoxelizerFloat>(6.0);
                    // tet2
                    *vol += vislib::math::Abs((tri[0] - p0).Dot((tri[2] - p0).Cross(tri[1] - p0)))
                        / static_cast<VoxelizerFloat>(6.0);
                //}
                //tri[0] = p;
                //tri[1] = p;
                //tri[2] = p;
                //tri[0].p[0] = VertexInterp(iso,g.p[v0],g.p[v3],g.val[v0],g.val[v3]);
                //tri[0].p[1] = VertexInterp(iso,g.p[v0],g.p[v2],g.val[v0],g.val[v2]);
                //tri[0].p[2] = VertexInterp(iso,g.p[v1],g.p[v3],g.val[v1],g.val[v3]);
                triOffset++;
                tri2.SetPointer(currVoxel.triangles + 3 * 3 * triOffset);
                vol2 = currVoxel.volumes + triOffset;
                tri2[0] = tri[2];
                tri2[1] = p1.Interpolate(p2, GetOffset(CubeValues[tets[tetIdx][1]],
                    CubeValues[tets[tetIdx][2]], static_cast<VoxelizerFloat>(0)));
                tri2[2] = tri[1];
                // tet1
                *vol2 = vislib::math::Abs((tri2[1] - p1).Dot((tri[1] - p1).Cross(tri[2] - p1)))
                    / static_cast<VoxelizerFloat>(6.0);
                //tri2[0] = p;
                //tri2[1] = p;
                //tri2[2] = p;
                //tri[1].p[0] = tri[0].p[2];
                //tri[1].p[1] = VertexInterp(iso,g.p[v1],g.p[v2],g.val[v1],g.val[v2]);
                //tri[1].p[2] = tri[0].p[1];

                if (CubeValues[tets[tetIdx][0]] > 0.0) {
                    *vol = fullVol - (*vol + *vol2);
                    *vol2 = 0.0;
                }

                triOffset++;
                break;
            case 0x0B:
            case 0x04:
                //if (CubeValues[tets[tetIdx][2]] > 0.0f) {
                //    tri[0] = p2.Interpolate(p0, GetOffset(CubeValues[tets[tetIdx][2]], CubeValues[tets[tetIdx][0]], 0.0f));
                //    tri[2] = p2.Interpolate(p1, GetOffset(CubeValues[tets[tetIdx][2]], CubeValues[tets[tetIdx][1]], 0.0f));
                //    tri[1] = p2.Interpolate(p3, GetOffset(CubeValues[tets[tetIdx][2]], CubeValues[tets[tetIdx][3]], 0.0f));
                //} else {
                    tri[0] = p2.Interpolate(p0, GetOffset(CubeValues[tets[tetIdx][2]],
                        CubeValues[tets[tetIdx][0]], static_cast<VoxelizerFloat>(0)));
                    tri[1] = p2.Interpolate(p1, GetOffset(CubeValues[tets[tetIdx][2]],
                        CubeValues[tets[tetIdx][1]], static_cast<VoxelizerFloat>(0)));
                    tri[2] = p2.Interpolate(p3, GetOffset(CubeValues[tets[tetIdx][2]],
                        CubeValues[tets[tetIdx][3]], static_cast<VoxelizerFloat>(0)));

                    *vol = vislib::math::Abs((tri[0] - p2).Dot((tri[1] - p2).Cross(tri[2] - p2)))
                        / static_cast<VoxelizerFloat>(6.0);
                    if (CubeValues[tets[tetIdx][2]] > 0.0) {
                        *vol = fullVol - *vol;
                    }
                //}
                //tri[0] = p;
                //tri[1] = p;
                //tri[2] = p;
                //tri[0].p[0] = VertexInterp(iso,g.p[v2],g.p[v0],g.val[v2],g.val[v0]);
                //tri[0].p[1] = VertexInterp(iso,g.p[v2],g.p[v1],g.val[v2],g.val[v1]);
                //tri[0].p[2] = VertexInterp(iso,g.p[v2],g.p[v3],g.val[v2],g.val[v3]);
                triOffset++;
                break;
            case 0x0A:
            case 0x05:
                // WARNING: per analogy = 3, subst 1 with 2
                // tetrahedron 1: around p2: p1->p2, p0->p1, p2->p3
                // tetrahedron 2: around p0: p0->p3, p2->p3, p0->p1
                // tetrahedron 3: around p2: p0, p2->p3, p0->p1
                tri[0] = p0.Interpolate(p1, GetOffset(CubeValues[tets[tetIdx][0]],
                    CubeValues[tets[tetIdx][1]], static_cast<VoxelizerFloat>(0)));
                tri[1] = p2.Interpolate(p3, GetOffset(CubeValues[tets[tetIdx][2]],
                    CubeValues[tets[tetIdx][3]], static_cast<VoxelizerFloat>(0)));
                tri[2] = p0.Interpolate(p3, GetOffset(CubeValues[tets[tetIdx][0]],
                    CubeValues[tets[tetIdx][3]], static_cast<VoxelizerFloat>(0)));

                // tet2
                *vol = vislib::math::Abs((tri[2] - p0).Dot((tri[1] - p0).Cross(tri[0] - p0)))
                    / static_cast<VoxelizerFloat>(6.0);
                //tri[0] = p;
                //tri[1] = p;
                //tri[2] = p;
                //tri[0].p[0] = VertexInterp(iso,g.p[v0],g.p[v1],g.val[v0],g.val[v1]);
                //tri[0].p[1] = VertexInterp(iso,g.p[v2],g.p[v3],g.val[v2],g.val[v3]);
                //tri[0].p[2] = VertexInterp(iso,g.p[v0],g.p[v3],g.val[v0],g.val[v3]);
                triOffset++;
                tri2.SetPointer(currVoxel.triangles + 3 * 3 * triOffset);
                vol2 = currVoxel.volumes + triOffset;
                tri2[0] = tri[0];
                tri2[1] = p1.Interpolate(p2, GetOffset(CubeValues[tets[tetIdx][1]],
                    CubeValues[tets[tetIdx][2]], static_cast<VoxelizerFloat>(0)));
                tri2[2] = tri[1];

                // tet1
                *vol2 = vislib::math::Abs((tri2[1] - p2).Dot((tri[0] - p2).Cross(tri[1] - p2)))
                    / static_cast<VoxelizerFloat>(6.0);
                // tet3
                *vol2 += vislib::math::Abs((p0 - p2).Dot((tri[1] - p2).Cross(tri[0] - p2)))
                    / static_cast<VoxelizerFloat>(6.0);
                //tri2[0] = p;
                //tri2[1] = p;
                //tri2[2] = p;
                //tri[1].p[0] = tri[0].p[0];
                //tri[1].p[1] = VertexInterp(iso,g.p[v1],g.p[v2],g.val[v1],g.val[v2]);
                //tri[1].p[2] = tri[0].p[1];

                if (CubeValues[tets[tetIdx][0]] > 0.0) {
                    *vol = fullVol - (*vol + *vol2);
                    *vol2 = 0.0;
                }

                triOffset++;
                break;
            case 0x09:
            case 0x06:
                //if (CubeValues[tets[tetIdx][3]] > 0.0f) {

                //} else {
                    // WARNING: per analogy = 3, subst 0 with 2
                    // tetrahedron 1: around p1: p1->p0, p0->p2, p1->p3
                    // tetrahedron 2: around p2: p2->p3, p1->p3, p0->p2
                    // tetrahedron 3: around p1: p2, p1->p3, p0->p2
                    tri[0] = p0.Interpolate(p1, GetOffset(CubeValues[tets[tetIdx][0]],
                        CubeValues[tets[tetIdx][1]], static_cast<VoxelizerFloat>(0)));
                    tri[1] = p1.Interpolate(p3, GetOffset(CubeValues[tets[tetIdx][1]],
                        CubeValues[tets[tetIdx][3]], static_cast<VoxelizerFloat>(0)));
                    tri[2] = p2.Interpolate(p3, GetOffset(CubeValues[tets[tetIdx][2]],
                        CubeValues[tets[tetIdx][3]], static_cast<VoxelizerFloat>(0)));

                    //tri[0] = p;
                    //tri[1] = p;
                    //tri[2] = p;
                    //tri[0].p[0] = VertexInterp(iso,g.p[v0],g.p[v1],g.val[v0],g.val[v1]);
                    //tri[0].p[1] = VertexInterp(iso,g.p[v1],g.p[v3],g.val[v1],g.val[v3]);
                    //tri[0].p[2] = VertexInterp(iso,g.p[v2],g.p[v3],g.val[v2],g.val[v3]);
                    triOffset++;
                    tri2.SetPointer(currVoxel.triangles + 3 * 3 * triOffset);
                    tri2[0] = tri[0];
                    tri2[1] = p0.Interpolate(p2, GetOffset(CubeValues[tets[tetIdx][0]],
                        CubeValues[tets[tetIdx][2]], static_cast<VoxelizerFloat>(0)));
                    tri2[2] = tri[2];

                    // tet1
                    *vol = vislib::math::Abs((tri[0] - p1).Dot((tri2[1] - p1).Cross(tri[1] - p1)))
                        / static_cast<VoxelizerFloat>(6.0);
                    vol2 = currVoxel.volumes + triOffset;
                    // tet2
                    *vol2 = vislib::math::Abs((tri[2] - p2).Dot((tri[1] - p2).Cross(tri2[1] - p2)))
                        / static_cast<VoxelizerFloat>(6.0);
                    // tet3
                    *vol2 += vislib::math::Abs((p2 - p1).Dot((tri[1] - p1).Cross(tri2[1] - p1)))
                        / static_cast<VoxelizerFloat>(6.0);
                    //tri2[0] = p;
                    //tri2[1] = p;
                    //tri2[2] = p;
                    //tri[1].p[0] = tri[0].p[0];
                    //tri[1].p[1] = VertexInterp(iso,g.p[v0],g.p[v2],g.val[v0],g.val[v2]);
                    //tri[1].p[2] = tri[0].p[2];
                //}

                if (CubeValues[tets[tetIdx][1]] > 0.0) {
                    *vol = fullVol - (*vol + *vol2);
                    *vol2 = 0.0;
                }

                triOffset++;
                break;
            case 0x07:
            case 0x08:
                //if (CubeValues[tets[tetIdx][3]] > 0.0f) {
                //    tri[0] = p3.Interpolate(p0, GetOffset(CubeValues[tets[tetIdx][3]], CubeValues[tets[tetIdx][0]], 0.0f));
                //    tri[2] = p3.Interpolate(p2, GetOffset(CubeValues[tets[tetIdx][3]], CubeValues[tets[tetIdx][2]], 0.0f));
                //    tri[1] = p3.Interpolate(p1, GetOffset(CubeValues[tets[tetIdx][3]], CubeValues[tets[tetIdx][1]], 0.0f));
                //} else {
                    tri[0] = p3.Interpolate(p0, GetOffset(CubeValues[tets[tetIdx][3]],
                        CubeValues[tets[tetIdx][0]], static_cast<VoxelizerFloat>(0)));
                    tri[1] = p3.Interpolate(p2, GetOffset(CubeValues[tets[tetIdx][3]],
                        CubeValues[tets[tetIdx][2]], static_cast<VoxelizerFloat>(0)));
                    tri[2] = p3.Interpolate(p1, GetOffset(CubeValues[tets[tetIdx][3]],
                        CubeValues[tets[tetIdx][1]], static_cast<VoxelizerFloat>(0)));

                    *vol = vislib::math::Abs((tri[0] - p3).Dot((tri[1] - p3).Cross(tri[2] - p3)))
                        / static_cast<VoxelizerFloat>(6.0);
                    if (CubeValues[tets[tetIdx][3]] > 0.0) {
                        *vol = fullVol - *vol;
                    }
                //}
                //tri[0] = p;
                //tri[1] = p;
                //tri[2] = p;
                //tri[0].p[0] = VertexInterp(iso,g.p[v3],g.p[v0],g.val[v3],g.val[v0]);
                //tri[0].p[1] = VertexInterp(iso,g.p[v3],g.p[v2],g.val[v3],g.val[v2]);
                //tri[0].p[2] = VertexInterp(iso,g.p[v3],g.p[v1],g.val[v3],g.val[v1]);
                triOffset++;
                break;
        }
        if (triIdx != 0 && triIdx != 0x0F) {
            if (CubeValues[tets[tetIdx][0]] <= 0 && CubeValues[tets[tetIdx][1]] <= 0
                && CubeValues[tets[tetIdx][2]] <= 0 && CubeValues[tets[tetIdx][3]] <= 0) {
                *vol = (sjd->CellSize * sjd->CellSize * sjd->CellSize)
                    / static_cast<VoxelizerFloat>(6.0);
            }
        }
    }
}

DWORD TetraVoxelizer::Run(void *userData) {
    using vislib::sys::Log;

    int x, y, z;
    unsigned int vertFloatSize = 0;
    VoxelizerFloat currRad = 0.f;
    //, maxRad = -FLT_MAX;
    VoxelizerFloat currDist;
    vislib::math::Point<unsigned int, 3> pStart, pEnd;
    vislib::math::Point<VoxelizerFloat, 3> p;
    sjd = static_cast<SubJobData*>(userData);
    SIZE_T numNeg = 0, numZero = 0, numPos = 0;

    unsigned int fifoEnd = 0, fifoCur = 0;
    FatVoxel *volume = new FatVoxel[sjd->resX * sjd->resY * sjd->resZ];
    for (SIZE_T i = 0; i < sjd->resX * sjd->resY * sjd->resZ; i++) {
        volume[i].distField = FLT_MAX;
        volume[i].borderVoxel = NULL;
        volume[i].mcCase = 0;
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
    vislib::math::Point<VoxelizerFloat, 3> Centroid = bx.CalcCenter();
    VoxelizerFloat distOffset = vislib::math::Sqrt(bx.Width() * bx.Width() + bx.Height() * bx.Height()
        + bx.Depth() * bx.Depth()) / static_cast<VoxelizerFloat>(2.0);
    // TODO: what did this do anyway
    //VoxelizerFloat g = sjd->MaxRad * sjd->RadMult - sjd->MaxRad;
    //if (g > static_cast<VoxelizerFloat>(0)) {
    //    bx.Grow(2 * g);
    //}

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
            if (Centroid.Distance(sp) > currRad + distOffset) {
                continue;
            }
            x = static_cast<int>((sp.X() - currRad - sjd->Bounds.Left()) / sjd->CellSize) - 1;
            if (x < 0) x = 0;
            y = static_cast<int>((sp.Y() - currRad - sjd->Bounds.Bottom()) / sjd->CellSize) - 1;
            if (y < 0) y = 0;
            z = static_cast<int>((sp.Z() - currRad - sjd->Bounds.Back()) / sjd->CellSize) - 1;
            if (z < 0) z = 0;
            pStart.Set(x, y, z);

            x = static_cast<int>((sp.X() + currRad - sjd->Bounds.Left()) / sjd->CellSize) + 2;
            if (x >= static_cast<int>(sjd->resX)) x = sjd->resX - 1;
            y = static_cast<int>((sp.Y() + currRad - sjd->Bounds.Bottom()) / sjd->CellSize) + 2;
            if (y >= static_cast<int>(sjd->resY)) y = sjd->resY - 1;
            z = static_cast<int>((sp.Z() + currRad - sjd->Bounds.Back()) / sjd->CellSize) + 2;
            if (z >= static_cast<int>(sjd->resZ)) z = sjd->resZ - 1;
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
                        currDist = sp.Distance<VoxelizerFloat>(p) - currRad;
                        SIZE_T i = (z * sjd->resY + y) * sjd->resX + x;
                        //if (x > 5 && x < 8 && z > 4 && z < 8 && y > 6 && y < 9) {
                            volume[i].distField = vislib::math::Min(volume[i].distField, currDist);
                        //} else {
                        //    volume[i].distField = 10000000;
                        //}
                            if (volume[i].distField < 0.0) {
                                numNeg++;
                            } else {
                                if (volume[i].distField > 0.0) {
                                    numPos++;
                                } else {
                                    numZero++;
                                }
                            }
                    }
                }
            }
        }
    }


    // this is a trivial test case.
    // ----------------------------

    //unsigned int rx = sjd->resX;
    //unsigned int ry = sjd->resY;
    //unsigned int rz = sjd->resZ;
    //sjd->resX = sjd->resY = sjd->resZ = 2;

    //FatVoxel *tv = new FatVoxel[2 * 2 * 2];
    //tv[(0 * sjd->resY + 0) * sjd->resX + 0].distField = 0.0;
    //tv[(0 * sjd->resY + 0) * sjd->resX + 1].distField = -0.70710678118654752440084436210485;
    //tv[(0 * sjd->resY + 1) * sjd->resX + 1].distField = -0.70710678118654752440084436210485;
    //tv[(0 * sjd->resY + 1) * sjd->resX + 0].distField = 0.0;
    //tv[(1 * sjd->resY + 0) * sjd->resX + 0].distField = 0.70710678118654752440084436210485;
    //tv[(1 * sjd->resY + 0) * sjd->resX + 1].distField = 0.0;
    //tv[(1 * sjd->resY + 1) * sjd->resX + 1].distField = 0.0;
    //tv[(1 * sjd->resY + 1) * sjd->resX + 0].distField = 0.70710678118654752440084436210485;

    //MarchCell(tv, 0, 0, 0);

    //double mv = 0.0;
    //FatVoxel &currVoxel = tv[(0 * sjd->resY + 0) * sjd->resX + 0];
    //for (unsigned int x = 0; x < currVoxel.numTriangles; x++) {
    //    mv += currVoxel.volumes[x];
    //}

    //sjd->resX = rx;
    //sjd->resY = ry;
    //sjd->resZ = rz;

    // end test case.
    // --------------

    // does this really define an empty sub-volume?
    if (numNeg == (sjd->resX) * (sjd->resY) * (sjd->resZ)) {
        // totally full
        Surface s;
        s.surface = 0.0;
        s.volume = (sjd->resX - 1) * (sjd->resY - 1) * (sjd->resZ - 1)
                * sjd->CellSize * sjd->CellSize * sjd->CellSize;
        sjd->Result.surfaces.Append(s);
    } else if (numPos == (sjd->resX) * (sjd->resY) * (sjd->resZ)) {
        // totally empty
        Surface s;
        s.surface = 0.0;
        s.volume = 0.0;
        sjd->Result.surfaces.Append(s);
    } else {
        // march it
        for (x = 0; x < static_cast<int>(sjd->resX) - 1; x++) {
            for (y = 0; y < static_cast<int>(sjd->resY) - 1; y++) {
                for (z = 0; z < static_cast<int>(sjd->resZ) - 1; z++) {
                    MarchCell(volume, x, y, z);
                }
            }
        }

        // collect the surfaces
        for (x = 0; x < static_cast<int>(sjd->resX) - 1; x++) {
            for (y = 0; y < static_cast<int>(sjd->resY) - 1; y++) {
                for (z = 0; z < static_cast<int>(sjd->resZ) - 1; z++) {
                    CollectCell(volume, x, y, z);
                }
            }
        }

    }
    // dealloc stuff in volume
    // dealloc volume as a whole etc.
    for (x = 0; x < static_cast<int>(sjd->resX) - 1; x++) {
        for (y = 0; y < static_cast<int>(sjd->resY) - 1; y++) {
            for (z = 0; z < static_cast<int>(sjd->resZ) - 1; z++) {
                if (MarchingCubeTables::a2ucTriangleConnectionCount[volume[(z * sjd->resY + y) * sjd->resX + x].mcCase] > 0) {
                    ARY_SAFE_DELETE(volume[(z * sjd->resY + y) * sjd->resX + x].triangles);
                    ARY_SAFE_DELETE(volume[(z * sjd->resY + y) * sjd->resX + x].volumes);
                    // do NOT delete!
                    volume[(z * sjd->resY + y) * sjd->resX + x].borderVoxel = NULL;
                }
            }
        }
    }
    ARY_SAFE_DELETE(volume);


    sjd->Result.done = true;

    return 0;
}

bool TetraVoxelizer::Terminate(void) {
    terminate = true;
    return true;
}