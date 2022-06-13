#include "TetraVoxelizer.h"
#include "VoluMetricJob.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/utility/log/Log.h"
#include "trisoup/volumetrics/JobStructures.h"
#include "trisoup/volumetrics/MarchingCubeTables.h"
#include "vislib/math/ShallowPoint.h"
#include "vislib/math/Vector.h"
#include <cfloat>
#include <climits>

using namespace megamol;
using namespace megamol::trisoup_gl;
using namespace megamol::trisoup_gl::volumetrics;

int TetraVoxelizer::tets[6][4] = {{0, 2, 3, 7}, {0, 2, 6, 7}, {0, 4, 6, 7}, {0, 6, 1, 2}, {0, 6, 1, 4}, {5, 6, 1, 4}};

//#define ULTRADEBUG

vislib::math::Point<int, 3> TetraVoxelizer::cornerNeighbors[8][7] = {
    {vislib::math::Point<int, 3>(-1, 0, 0), vislib::math::Point<int, 3>(-1, 0, -1),
        vislib::math::Point<int, 3>(-1, -1, 0), vislib::math::Point<int, 3>(-1, -1, -1),
        vislib::math::Point<int, 3>(0, 0, -1), vislib::math::Point<int, 3>(0, -1, -1),
        vislib::math::Point<int, 3>(0, -1, 0)},

    {vislib::math::Point<int, 3>(1, 0, 0), vislib::math::Point<int, 3>(1, 0, -1), vislib::math::Point<int, 3>(1, -1, 0),
        vislib::math::Point<int, 3>(1, -1, -1), vislib::math::Point<int, 3>(0, 0, -1),
        vislib::math::Point<int, 3>(0, -1, -1), vislib::math::Point<int, 3>(0, -1, 0)},

    {vislib::math::Point<int, 3>(1, 1, 0), vislib::math::Point<int, 3>(1, 1, -1), vislib::math::Point<int, 3>(1, 0, 0),
        vislib::math::Point<int, 3>(1, 0, -1), vislib::math::Point<int, 3>(0, 1, -1),
        vislib::math::Point<int, 3>(0, 0, -1), vislib::math::Point<int, 3>(0, 1, 0)},

    {vislib::math::Point<int, 3>(-1, 1, 0), vislib::math::Point<int, 3>(-1, 1, -1),
        vislib::math::Point<int, 3>(-1, 0, 0), vislib::math::Point<int, 3>(-1, 0, -1),
        vislib::math::Point<int, 3>(0, 1, -1), vislib::math::Point<int, 3>(0, 0, -1),
        vislib::math::Point<int, 3>(0, 1, 0)},

    // front
    {vislib::math::Point<int, 3>(-1, 0, 1), vislib::math::Point<int, 3>(-1, 0, 0),
        vislib::math::Point<int, 3>(-1, -1, 1), vislib::math::Point<int, 3>(-1, -1, 0),
        vislib::math::Point<int, 3>(0, 0, 1), vislib::math::Point<int, 3>(0, -1, 1),
        vislib::math::Point<int, 3>(0, -1, 0)},

    {vislib::math::Point<int, 3>(1, 0, 1), vislib::math::Point<int, 3>(1, 0, 0), vislib::math::Point<int, 3>(1, -1, 1),
        vislib::math::Point<int, 3>(1, -1, 0), vislib::math::Point<int, 3>(0, 0, 1),
        vislib::math::Point<int, 3>(0, -1, 1), vislib::math::Point<int, 3>(0, -1, 0)},

    {vislib::math::Point<int, 3>(1, 1, 1), vislib::math::Point<int, 3>(1, 1, 0), vislib::math::Point<int, 3>(1, 0, 1),
        vislib::math::Point<int, 3>(1, 0, 0), vislib::math::Point<int, 3>(0, 1, 1),
        vislib::math::Point<int, 3>(0, 0, 1), vislib::math::Point<int, 3>(0, 1, 0)},

    {vislib::math::Point<int, 3>(-1, 1, 1), vislib::math::Point<int, 3>(-1, 1, 0),
        vislib::math::Point<int, 3>(-1, 0, 1), vislib::math::Point<int, 3>(-1, 0, 0),
        vislib::math::Point<int, 3>(0, 1, 1), vislib::math::Point<int, 3>(0, 0, 1),
        vislib::math::Point<int, 3>(0, 1, 0)}};

vislib::math::Point<int, 3> TetraVoxelizer::moreNeighbors[6] = {
    vislib::math::Point<int, 3>(-1, 0, 0),
    vislib::math::Point<int, 3>(1, 0, 0),
    vislib::math::Point<int, 3>(0, -1, 0),
    vislib::math::Point<int, 3>(0, 1, 0),
    vislib::math::Point<int, 3>(0, 0, -1),
    vislib::math::Point<int, 3>(0, 0, 1),
};

void TetraVoxelizer::debugPrintTriangle(vislib::math::ShallowShallowTriangle<float, 3>& tri) {
    megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO,
        "[%08u] (%03.3f, %03.3f, %03.3f), (%03.3f, %03.3f, %03.3f), (%03.3f, %03.3f, %03.3f)",
        vislib::sys::Thread::CurrentID(), tri.PeekCoordinates()[0][0], tri.PeekCoordinates()[0][1],
        tri.PeekCoordinates()[0][2], tri.PeekCoordinates()[1][0], tri.PeekCoordinates()[1][1],
        tri.PeekCoordinates()[1][2], tri.PeekCoordinates()[2][0], tri.PeekCoordinates()[2][1],
        tri.PeekCoordinates()[2][2]);
}

void TetraVoxelizer::debugPrintTriangle(vislib::math::ShallowShallowTriangle<double, 3>& tri) {
    megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO,
        "[%08u] (%03.3lf, %03.3lf, %03.3lf), (%03.3lf, %03.3lf, %03.3lf), (%03.3lf, %03.3lf, %03.3lf)",
        vislib::sys::Thread::CurrentID(), tri.PeekCoordinates()[0][0], tri.PeekCoordinates()[0][1],
        tri.PeekCoordinates()[0][2], tri.PeekCoordinates()[1][0], tri.PeekCoordinates()[1][1],
        tri.PeekCoordinates()[1][2], tri.PeekCoordinates()[2][0], tri.PeekCoordinates()[2][1],
        tri.PeekCoordinates()[2][2]);
}

TetraVoxelizer::TetraVoxelizer(void) : terminate(false), sjd(NULL) {
    //triangleSoup.SetCapacityIncrement(90); // AKA 10 triangles?
}


TetraVoxelizer::~TetraVoxelizer(void) {}

bool TetraVoxelizer::CellHasNoGeometry(trisoup::volumetrics::FatVoxel* theVolume, unsigned x, unsigned y, unsigned z) {
    //   unsigned int i;
    //   bool neg = false, pos = false;
    //   trisoup::volumetrics::VoxelizerFloat f;

    //   for (i = 0; i < 8; i++) {
    //       f = theVolume[sjd->cellIndx(
    //               x + MarchingCubeTables::a2fVertexOffset[i][0],
    //               y + MarchingCubeTables::a2fVertexOffset[i][1],
    //               z + MarchingCubeTables::a2fVertexOffset[i][2])
    //       ].distField;
    //       neg = neg | (f < 0.0);
    //       pos = pos | (f >= 0.0);
    //   }

    unsigned index = sjd->cellIndex(x, y, z);

    //ASSERT ((theVolume[index].mcCase == 255 || theVolume[index].mcCase == 0)
    //    == !(neg &&pos));
    //   return !(neg && pos);

    return theVolume[index].mcCase == 255 || theVolume[index].mcCase == 0;
}

bool TetraVoxelizer::CellFull(trisoup::volumetrics::FatVoxel* theVolume, unsigned x, unsigned y, unsigned z) {
    //unsigned int i;
    //bool neg = true;
    //float f;

    //for (i = 0; i < 8; i++) {
    //       f = theVolume[sjd->cellIndx(
    //               x + MarchingCubeTables::a2fVertexOffset[i][0],
    //               y + MarchingCubeTables::a2fVertexOffset[i][1],
    //               z + MarchingCubeTables::a2fVertexOffset[i][2])
    //       ].distField;
    //    neg = neg && (f < 0.0f);
    //}
    //ASSERT (neg == (theVolume[sjd->cellIndex(x, y, z)].mcCase == 255));
    //return neg;
    return theVolume[sjd->cellIndex(x, y, z)].mcCase == 255;
}

void TetraVoxelizer::CollectCell(
    trisoup::volumetrics::FatVoxel* theVolume, unsigned int x, unsigned int y, unsigned int z) {
    if (CellHasNoGeometry(theVolume, x, y, z))
        return;

    trisoup::volumetrics::FatVoxel& cell = theVolume[sjd->cellIndex(x, y, z)];

    // WTF ?
    //    if (cell.numTriangles > 0)
    //        vislib::math::ShallowShallowTriangle<VoxelizerFloat, 3> sst(cell.triangles);

    //vislib::math::ShallowShallowTriangle<float, 3> sst2(cell.triangles);
#ifdef ULTRADEBUG
    megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO,
        "[%08u] collecting (%04u, %04u, %04u)\n", vislib::sys::Thread::CurrentID(), x, y, z);
#endif /* ULTRADEBUG */
    for (unsigned int triIdx = 0; triIdx < cell.numTriangles; triIdx++) {
        if (cell.consumedTriangles & (1 << triIdx))
            continue;

        // this is a new surface
        trisoup::volumetrics::Surface surf;
        surf.border->SetCapacityIncrement(10);
        surf.mesh.SetCapacityIncrement(90);
        surf.surface = static_cast<trisoup::volumetrics::VoxelizerFloat>(0.0);
        surf.volume = static_cast<trisoup::volumetrics::VoxelizerFloat>(0.0);
        surf.voidVolume = static_cast<trisoup::volumetrics::VoxelizerFloat>(0.0); // this is empty as well ...
        surf.fullFaces = 0;
        surf.globalID = UINT_MAX;

        for (SIZE_T cellIdx = 0; cellIdx < static_cast<SIZE_T>(sjd->resX * sjd->resY * sjd->resZ); cellIdx++)
            theVolume[cellIdx].borderVoxel = NULL;

        cellFIFO.Append(vislib::math::Point<unsigned int, 4>(x, y, z, triIdx));
#ifdef ULTRADEBUG
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO,
            "[%08u] appending  (%04u, %04u, %04u)[%u]\n", vislib::sys::Thread::CurrentID(), x, y, z, l);
#endif /* ULTRADEBUG */
        while (cellFIFO.Count() > 0) {
            vislib::math::Point<unsigned int, 4> p = cellFIFO.First();
            cellFIFO.RemoveFirst();
#ifdef ULTRADEBUG
            megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO,
                "[%08u] growing    (%04u, %04u, %04u)[%u]\n", vislib::sys::Thread::CurrentID(), p.X(), p.Y(), p.Z(),
                p.W());
#endif /* ULTRADEBUG */
            growSurfaceFromTriangle(theVolume, p.X(), p.Y(), p.Z(), p.W(), surf);
        }
        sjd->Result.surfaces.Append(surf);

        // thomasbm:
        //#error das kann so nicht gehen ... weil immer  cell.enclosingCandidate == &surf gilt!?
        //if (cell.enclosingCandidate && cell.enclosingCandidate != &surf) {
        //    cell.enclosingCandidate->enclSurfaces.Append(&surf);
        //}
    } /* end for */
}

trisoup::volumetrics::VoxelizerFloat TetraVoxelizer::GetOffset(trisoup::volumetrics::VoxelizerFloat fValue1,
    trisoup::volumetrics::VoxelizerFloat fValue2, trisoup::volumetrics::VoxelizerFloat fValueDesired) {
    trisoup::volumetrics::VoxelizerFloat fDelta = fValue2 - fValue1;
    ASSERT(fDelta != static_cast<trisoup::volumetrics::VoxelizerFloat>(0));
    trisoup::volumetrics::VoxelizerFloat res = (fValueDesired - fValue1) / fDelta;
    ASSERT(res <= static_cast<trisoup::volumetrics::VoxelizerFloat>(1) &&
           res >= static_cast<trisoup::volumetrics::VoxelizerFloat>(0));
    return res;
}

trisoup::volumetrics::VoxelizerFloat TetraVoxelizer::growVolume(trisoup::volumetrics::FatVoxel* theVolume,
    trisoup::volumetrics::Surface& surf, const vislib::math::Point<int, 3>& seed, bool emptyVolume) {
    SIZE_T cells = 0;
    vislib::math::Point<int, 3> p;
    vislib::Array<vislib::math::Point<int, 3>> queue;
    queue.SetCapacityIncrement(128);
    queue.Add(seed);

    /* avoid recursion using a queue */
    while (queue.Count() > 0) {
        p = queue.Last();
        queue.RemoveLast();
        trisoup::volumetrics::FatVoxel& cell = theVolume[sjd->cellIndex(p)];

        if (!emptyVolume) {
            ASSERT(cell.mcCase == 255 && vislib::math::Abs(cell.consumedTriangles) < 2);
            // nach dem assert kann man sich das if sparen ...
            /*if (cell.mcCase != 255 || cell.consumedTriangles >= 2) continue; */
        } else {
            ASSERT(cell.mcCase == 0 && vislib::math::Abs(cell.consumedTriangles) < 2);
            /*if (cell.mcCase != 0 || cell.consumedTriangles >= 2) continue; */
        }

        {
            cells++;
            if (!emptyVolume) {
                /* this is some sort of 'already processed' flag ...?! */
                cell.consumedTriangles = 2;
                if (p.X() == 0)
                    surf.fullFaces |= 1;
                if (p.Y() == 0)
                    surf.fullFaces |= 4;
                if (p.Z() == 0)
                    surf.fullFaces |= 16;
                if (p.X() == sjd->resX - 2)
                    surf.fullFaces |= 2;
                if (p.Y() == sjd->resY - 2)
                    surf.fullFaces |= 8;
                if (p.Z() == sjd->resZ - 2)
                    surf.fullFaces |= 32;
            } else
                cell.consumedTriangles = -2;

#ifdef ULTRADEBUG
            megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO,
                "[%08u] grew to (%04u, %04u, %04u)", vislib::sys::Thread::CurrentID(), p.X(), p.Y(), p.Z());
#endif /* ULTRADEBUG */

            for (unsigned int neighbIdx = 0; neighbIdx < 6; neighbIdx++) {
                const vislib::math::Point<int, 3>& mN = moreNeighbors[neighbIdx];
                /* coordinates may become negative here -> signed integer */
                vislib::math::Point<int, 3> neighbCrd(p.X() + mN.X(), p.Y() + mN.Y(), p.Z() + mN.Z());

                if (sjd->coordsInside(neighbCrd)) {
                    trisoup::volumetrics::FatVoxel& neighbCell = theVolume[sjd->cellIndex(neighbCrd)];

                    if (!emptyVolume) {
                        if (neighbCell.mcCase == 255 && neighbCell.consumedTriangles == 0) {
                            neighbCell.consumedTriangles = -1; /* empty - negative*/
                            queue.Add(neighbCrd);              // recursion ...
                        }
                    } else {
                        // thomasbm: surfaces inside this cell might be enclosed by 'surf'
                        //    if (!CellHasNoGeometry(theVolume, x, y, z)
                        //        neighbCell.enclosingCandidate = &surf;
                        if (neighbCell.mcCase == 0 && neighbCell.consumedTriangles == 0) {
                            neighbCell.consumedTriangles = 1;
                            queue.Add(neighbCrd); // recursion ...
                        }
                    }
                }
            }
        }
    } /* end while (queue.Count() > 0) */

#ifdef ULTRADEBUG
    if (cells > 0) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO,
            "[%08u] grew volume from (%04u, %04u, %04u) yielding %u cells and a volume of %f",
            vislib::sys::Thread::CurrentID(), x, y, z, cells, cells * sjd->CellSize * sjd->CellSize * sjd->CellSize);
    }
#endif /* ULTRADEBUG */
    return cells * sjd->CellSize * sjd->CellSize * sjd->CellSize;
}

#define __STR2__(x) #x
#define __STR1__(x) __STR2__(x)
#define __LOC__ __FILE__ "("__STR1__(__LINE__) ") : Warning Msg: "

/**
 * TODO: sinnvoller Kommentar! bissle erklaeren!
 */
void TetraVoxelizer::growSurfaceFromTriangle(trisoup::volumetrics::FatVoxel* theVolume, unsigned int x, unsigned int y,
    unsigned int z, unsigned seedTriIndex, trisoup::volumetrics::Surface& surf) {
    typedef vislib::math::ShallowShallowTriangle<trisoup::volumetrics::VoxelizerFloat, 3> Triangle;

    trisoup::volumetrics::FatVoxel& cell = theVolume[sjd->cellIndex(x, y, z)];
    //int currSurfID = MarchingCubeTables::a2ucTriangleSurfaceID[cell.mcCase][seedTriIndex];

    // first, grow the full neighbors
    // thomasbm: now we grow full and empty neighbours (volume and voidVolume) ...
    for (unsigned int cornerIdx = 0; cornerIdx < 8; cornerIdx++) {
        //if (!(cell.mcCase & (1 << cornerIdx))) continue;
        //#pragma message(__LOC__"Guido's Code  mit cell.corners wurde hier auskommentiert - liegt der Bug wirklich daran?!")
        int fullNeighb = cell.mcCase & (1 << cornerIdx) & cell.corners[seedTriIndex];

        for (unsigned int cornerNeighbIdx = 0; cornerNeighbIdx < 7; cornerNeighbIdx++) {
            vislib::math::Point<int, 3>& cN = cornerNeighbors[cornerIdx][cornerNeighbIdx];
            /* coordinates may become negative here -> signed integer */
            vislib::math::Point<int, 3> crnCrd(x + cN.X(), y + cN.Y(), z + cN.Z());

            if (sjd->coordsInside(crnCrd)) {
                // der aktuelle Nachbar
                trisoup::volumetrics::FatVoxel& neighbCell = theVolume[sjd->cellIndex(crnCrd)];

                if (fullNeighb) {
                    // 'neighbCell' located completely inside?
                    if (neighbCell.mcCase == 255 && neighbCell.consumedTriangles == 0)
                        surf.volume += growVolume(theVolume, surf, crnCrd, false);
                } else {
                    // 'neighbCell' located completely outside?
                    if (neighbCell.mcCase == 0 && neighbCell.consumedTriangles == 0)
                        surf.voidVolume += growVolume(theVolume, surf, crnCrd, true);
                }
            }
        }
    }

    // seed triangle
    unsigned short inCellSurf = 1 << seedTriIndex;
    bool foundNew;

    // find all in-cell triangles connected with the seed triangle
    // TODO this is slow and very expensive
    do {
        foundNew = false;
        for (int triIdx = 0; triIdx < cell.numTriangles; triIdx++) {
            if ((inCellSurf & (1 << triIdx)))
                continue;
            // we haven't been here before, or it did not fit.
            Triangle triangle(cell.triangles + 3 * 3 * triIdx);

            // does it fit to any of the collected triangles?
            for (int niTriIdx = 0; niTriIdx < cell.numTriangles; niTriIdx++) {
                if (inCellSurf & (1 << niTriIdx)) {
                    Triangle neighbTriangle(cell.triangles + 3 * 3 * niTriIdx);
                    //if (triangle.HasCommonEdge(neighbTriangle)) {
                    if (trisoup::volumetrics::Dowel::HaveCommonEdge(triangle, neighbTriangle)) {
                        inCellSurf |= (1 << triIdx);
                        foundNew = true;
                    }
                }
            }
        }
    } while (foundNew);

    trisoup::volumetrics::VoxelizerFloat cellVolume = 0;
    bool collected = false;
    for (int triIdx = 0; triIdx < cell.numTriangles; triIdx++) {
        // is this part of the in-cell surface?
        if (!(inCellSurf & (1 << triIdx)))
            continue;

        Triangle triangle(cell.triangles + 3 * 3 * triIdx);
        if (!(cell.consumedTriangles & (1 << triIdx))) {
            ProcessTriangle(triangle, cell, triIdx, surf, x, y, z);
            cell.consumedTriangles |= (1 << triIdx);
            cellVolume += cell.volumes[triIdx];
            collected = true;
        }

        // loop over all 6 neighbour cells ...
        for (int neighbIdx = 0; neighbIdx < 6; neighbIdx++) {
            const vislib::math::Point<int, 3>& mN = moreNeighbors[neighbIdx];
            /* coordinates may become negative here -> signed integer */
            const vislib::math::Point<int, 3> neighbCrd(x + mN.X(), y + mN.Y(), z + mN.Z());

            if (!sjd->coordsInside(neighbCrd))
                continue;

            trisoup::volumetrics::FatVoxel& neighbCell = theVolume[sjd->cellIndex(neighbCrd)];
            trisoup::volumetrics::VoxelizerFloat niCellVolume = 0;
            bool niCollected = false;

            for (int niTriIdx = 0; niTriIdx < neighbCell.numTriangles; niTriIdx++) {
                if (neighbCell.consumedTriangles & (1 << niTriIdx))
                    continue;

                Triangle neighbTriangle(neighbCell.triangles + 3 * 3 * niTriIdx);
#ifdef ULTRADEBUG
                megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO,
                    "[%08u] comparing with (%04u, %04u, %04u)[%u/%u]", vislib::sys::Thread::CurrentID(), neighbCrd.X(),
                    neighbCrd.Y(), neighbCrd.Z(), niTriIdx, neighbCell.numTriangles);
                debugPrintTriangle(neighbTriangle);
                debugPrintTriangle(triangle);
#endif /* ULTRADEBUG */
                if (trisoup::volumetrics::Dowel::HaveCommonEdge(neighbTriangle, triangle)) {
#ifdef ULTRADEBUG
                    megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO,
                        "[%08u] -> has common edge", vislib::sys::Thread::CurrentID());
#endif /* ULTRADEBUG */
                    /*if (!(neighbCell.consumedTriangles & (1 << niTriIdx)))*/
                    {
                        ProcessTriangle(
                            neighbTriangle, neighbCell, niTriIdx, surf, neighbCrd.X(), neighbCrd.Y(), neighbCrd.Z());
                        neighbCell.consumedTriangles |= (1 << niTriIdx);
                        niCellVolume += neighbCell.volumes[niTriIdx];
                        niCollected = true;
                    }
                    /* this causes a recursive mechanism using a qeue */
                    cellFIFO.Append(
                        vislib::math::Point<unsigned, 4>(neighbCrd.X(), neighbCrd.Y(), neighbCrd.Z(), niTriIdx));
                }
            }

            // collect volume for the neighbour cell
            if (niCollected) // (niCellVolume > 0)
            {
                surf.volume += niCellVolume;
                surf.voidVolume += (sjd->CellSize * sjd->CellSize * sjd->CellSize - niCellVolume);
            }
        }
    }

    // only add volume of cells that contain triangles ...
    if (collected) {
        surf.volume += cellVolume;
        surf.voidVolume += (sjd->CellSize * sjd->CellSize * sjd->CellSize - cellVolume);
    }
}

/**
 * Adds the triangle 'triangle' with index 'triIdx' inside 'cell' to 'surf' and collects trisoup::volumetrics::Surface and volume data.
 */
VISLIB_FORCEINLINE void TetraVoxelizer::ProcessTriangle(
    vislib::math::ShallowShallowTriangle<trisoup::volumetrics::VoxelizerFloat, 3>& triangle,
    trisoup::volumetrics::FatVoxel& cell, unsigned triIdx, trisoup::volumetrics::Surface& surf, unsigned int x,
    unsigned int y, unsigned int z) {

    vislib::math::ShallowShallowTriangle<trisoup::volumetrics::VoxelizerFloat, 3> tmpTriangle(
        cell.triangles + 3 * 3 * triIdx);

    /* copy 'triangle' to 'surf.mesh' if we want to store the geometry */
    if (sjd->storeMesh) {
        surf.mesh.SetCount(surf.mesh.Count() + 9);
        tmpTriangle.SetPointer(
            const_cast<trisoup::volumetrics::VoxelizerFloat*>(surf.mesh.PeekElements() + surf.mesh.Count() - 9));
        tmpTriangle = triangle;
    }

    surf.surface += triangle.Area<trisoup::volumetrics::VoxelizerFloat>();
    //    surf.volume += cell.triangles[triIdx];

    // thomasbm: grow bounding volume based on intersecting voxels ...
    vislib::math::Point<unsigned, 3> voxelCoords(x + sjd->offsetX, y + sjd->offsetY, z + sjd->offsetZ);
    surf.boundingBox.AddPoint(voxelCoords);

    if (sjd->isBorder(x, y, z)) {
        if (cell.borderVoxel == NULL) {
            cell.borderVoxel = new trisoup::volumetrics::BorderVoxel();
            cell.borderVoxel->x = x + sjd->offsetX;
            cell.borderVoxel->y = y + sjd->offsetY;
            cell.borderVoxel->z = z + sjd->offsetZ;
            cell.borderVoxel->triangles.AssertCapacity(cell.numTriangles * 9);
            surf.border->Add(cell.borderVoxel);
        }
        cell.borderVoxel->triangles.SetCount(cell.borderVoxel->triangles.Count() + 9);
        tmpTriangle.SetPointer(const_cast<trisoup::volumetrics::VoxelizerFloat*>(
            cell.borderVoxel->triangles.PeekElements() + cell.borderVoxel->triangles.Count() - 9));
        tmpTriangle = triangle;
    }
#ifdef ULTRADEBUG
    megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO,
        "[%08u] consuming  (%04u, %04u, %04u)[%u/%u]"
        " (%03.3f, %03.3f, %03.3f), (%03.3f, %03.3f, %03.3f), (%03.3f, %03.3f, %03.3f)\n",
        vislib::sys::Thread::CurrentID(), x, y, z, triIdx, MarchingCubeTables::a2ucTriangleConnectionCount[n->mcCase],
        triangle.PeekCoordinates()[0][0], triangle.PeekCoordinates()[0][1], triangle.PeekCoordinates()[0][2],
        triangle.PeekCoordinates()[1][0], triangle.PeekCoordinates()[1][1], triangle.PeekCoordinates()[1][2],
        triangle.PeekCoordinates()[2][0], triangle.PeekCoordinates()[2][1], triangle.PeekCoordinates()[2][2]);
#endif /* ULTRADEBUG */
}

void TetraVoxelizer::MarchCell(
    trisoup::volumetrics::FatVoxel* theVolume, unsigned int x, unsigned int y, unsigned int z) {

    trisoup::volumetrics::FatVoxel& currVoxel = theVolume[sjd->cellIndex(x, y, z)];
    currVoxel.consumedTriangles = 0;
    currVoxel.numTriangles = 0;

    unsigned int i;
    trisoup::volumetrics::VoxelizerFloat CubeValues[8];
    vislib::math::Point<trisoup::volumetrics::VoxelizerFloat, 3> EdgeVertex[12];

    currVoxel.mcCase = 0;
    //Make a local copy of the values at the cube's corners
    for (i = 0; i < 8; i++) {
        CubeValues[i] = theVolume[sjd->cellIndex(x + trisoup::volumetrics::MarchingCubeTables::a2fVertexOffset[i][0],
                                      y + trisoup::volumetrics::MarchingCubeTables::a2fVertexOffset[i][1],
                                      z + trisoup::volumetrics::MarchingCubeTables::a2fVertexOffset[i][2])]
                            .distField;
        if (CubeValues[i] < 0.0f)
            currVoxel.mcCase |= 1 << i;
    }
    //CellFull(theVolume, x, y, z);
    if (CellHasNoGeometry(theVolume, x, y, z)) { // || !((x==6) && (y==7) && (z==6))) {
        currVoxel.consumedTriangles = 0;
        currVoxel.triangles = NULL;
        currVoxel.volumes = NULL;
        currVoxel.corners = NULL;
        currVoxel.numTriangles = 0;
        return;
    }

    // reference corner of this cell
    vislib::math::Point<trisoup::volumetrics::VoxelizerFloat, 3> p(sjd->Bounds.Left() + x * sjd->CellSize,
        sjd->Bounds.Bottom() + y * sjd->CellSize, sjd->Bounds.Back() + z * sjd->CellSize);

    // how many triangles will we get?
    for (int tetIdx = 0; tetIdx < 6; tetIdx++) {
        int triIdx = 0;
        if (CubeValues[tets[tetIdx][0]] < 0.0f)
            triIdx |= 1;
        if (CubeValues[tets[tetIdx][1]] < 0.0f)
            triIdx |= 2;
        if (CubeValues[tets[tetIdx][2]] < 0.0f)
            triIdx |= 4;
        if (CubeValues[tets[tetIdx][3]] < 0.0f)
            triIdx |= 8;

        switch (triIdx) {
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

    currVoxel.triangles = new trisoup::volumetrics::VoxelizerFloat[currVoxel.numTriangles * 3 * 3];
    currVoxel.volumes = new trisoup::volumetrics::VoxelizerFloat[currVoxel.numTriangles];
    currVoxel.corners = new unsigned char[currVoxel.numTriangles];
    vislib::math::ShallowShallowTriangle<trisoup::volumetrics::VoxelizerFloat, 3> tri(currVoxel.triangles);
    vislib::math::ShallowShallowTriangle<trisoup::volumetrics::VoxelizerFloat, 3> tri2(currVoxel.triangles);
    vislib::math::Point<trisoup::volumetrics::VoxelizerFloat, 3> temp;
    trisoup::volumetrics::VoxelizerFloat* vol = NULL;
    trisoup::volumetrics::VoxelizerFloat* vol2 = NULL;
    int triOffset = 0;

    // now we repeat this for all six sub-tetrahedra
    for (int tetIdx = 0; tetIdx < 6; tetIdx++) {
        int triIdx = 0;
        if (CubeValues[tets[tetIdx][0]] < 0.0f)
            triIdx |= 1;
        if (CubeValues[tets[tetIdx][1]] < 0.0f)
            triIdx |= 2;
        if (CubeValues[tets[tetIdx][2]] < 0.0f)
            triIdx |= 4;
        if (CubeValues[tets[tetIdx][3]] < 0.0f)
            triIdx |= 8;

        vislib::math::Point<trisoup::volumetrics::VoxelizerFloat, 3> p0(
            p.X() + (trisoup::volumetrics::VoxelizerFloat)
                            trisoup::volumetrics::MarchingCubeTables::a2fVertexOffset[tets[tetIdx][0]][0] *
                        sjd->CellSize,
            p.Y() + (trisoup::volumetrics::VoxelizerFloat)
                            trisoup::volumetrics::MarchingCubeTables::a2fVertexOffset[tets[tetIdx][0]][1] *
                        sjd->CellSize,
            p.Z() + (trisoup::volumetrics::VoxelizerFloat)
                            trisoup::volumetrics::MarchingCubeTables::a2fVertexOffset[tets[tetIdx][0]][2] *
                        sjd->CellSize);
        vislib::math::Point<trisoup::volumetrics::VoxelizerFloat, 3> p1(
            p.X() + (trisoup::volumetrics::VoxelizerFloat)
                            trisoup::volumetrics::MarchingCubeTables::a2fVertexOffset[tets[tetIdx][1]][0] *
                        sjd->CellSize,
            p.Y() + (trisoup::volumetrics::VoxelizerFloat)
                            trisoup::volumetrics::MarchingCubeTables::a2fVertexOffset[tets[tetIdx][1]][1] *
                        sjd->CellSize,
            p.Z() + (trisoup::volumetrics::VoxelizerFloat)
                            trisoup::volumetrics::MarchingCubeTables::a2fVertexOffset[tets[tetIdx][1]][2] *
                        sjd->CellSize);
        vislib::math::Point<trisoup::volumetrics::VoxelizerFloat, 3> p2(
            p.X() + (trisoup::volumetrics::VoxelizerFloat)
                            trisoup::volumetrics::MarchingCubeTables::a2fVertexOffset[tets[tetIdx][2]][0] *
                        sjd->CellSize,
            p.Y() + (trisoup::volumetrics::VoxelizerFloat)
                            trisoup::volumetrics::MarchingCubeTables::a2fVertexOffset[tets[tetIdx][2]][1] *
                        sjd->CellSize,
            p.Z() + (trisoup::volumetrics::VoxelizerFloat)
                            trisoup::volumetrics::MarchingCubeTables::a2fVertexOffset[tets[tetIdx][2]][2] *
                        sjd->CellSize);
        vislib::math::Point<trisoup::volumetrics::VoxelizerFloat, 3> p3(
            p.X() + (trisoup::volumetrics::VoxelizerFloat)
                            trisoup::volumetrics::MarchingCubeTables::a2fVertexOffset[tets[tetIdx][3]][0] *
                        sjd->CellSize,
            p.Y() + (trisoup::volumetrics::VoxelizerFloat)
                            trisoup::volumetrics::MarchingCubeTables::a2fVertexOffset[tets[tetIdx][3]][1] *
                        sjd->CellSize,
            p.Z() + (trisoup::volumetrics::VoxelizerFloat)
                            trisoup::volumetrics::MarchingCubeTables::a2fVertexOffset[tets[tetIdx][3]][2] *
                        sjd->CellSize);

        trisoup::volumetrics::VoxelizerFloat fullVol = vislib::math::Abs((p1 - p0).Dot((p2 - p0).Cross(p3 - p0))) /
                                                       static_cast<trisoup::volumetrics::VoxelizerFloat>(6.0);

        tri.SetPointer(currVoxel.triangles + 3 * 3 * triOffset);
        vol = currVoxel.volumes + triOffset;
        switch (triIdx) {
        case 0x00:
        case 0x0F:
            break;
        case 0x0E:
        case 0x01:
            //if (CubeValues[tets[tetIdx][0]] > 0.0f) {
            //} else {

            tri[0] = p0.Interpolate(p1, GetOffset(CubeValues[tets[tetIdx][0]], CubeValues[tets[tetIdx][1]],
                                            static_cast<trisoup::volumetrics::VoxelizerFloat>(0)));
            tri[2] = p0.Interpolate(p2, GetOffset(CubeValues[tets[tetIdx][0]], CubeValues[tets[tetIdx][2]],
                                            static_cast<trisoup::volumetrics::VoxelizerFloat>(0)));
            tri[1] = p0.Interpolate(p3, GetOffset(CubeValues[tets[tetIdx][0]], CubeValues[tets[tetIdx][3]],
                                            static_cast<trisoup::volumetrics::VoxelizerFloat>(0)));

            *vol = vislib::math::Abs((tri[0] - p0).Dot((tri[2] - p0).Cross(tri[1] - p0))) /
                   static_cast<trisoup::volumetrics::VoxelizerFloat>(6.0);
            if (CubeValues[tets[tetIdx][0]] > 0.0) {
                *vol = fullVol - *vol;
                // any but 0
                currVoxel.corners[triOffset] = 1 << tets[tetIdx][1];
                currVoxel.corners[triOffset] |= 1 << tets[tetIdx][2];
                currVoxel.corners[triOffset] |= 1 << tets[tetIdx][3];
            } else {
                currVoxel.corners[triOffset] = 1 << tets[tetIdx][0];
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
            tri[0] = p1.Interpolate(p0, GetOffset(CubeValues[tets[tetIdx][1]], CubeValues[tets[tetIdx][0]],
                                            static_cast<trisoup::volumetrics::VoxelizerFloat>(0)));
            tri[1] = p1.Interpolate(p3, GetOffset(CubeValues[tets[tetIdx][1]], CubeValues[tets[tetIdx][3]],
                                            static_cast<trisoup::volumetrics::VoxelizerFloat>(0)));
            tri[2] = p1.Interpolate(p2, GetOffset(CubeValues[tets[tetIdx][1]], CubeValues[tets[tetIdx][2]],
                                            static_cast<trisoup::volumetrics::VoxelizerFloat>(0)));

            *vol = vislib::math::Abs((tri[0] - p1).Dot((tri[1] - p1).Cross(tri[2] - p1))) /
                   static_cast<trisoup::volumetrics::VoxelizerFloat>(6.0);
            if (CubeValues[tets[tetIdx][1]] > 0.0) {
                *vol = fullVol - *vol;
                // any but 1
                currVoxel.corners[triOffset] = 1 << tets[tetIdx][0];
                currVoxel.corners[triOffset] |= 1 << tets[tetIdx][2];
                currVoxel.corners[triOffset] |= 1 << tets[tetIdx][3];
            } else {
                currVoxel.corners[triOffset] = 1 << tets[tetIdx][1];
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
            tri[0] = p0.Interpolate(p3, GetOffset(CubeValues[tets[tetIdx][0]], CubeValues[tets[tetIdx][3]],
                                            static_cast<trisoup::volumetrics::VoxelizerFloat>(0)));
            tri[1] = p0.Interpolate(p2, GetOffset(CubeValues[tets[tetIdx][0]], CubeValues[tets[tetIdx][2]],
                                            static_cast<trisoup::volumetrics::VoxelizerFloat>(0)));
            tri[2] = p1.Interpolate(p3, GetOffset(CubeValues[tets[tetIdx][1]], CubeValues[tets[tetIdx][3]],
                                            static_cast<trisoup::volumetrics::VoxelizerFloat>(0)));

            // tet3
            *vol = vislib::math::Abs((p0 - p1).Dot((tri[2] - p1).Cross(tri[1] - p1))) /
                   static_cast<trisoup::volumetrics::VoxelizerFloat>(6.0);
            // tet2
            *vol += vislib::math::Abs((tri[0] - p0).Dot((tri[2] - p0).Cross(tri[1] - p0))) /
                    static_cast<trisoup::volumetrics::VoxelizerFloat>(6.0);
            if (CubeValues[tets[tetIdx][0]] > 0.0) {
                // any but 0, 1
                currVoxel.corners[triOffset] = 1 << tets[tetIdx][2];
            } else {
                currVoxel.corners[triOffset] = 1 << tets[tetIdx][0];
            }
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
            tri2[1] = p1.Interpolate(p2, GetOffset(CubeValues[tets[tetIdx][1]], CubeValues[tets[tetIdx][2]],
                                             static_cast<trisoup::volumetrics::VoxelizerFloat>(0)));
            tri2[2] = tri[1];
            // tet1
            *vol2 = vislib::math::Abs((tri2[1] - p1).Dot((tri[1] - p1).Cross(tri[2] - p1))) /
                    static_cast<trisoup::volumetrics::VoxelizerFloat>(6.0);
            //tri2[0] = p;
            //tri2[1] = p;
            //tri2[2] = p;
            //tri[1].p[0] = tri[0].p[2];
            //tri[1].p[1] = VertexInterp(iso,g.p[v1],g.p[v2],g.val[v1],g.val[v2]);
            //tri[1].p[2] = tri[0].p[1];

            if (CubeValues[tets[tetIdx][0]] > 0.0) {
                *vol = fullVol - (*vol + *vol2);
                *vol2 = 0.0;
                // any but 0, 1
                currVoxel.corners[triOffset] = 1 << tets[tetIdx][3];
            } else {
                currVoxel.corners[triOffset] = 1 << tets[tetIdx][1];
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
            tri[0] = p2.Interpolate(p0, GetOffset(CubeValues[tets[tetIdx][2]], CubeValues[tets[tetIdx][0]],
                                            static_cast<trisoup::volumetrics::VoxelizerFloat>(0)));
            tri[1] = p2.Interpolate(p1, GetOffset(CubeValues[tets[tetIdx][2]], CubeValues[tets[tetIdx][1]],
                                            static_cast<trisoup::volumetrics::VoxelizerFloat>(0)));
            tri[2] = p2.Interpolate(p3, GetOffset(CubeValues[tets[tetIdx][2]], CubeValues[tets[tetIdx][3]],
                                            static_cast<trisoup::volumetrics::VoxelizerFloat>(0)));

            *vol = vislib::math::Abs((tri[0] - p2).Dot((tri[1] - p2).Cross(tri[2] - p2))) /
                   static_cast<trisoup::volumetrics::VoxelizerFloat>(6.0);
            if (CubeValues[tets[tetIdx][2]] > 0.0) {
                *vol = fullVol - *vol;
                // any but 2
                currVoxel.corners[triOffset] = 1 << tets[tetIdx][0];
                currVoxel.corners[triOffset] |= 1 << tets[tetIdx][1];
                currVoxel.corners[triOffset] |= 1 << tets[tetIdx][3];
            } else {
                currVoxel.corners[triOffset] = 1 << tets[tetIdx][2];
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
            tri[0] = p0.Interpolate(p1, GetOffset(CubeValues[tets[tetIdx][0]], CubeValues[tets[tetIdx][1]],
                                            static_cast<trisoup::volumetrics::VoxelizerFloat>(0)));
            tri[1] = p2.Interpolate(p3, GetOffset(CubeValues[tets[tetIdx][2]], CubeValues[tets[tetIdx][3]],
                                            static_cast<trisoup::volumetrics::VoxelizerFloat>(0)));
            tri[2] = p0.Interpolate(p3, GetOffset(CubeValues[tets[tetIdx][0]], CubeValues[tets[tetIdx][3]],
                                            static_cast<trisoup::volumetrics::VoxelizerFloat>(0)));

            // tet2
            *vol = vislib::math::Abs((tri[2] - p0).Dot((tri[1] - p0).Cross(tri[0] - p0))) /
                   static_cast<trisoup::volumetrics::VoxelizerFloat>(6.0);
            if (CubeValues[tets[tetIdx][0]] > 0.0) {
                // any but 0, 2
                currVoxel.corners[triOffset] = 1 << tets[tetIdx][1];
            } else {
                currVoxel.corners[triOffset] = 1 << tets[tetIdx][0];
            }
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
            tri2[1] = p1.Interpolate(p2, GetOffset(CubeValues[tets[tetIdx][1]], CubeValues[tets[tetIdx][2]],
                                             static_cast<trisoup::volumetrics::VoxelizerFloat>(0)));
            tri2[2] = tri[1];

            // tet1
            *vol2 = vislib::math::Abs((tri2[1] - p2).Dot((tri[0] - p2).Cross(tri[1] - p2))) /
                    static_cast<trisoup::volumetrics::VoxelizerFloat>(6.0);
            // tet3
            *vol2 += vislib::math::Abs((p0 - p2).Dot((tri[1] - p2).Cross(tri[0] - p2))) /
                     static_cast<trisoup::volumetrics::VoxelizerFloat>(6.0);
            //tri2[0] = p;
            //tri2[1] = p;
            //tri2[2] = p;
            //tri[1].p[0] = tri[0].p[0];
            //tri[1].p[1] = VertexInterp(iso,g.p[v1],g.p[v2],g.val[v1],g.val[v2]);
            //tri[1].p[2] = tri[0].p[1];

            if (CubeValues[tets[tetIdx][0]] > 0.0) {
                *vol = fullVol - (*vol + *vol2);
                *vol2 = 0.0;
                // any but 0, 2
                currVoxel.corners[triOffset] = 1 << tets[tetIdx][3];
            } else {
                currVoxel.corners[triOffset] = 1 << tets[tetIdx][2];
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
            tri[0] = p0.Interpolate(p1, GetOffset(CubeValues[tets[tetIdx][0]], CubeValues[tets[tetIdx][1]],
                                            static_cast<trisoup::volumetrics::VoxelizerFloat>(0)));
            tri[1] = p1.Interpolate(p3, GetOffset(CubeValues[tets[tetIdx][1]], CubeValues[tets[tetIdx][3]],
                                            static_cast<trisoup::volumetrics::VoxelizerFloat>(0)));
            tri[2] = p2.Interpolate(p3, GetOffset(CubeValues[tets[tetIdx][2]], CubeValues[tets[tetIdx][3]],
                                            static_cast<trisoup::volumetrics::VoxelizerFloat>(0)));

            if (CubeValues[tets[tetIdx][1]] > 0.0) {
                // any but 1, 2
                currVoxel.corners[triOffset] = 1 << tets[tetIdx][0];
            } else {
                currVoxel.corners[triOffset] = 1 << tets[tetIdx][1];
            }
            //tri[0] = p;
            //tri[1] = p;
            //tri[2] = p;
            //tri[0].p[0] = VertexInterp(iso,g.p[v0],g.p[v1],g.val[v0],g.val[v1]);
            //tri[0].p[1] = VertexInterp(iso,g.p[v1],g.p[v3],g.val[v1],g.val[v3]);
            //tri[0].p[2] = VertexInterp(iso,g.p[v2],g.p[v3],g.val[v2],g.val[v3]);
            triOffset++;
            tri2.SetPointer(currVoxel.triangles + 3 * 3 * triOffset);
            tri2[0] = tri[0];
            tri2[1] = p0.Interpolate(p2, GetOffset(CubeValues[tets[tetIdx][0]], CubeValues[tets[tetIdx][2]],
                                             static_cast<trisoup::volumetrics::VoxelizerFloat>(0)));
            tri2[2] = tri[2];

            // tet1
            *vol = vislib::math::Abs((tri[0] - p1).Dot((tri2[1] - p1).Cross(tri[1] - p1))) /
                   static_cast<trisoup::volumetrics::VoxelizerFloat>(6.0);
            vol2 = currVoxel.volumes + triOffset;
            // tet2
            *vol2 = vislib::math::Abs((tri[2] - p2).Dot((tri[1] - p2).Cross(tri2[1] - p2))) /
                    static_cast<trisoup::volumetrics::VoxelizerFloat>(6.0);
            // tet3
            *vol2 += vislib::math::Abs((p2 - p1).Dot((tri[1] - p1).Cross(tri2[1] - p1))) /
                     static_cast<trisoup::volumetrics::VoxelizerFloat>(6.0);
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
                // any but 1, 2
                currVoxel.corners[triOffset] = 1 << tets[tetIdx][3];
            } else {
                currVoxel.corners[triOffset] = 1 << tets[tetIdx][2];
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
            tri[0] = p3.Interpolate(p0, GetOffset(CubeValues[tets[tetIdx][3]], CubeValues[tets[tetIdx][0]],
                                            static_cast<trisoup::volumetrics::VoxelizerFloat>(0)));
            tri[1] = p3.Interpolate(p2, GetOffset(CubeValues[tets[tetIdx][3]], CubeValues[tets[tetIdx][2]],
                                            static_cast<trisoup::volumetrics::VoxelizerFloat>(0)));
            tri[2] = p3.Interpolate(p1, GetOffset(CubeValues[tets[tetIdx][3]], CubeValues[tets[tetIdx][1]],
                                            static_cast<trisoup::volumetrics::VoxelizerFloat>(0)));

            *vol = vislib::math::Abs((tri[0] - p3).Dot((tri[1] - p3).Cross(tri[2] - p3))) /
                   static_cast<trisoup::volumetrics::VoxelizerFloat>(6.0);
            if (CubeValues[tets[tetIdx][3]] > 0.0) {
                *vol = fullVol - *vol;
                // any but 3
                currVoxel.corners[triOffset] = 1 << tets[tetIdx][0];
                currVoxel.corners[triOffset] |= 1 << tets[tetIdx][1];
                currVoxel.corners[triOffset] |= 1 << tets[tetIdx][2];
            } else {
                currVoxel.corners[triOffset] = tets[tetIdx][3];
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
            if (CubeValues[tets[tetIdx][0]] <= 0 && CubeValues[tets[tetIdx][1]] <= 0 &&
                CubeValues[tets[tetIdx][2]] <= 0 && CubeValues[tets[tetIdx][3]] <= 0) {
                *vol = (sjd->CellSize * sjd->CellSize * sjd->CellSize) /
                       static_cast<trisoup::volumetrics::VoxelizerFloat>(6.0);
            }
        }
    }
}


DWORD TetraVoxelizer::Run(void* userData) {
    using megamol::core::utility::log::Log;

    unsigned int vertFloatSize = 0;
    trisoup::volumetrics::VoxelizerFloat currRad = 0.f;
    //, maxRad = -FLT_MAX;
    trisoup::volumetrics::VoxelizerFloat currDist;
    vislib::math::Point<unsigned int, 3> pStart, pEnd;
    vislib::math::Point<trisoup::volumetrics::VoxelizerFloat, 3> p;
    sjd = static_cast<trisoup::volumetrics::SubJobData*>(userData);
    SIZE_T numNeg = 0, numZero = 0, numPos = 0, numPartSkipped = 0, numPartAdded = 0;

    // my hacky breakpoint to catch a specific thread ^^ ;-)
    //#ifdef _DEBUG
    //        if ( sjd->resX == 5 && sjd->resY == 5 && sjd->resZ == 5) {
    //            sjd->resX = sjd->resY;
    //        }
    //#endif

    unsigned int fifoEnd = 0, fifoCur = 0;
    trisoup::volumetrics::FatVoxel* volume = new trisoup::volumetrics::FatVoxel[sjd->resX * sjd->resY * sjd->resZ];
    // we can do that when using structs ... - its safer doing memzero (in case new members get added to the trisoup::volumetrics::FatVoxel struct)
    memset(volume, 0, sizeof(trisoup::volumetrics::FatVoxel) * (sjd->resX * sjd->resY * sjd->resZ));
    for (SIZE_T i = 0; i < static_cast<SIZE_T>(sjd->resX * sjd->resY * sjd->resZ); i++) {
        volume[i].distField = FLT_MAX;
        volume[i].borderVoxel = NULL;
        volume[i].mcCase = 0;
        //volume[i].enclosingCandidate = 0; // thomasbm
    }

    unsigned int partListCnt = sjd->datacall->GetParticleListCount();
    for (unsigned int partListI = 0; partListI < partListCnt; partListI++) {
        geocalls::MultiParticleDataCall::Particles ps = sjd->datacall->AccessParticles(partListI);
        UINT64 numParticles = ps.GetCount();
        unsigned int stride = ps.GetVertexDataStride();
        geocalls::MultiParticleDataCall::Particles::VertexDataType dataType = ps.GetVertexDataType();
        unsigned char* vertexData = (unsigned char*)ps.GetVertexData();
        switch (dataType) {
        case geocalls::MultiParticleDataCall::Particles::VERTDATA_NONE:
            continue;
        case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
            vertFloatSize = 3 * sizeof(float);
            break;
        case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
            vertFloatSize = 4 * sizeof(float);
            break;
        case geocalls::MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
            Log::DefaultLog.WriteError("This module does not yet like quantized data");
            return -2;
        }
    }

    // sample everything into our temporary volume
    vislib::math::Cuboid<float> bx(sjd->Bounds);
    vislib::math::Point<trisoup::volumetrics::VoxelizerFloat, 3> Centroid = bx.CalcCenter();
    trisoup::volumetrics::VoxelizerFloat distOffset =
        vislib::math::Sqrt(bx.Width() * bx.Width() + bx.Height() * bx.Height() + bx.Depth() * bx.Depth()) /
        static_cast<trisoup::volumetrics::VoxelizerFloat>(2.0);
    // TODO: what did this do anyway
    //trisoup::volumetrics::VoxelizerFloat g = sjd->MaxRad * sjd->RadMult - sjd->MaxRad;
    //if (g > static_cast<trisoup::volumetrics::VoxelizerFloat>(0)) {
    //    bx.Grow(2 * g);
    //}

    for (unsigned int partListI = 0; partListI < partListCnt; partListI++) {
        geocalls::MultiParticleDataCall::Particles ps = sjd->datacall->AccessParticles(partListI);
        currRad = ps.GetGlobalRadius() * sjd->RadMult;
        UINT64 numParticles = ps.GetCount();
        unsigned int stride = ps.GetVertexDataStride();
        geocalls::MultiParticleDataCall::Particles::VertexDataType dataType = ps.GetVertexDataType();
        unsigned char* vertexData = (unsigned char*)ps.GetVertexData();
        switch (dataType) {
        case geocalls::MultiParticleDataCall::Particles::VERTDATA_NONE:
            continue;
        case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
            vertFloatSize = 3 * sizeof(float);
            break;
        case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
            vertFloatSize = 4 * sizeof(float);
            break;
        case geocalls::MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
            Log::DefaultLog.WriteError("This module does not yet like quantized data");
            return -2;
        }
        for (UINT64 l = 0; l < numParticles; l++) {
            vislib::math::ShallowPoint<float, 3> sp((float*)&vertexData[(vertFloatSize + stride) * l]);
            if (dataType == geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR) {
                currRad = (float)vertexData[(vertFloatSize + stride) * l + 3 * sizeof(float)];
                currRad *= sjd->RadMult;
            }
            if (Centroid.Distance(sp) > currRad + distOffset) {
                numPartSkipped++;
                continue;
            }
            numPartAdded++;

            int x, y, z;
            x = static_cast<int>((sp.X() - currRad - sjd->Bounds.Left()) / sjd->CellSize) - 1;
            if (x < 0)
                x = 0;
            y = static_cast<int>((sp.Y() - currRad - sjd->Bounds.Bottom()) / sjd->CellSize) - 1;
            if (y < 0)
                y = 0;
            z = static_cast<int>((sp.Z() - currRad - sjd->Bounds.Back()) / sjd->CellSize) - 1;
            if (z < 0)
                z = 0;
            pStart.Set(x, y, z);

            x = static_cast<int>((sp.X() + currRad - sjd->Bounds.Left()) / sjd->CellSize) + 2;
            if (x >= static_cast<int>(sjd->resX))
                x = sjd->resX - 1;
            y = static_cast<int>((sp.Y() + currRad - sjd->Bounds.Bottom()) / sjd->CellSize) + 2;
            if (y >= static_cast<int>(sjd->resY))
                y = sjd->resY - 1;
            z = static_cast<int>((sp.Z() + currRad - sjd->Bounds.Back()) / sjd->CellSize) + 2;
            if (z >= static_cast<int>(sjd->resZ))
                z = sjd->resZ - 1;
            pEnd.Set(x, y, z);

            for (int z = pStart.Z(); z <= static_cast<int>(pEnd.Z()); z++) {
                for (int y = pStart.Y(); y <= static_cast<int>(pEnd.Y()); y++) {
                    for (int x = pStart.X(); x <= static_cast<int>(pEnd.X()); x++) {
                        // TODO think about this. here the voxel content is determined by a corner
                        p.Set(sjd->Bounds.Left() + x * sjd->CellSize, sjd->Bounds.Bottom() + y * sjd->CellSize,
                            sjd->Bounds.Back() + z * sjd->CellSize);

                        // and here it is the center!
                        //p.Set(
                        //    sjd->Bounds.Left() + x * sjd->CellSize + sjd->CellSize * 0.5f,
                        //    sjd->Bounds.Bottom() + y * sjd->CellSize + sjd->CellSize * 0.5f,
                        //    sjd->Bounds.Back() + z * sjd->CellSize + sjd->CellSize * 0.5f);
                        currDist = sp.Distance<trisoup::volumetrics::VoxelizerFloat>(p) - currRad;
                        SIZE_T i = sjd->cellIndex(x, y, z);
                        //if (x > 5 && x < 8 && z > 4 && z < 8 && y > 6 && y < 9) {
                        volume[i].distField = vislib::math::Min(volume[i].distField, currDist);
                        //} else {
                        //    volume[i].distField = 10000000;
                        //}
                        // thomasmbm: i think the 'numNeg'/'numPos' using to determinate full/empty volumes is i not correct, since particle-spheres may overlap!
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

    //trisoup::volumetrics::FatVoxel *tv = new trisoup::volumetrics::FatVoxel[2 * 2 * 2];
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
    //trisoup::volumetrics::FatVoxel &currVoxel = tv[(0 * sjd->resY + 0) * sjd->resX + 0];
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
        trisoup::volumetrics::Surface s;
        s.surface = 0.0;
        s.volume = (sjd->resX - 1) * (sjd->resY - 1) * (sjd->resZ - 1) * sjd->CellSize * sjd->CellSize * sjd->CellSize;
        s.voidVolume = 0;
        sjd->Result.surfaces.Append(s);
    } else if (numPartAdded == 0 || numPos == (sjd->resX) * (sjd->resY) * (sjd->resZ)) {
        // totally empty
        trisoup::volumetrics::Surface s;
        s.surface = 0.0;
        s.volume = 0.0; // TODO: negative volume maybe?
        s.voidVolume =
            (sjd->resX - 1) * (sjd->resY - 1) * (sjd->resZ - 1) * sjd->CellSize * sjd->CellSize * sjd->CellSize;
        sjd->Result.surfaces.Append(s);
    } else {
        // march it
        for (int x = 0; x < static_cast<int>(sjd->resX) - 1; x++) {
            for (int y = 0; y < static_cast<int>(sjd->resY) - 1; y++) {
                for (int z = 0; z < static_cast<int>(sjd->resZ) - 1; z++) {
                    MarchCell(volume, x, y, z);
                }
            }
        }

        // collect the surfaces
        for (int x = 0; x < static_cast<int>(sjd->resX) - 1; x++) {
            for (int y = 0; y < static_cast<int>(sjd->resY) - 1; y++) {
                for (int z = 0; z < static_cast<int>(sjd->resZ) - 1; z++) {
                    CollectCell(volume, x, y, z);
                }
            }
        }

        // thomasbm: collect enclosed surfaces ...
        //DetectEncapsulatedSurfs();
    }

    if (sjd->storeVolume) {
        trisoup::trisoupVolumetricDataCall::Volume& v = sjd->Result.debugVolume;
        v.volumeData = new trisoup::trisoupVolumetricDataCall::VoxelType[(sjd->resX) * (sjd->resY) * (sjd->resZ)];
        memset(v.volumeData, 0,
            sizeof(trisoup::trisoupVolumetricDataCall::VoxelType) * (sjd->resX) * (sjd->resY) * (sjd->resZ));

        v.resX = sjd->resX;
        v.resY = sjd->resY;
        v.resZ = sjd->resZ;
        v.origin[0] = sjd->Bounds.Left();
        v.origin[1] = sjd->Bounds.Bottom();
        v.origin[2] = sjd->Bounds.Back();
        v.scaling[0] = sjd->CellSize, v.scaling[1] = sjd->CellSize, v.scaling[2] = sjd->CellSize;
        /*            // reference corner of this cell
            vislib::math::Point<trisoup::volumetrics::VoxelizerFloat, 3> p(sjd->Bounds.Left() + x * sjd->CellSize,
                sjd->Bounds.Bottom() + y * sjd->CellSize,
                sjd->Bounds.Back() + z * sjd->CellSize);
                */
        for (int x = 0; x < static_cast<int>(sjd->resX) - 1; x++) {
            for (int y = 0; y < static_cast<int>(sjd->resY) - 1; y++) {
                for (int z = 0; z < static_cast<int>(sjd->resZ) - 1; z++) {
                    unsigned int index = sjd->cellIndex(x, y, z);
                    trisoup::volumetrics::FatVoxel& fv = volume[index];
                    if (fv.mcCase == 255 || fv.mcCase == 0) {
                        // if (fv.consumedTriangles != 0) asm { int 3 };
                        v.volumeData[index] =
                            static_cast<megamol::trisoup::trisoupVolumetricDataCall::VoxelType>(fv.consumedTriangles);
                    } // else __asm int 3;
                }
            }
        }
    }

    // dealloc stuff in volume
    // dealloc volume as a whole etc.
    for (int x = 0; x < static_cast<int>(sjd->resX) - 1; x++) {
        for (int y = 0; y < static_cast<int>(sjd->resY) - 1; y++) {
            for (int z = 0; z < static_cast<int>(sjd->resZ) - 1; z++) {
                if (trisoup::volumetrics::MarchingCubeTables::a2ucTriangleConnectionCount
                        [volume[sjd->cellIndex(x, y, z)].mcCase] > 0) {
                    ARY_SAFE_DELETE(volume[sjd->cellIndex(x, y, z)].triangles);
                    ARY_SAFE_DELETE(volume[sjd->cellIndex(x, y, z)].volumes);
                    ARY_SAFE_DELETE(volume[sjd->cellIndex(x, y, z)].corners);
                    // do NOT delete!
                    volume[sjd->cellIndex(x, y, z)].borderVoxel = NULL;
                }
            }
        }
    }
    ARY_SAFE_DELETE(volume);

#ifdef ULTRADEBUG
    megamol::core::utility::log::Log::DefaultLog.WriteInfo("job done: (%u,%u,%u)", sjd->gridX, sjd->gridY, sjd->gridZ);
#endif /* ULTRADEBUG */

    // TODO: it would really help if this dude already checked for finished neighbors and took the globalID from there
    // if possible
    unsigned int MinGID;
    sjd->parent->AccessMaxGlobalID.Lock();
    MinGID = sjd->parent->MaxGlobalID;
    sjd->parent->MaxGlobalID += static_cast<unsigned int>(sjd->Result.surfaces.Count());
    sjd->parent->AccessMaxGlobalID.Unlock();

    for (unsigned int surfIdx = 0; surfIdx < sjd->Result.surfaces.Count(); surfIdx++)
        sjd->Result.surfaces[surfIdx].globalID = MinGID + surfIdx;

    // first find self to set thisIndex
    unsigned int thisIndex;
    for (thisIndex = 0; thisIndex < sjd->parent->SubJobDataList.Count(); thisIndex++) {
        if (sjd->parent->SubJobDataList[thisIndex] == sjd)
            break;
    }

    vislib::Array<unsigned int> joinableSurfs;
    joinableSurfs.SetCapacityIncrement(10);
    for (unsigned int sjdIdx = 0; sjdIdx < sjd->parent->SubJobDataList.Count(); sjdIdx++) {
        trisoup::volumetrics::SubJobData* parentSubJob = sjd->parent->SubJobDataList[sjdIdx];

        if (!parentSubJob->Result.done || parentSubJob->Result.surfaces.Count() == 0 || sjdIdx == thisIndex)
            continue;

        vislib::math::Cuboid<trisoup::volumetrics::VoxelizerFloat> box = sjd->Bounds;
        box.Union(parentSubJob->Bounds);

        if (box.Volume() > sjd->Bounds.Volume() + parentSubJob->Bounds.Volume())
            continue;

        for (unsigned int surfIdx = 0; surfIdx < sjd->Result.surfaces.Count(); surfIdx++) {
            joinableSurfs.Clear();
            sjd->parent->RewriteGlobalID.Lock();

            for (unsigned int otherSurfIdx = 0; otherSurfIdx < parentSubJob->Result.surfaces.Count(); otherSurfIdx++) {
                if (sjd->parent->areSurfacesJoinable(thisIndex, surfIdx, sjdIdx, otherSurfIdx)) {
                    joinableSurfs.Add(otherSurfIdx);
                } else {
#ifdef PARALLEL_BBOX_COLLECT // cs: RewriteGlobalID
                    // thomasbm: a surface may be entirely located within a subjob
                    Surface& surf = sjd->Result.surfaces[surfIdx];
                    if (sjd->parent->globalIdBoxes.Count() <= surf.globalID)
                        sjd->parent->globalIdBoxes.SetCount(surf.globalID + 1);
                    sjd->parent->globalIdBoxes[surf.globalID].Union(surf.boundingBox);
#endif
                }
            }

            if (joinableSurfs.Count() > 0) {
                // Lock() and Unlock() should be called symmetrically
                //sjd->parent->RewriteGlobalID.Lock();
                unsigned int smallest = INT_MAX;
                for (unsigned int jsurfIdx = 0; jsurfIdx < joinableSurfs.Count(); jsurfIdx++) {
                    unsigned int gid = parentSubJob->Result.surfaces[joinableSurfs[jsurfIdx]].globalID;
                    if (gid < smallest)
                        smallest = gid;
                }

#ifdef PARALLEL_BBOX_COLLECT
                //thomasbm:
                if (sjd->parent->globalIdBoxes.Count() <= smallest)
                    sjd->parent->globalIdBoxes.SetCount(smallest + 1);
#endif
                for (unsigned int jsurfIdx = 0; jsurfIdx < joinableSurfs.Count(); jsurfIdx++) {
                    trisoup::volumetrics::Surface& surf = parentSubJob->Result.surfaces[joinableSurfs[jsurfIdx]];
#ifdef PARALLEL_BBOX_COLLECT
                    // thomasbm: gather global surface-bounding boxes
                    //if (surf.globalID ??)
                    //    sjd->parent->globalIdBoxes[smallest].Union(sjd->parent->globalIdBoxes[surf.globalID]);
                    sjd->parent->globalIdBoxes[smallest].Union(surf.boundingBox);
#endif
                    surf.globalID = smallest;
                }

                sjd->Result.surfaces[surfIdx].globalID = smallest;
                //sjd->parent->RewriteGlobalID.Unlock();
            }
            //sjd->Result.surfaces[surfIdx].globalID = parentSubJob->Result.surfaces[otherSurfIdx].globalID;
            sjd->parent->RewriteGlobalID.Unlock();
        }
    }

    // END TODO

    sjd->Result.done = true;

    return 0;
}

bool TetraVoxelizer::Terminate(void) {
    terminate = true;
    return true;
}
