#ifndef MEGAMOLCORE_TETRAVOXELIZER_H_INCLUDED
#define MEGAMOLCORE_TETRAVOXELIZER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "JobStructures.h"
#include "TagVolume.h"
#include "mmcore/utility/sys/ThreadPool.h"
#include "vislib/math/ShallowShallowTriangle.h"

namespace megamol {
namespace trisoup {
namespace volumetrics {

class TetraVoxelizer : public vislib::sys::Runnable {
public:
    TetraVoxelizer(void);
    ~TetraVoxelizer(void);

    VoxelizerFloat GetOffset(VoxelizerFloat fValue1, VoxelizerFloat fValue2, VoxelizerFloat fValueDesired);

    void growSurfaceFromTriangle(
        FatVoxel* theVolume, unsigned int x, unsigned int y, unsigned int z, unsigned seedTriIndex, Surface& surf);

    /**
     * also does compute fullFaces
     */
    VoxelizerFloat growVolume(
        FatVoxel* theVolume, Surface& surf, const vislib::math::Point<int, 3>& seed, bool emptyVolume);

    bool CellHasNoGeometry(FatVoxel* theVolume, unsigned x, unsigned y, unsigned z);

    bool CellFull(FatVoxel* theVolume, unsigned x, unsigned y, unsigned z);

    void MarchCell(FatVoxel* theVolume, unsigned int x, unsigned int y, unsigned int z);

    void CollectCell(FatVoxel* theVolume, unsigned int x, unsigned int y, unsigned int z);

    void DetectEncapsulatedSurfs();

    static int tets[6][4];
    static vislib::math::Point<int, 3> cornerNeighbors[8][7];
    static vislib::math::Point<int, 3> moreNeighbors[6];

    /**
     * Thread entry point.
     *
     * @param userData Pointer to a 'JobDescription' object holding the range
     *                 of clusters to work on.
     *
     * @return 0 on success, positive on abort, negative on error.
     */
    virtual DWORD Run(void* userData);

    /**
     * Asks the thread to terminate after the calculation of the current
     * cluster is finished
     *
     * @return Always 'true'
     */
    virtual bool Terminate(void);

    /**
     * Test for equality. Actually this checks for equality of the
     * associated SubJobData.
     *
     * @param rhs the right hand set operand
     *
     * @return whether this->sjd and rhs.sjd are equal
     */
    inline bool operator==(const TetraVoxelizer& rhs) const {
        return this->sjd == rhs.sjd;
    }

private:
    void debugPrintTriangle(vislib::math::ShallowShallowTriangle<float, 3>& tri);
    void debugPrintTriangle(vislib::math::ShallowShallowTriangle<double, 3>& tri);

    void ProcessTriangle(vislib::math::ShallowShallowTriangle<VoxelizerFloat, 3>& sstI, FatVoxel& f, unsigned triIdx,
        Surface& surf, unsigned int x, unsigned int y, unsigned int z);


    bool terminate;

    SubJobData* sjd;

    vislib::SingleLinkedList<vislib::math::Point<unsigned int, 4>> cellFIFO;
};

} /* end namespace volumetrics */
} /* end namespace trisoup */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_TETRAVOXELIZER_H_INCLUDED */
