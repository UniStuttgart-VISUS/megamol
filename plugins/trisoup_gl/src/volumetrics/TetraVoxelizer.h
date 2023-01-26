#ifndef MEGAMOLCORE_TETRAVOXELIZER_H_INCLUDED
#define MEGAMOLCORE_TETRAVOXELIZER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "trisoup/volumetrics/JobStructures.h"
#include "trisoup/volumetrics/TagVolume.h"
#include "vislib/math/ShallowShallowTriangle.h"
#include "vislib/sys/ThreadPool.h"

namespace megamol {
namespace trisoup_gl {
namespace volumetrics {

class TetraVoxelizer : public vislib::sys::Runnable {
public:
    TetraVoxelizer(void);
    ~TetraVoxelizer(void) override;

    trisoup::volumetrics::VoxelizerFloat GetOffset(trisoup::volumetrics::VoxelizerFloat fValue1,
        trisoup::volumetrics::VoxelizerFloat fValue2, trisoup::volumetrics::VoxelizerFloat fValueDesired);

    void growSurfaceFromTriangle(trisoup::volumetrics::FatVoxel* theVolume, unsigned int x, unsigned int y,
        unsigned int z, unsigned seedTriIndex, trisoup::volumetrics::Surface& surf);

    /**
     * also does compute fullFaces
     */
    trisoup::volumetrics::VoxelizerFloat growVolume(trisoup::volumetrics::FatVoxel* theVolume,
        trisoup::volumetrics::Surface& surf, const vislib::math::Point<int, 3>& seed, bool emptyVolume);

    bool CellHasNoGeometry(trisoup::volumetrics::FatVoxel* theVolume, unsigned x, unsigned y, unsigned z);

    bool CellFull(trisoup::volumetrics::FatVoxel* theVolume, unsigned x, unsigned y, unsigned z);

    void MarchCell(trisoup::volumetrics::FatVoxel* theVolume, unsigned int x, unsigned int y, unsigned int z);

    void CollectCell(trisoup::volumetrics::FatVoxel* theVolume, unsigned int x, unsigned int y, unsigned int z);

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
    DWORD Run(void* userData) override;

    /**
     * Asks the thread to terminate after the calculation of the current
     * cluster is finished
     *
     * @return Always 'true'
     */
    bool Terminate(void) override;

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

    void ProcessTriangle(vislib::math::ShallowShallowTriangle<trisoup::volumetrics::VoxelizerFloat, 3>& sstI,
        trisoup::volumetrics::FatVoxel& f, unsigned triIdx, trisoup::volumetrics::Surface& surf, unsigned int x,
        unsigned int y, unsigned int z);


    bool terminate;

    trisoup::volumetrics::SubJobData* sjd;

    vislib::SingleLinkedList<vislib::math::Point<unsigned int, 4>> cellFIFO;
};

} /* end namespace volumetrics */
} // namespace trisoup_gl
} /* end namespace megamol */

#endif /* MEGAMOLCORE_TETRAVOXELIZER_H_INCLUDED */
