/*
* PointcloudHelpers.h
*
* Copyright (C) 2017 by MegaMol team
* Alle Rechte vorbehalten.
*/

#ifndef MMSTD_DATATOOLS_POINTCLOUDHELPERS_H_INCLUDED
#define MMSTD_DATATOOLS_POINTCLOUDHELPERS_H_INCLUDED
#pragma once

#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/moldyn/DirectionalParticleDataCall.h"
#include <vector>

namespace megamol {
namespace stdplugin {
namespace datatools {

/**
* Class that implements the interface nanoflann needs for simple spherical particles.
* The index vector addresses all of the particles across all lists, so its
* range is (0,Sum(Allof(particleLists).Count)).
*/
class simplePointcloud {
private:

    megamol::core::moldyn::MultiParticleDataCall *dat;
    std::vector<size_t> &indices;
    bool cycleX, cycleY, cycleZ;

public:

    typedef float coord_t;

    simplePointcloud(megamol::core::moldyn::MultiParticleDataCall *dat, std::vector<size_t> &indices)
        : dat(dat), indices(indices) {
        // intentionally empty
    }
    ~simplePointcloud() {
        // intentionally empty
    }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const {
        return indices.size();
    }

    // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
    inline coord_t kdtree_distance(const coord_t *p1, const size_t idx_p2, size_t /*size*/) const {
        float const *p2 = get_position(idx_p2);

        const coord_t d0 = p1[0] - p2[0];
        const coord_t d1 = p1[1] - p2[1];
        const coord_t d2 = p1[2] - p2[2];

        return d0 * d0 + d1 * d1 + d2 * d2;
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline coord_t kdtree_get_pt(const size_t idx, int dim) const {
        assert((dim >= 0) && (dim < 3));
        return get_position(idx)[dim];
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& bb) const {
        //return false;

        assert(bb.size() == 3);
        const auto& cbox = dat->AccessBoundingBoxes().ObjectSpaceBBox();
        bb[0].low = cbox.Left();
        bb[0].high = cbox.Right();
        bb[1].low = cbox.Bottom();
        bb[1].high = cbox.Top();
        bb[2].low = cbox.Back();
        bb[2].high = cbox.Front();
        return true;
    }

    inline const coord_t* get_position(size_t index) const {
        using megamol::core::moldyn::SimpleSphericalParticles;

        unsigned int plc = dat->GetParticleListCount();
        for (unsigned int pli = 0; pli < plc; pli++) {
            auto& pl = dat->AccessParticles(pli);
            if ((pl.GetVertexDataType() != SimpleSphericalParticles::VERTDATA_FLOAT_XYZ)
                && (pl.GetVertexDataType() != SimpleSphericalParticles::VERTDATA_FLOAT_XYZR)) {
                continue;
            }

            if (index < pl.GetCount()) {
                unsigned int vert_stride = 0;
                if (pl.GetVertexDataType() == SimpleSphericalParticles::VERTDATA_FLOAT_XYZ) vert_stride = 12;
                else if (pl.GetVertexDataType() == SimpleSphericalParticles::VERTDATA_FLOAT_XYZR) vert_stride = 16;
                else continue;
                vert_stride = std::max<unsigned int>(vert_stride, pl.GetVertexDataStride());
                const unsigned char *vert = static_cast<const unsigned char*>(pl.GetVertexData());

                return reinterpret_cast<const float *>(vert + (index * vert_stride));
            }

            index -= static_cast<size_t>(pl.GetCount());
        }

        return nullptr;
    }

    inline const void get(size_t global_index, megamol::core::moldyn::SimpleSphericalParticles& par_list, size_t& local_index) const {
        using megamol::core::moldyn::SimpleSphericalParticles;

        unsigned int plc = dat->GetParticleListCount();
        for (unsigned int pli = 0; pli < plc; pli++) {
            auto& pl = dat->AccessParticles(pli);
            if ((pl.GetVertexDataType() != SimpleSphericalParticles::VERTDATA_FLOAT_XYZ)
                && (pl.GetVertexDataType() != SimpleSphericalParticles::VERTDATA_FLOAT_XYZR)) {
                continue;
            }

            if (global_index < pl.GetCount()) {
                local_index = global_index;
                par_list = pl;
                return;
            }

            global_index -= static_cast<size_t>(pl.GetCount());
        }
    }

};

/**
* Class that implements the interface nanoflann needs for directional particles.
* The index vector addresses all of the particles across all lists, so its
* range is (0,Sum(Allof(particleLists).Count)).
*/
class directionalPointcloud {
private:

    megamol::core::moldyn::DirectionalParticleDataCall *dat;
    std::vector<size_t> &indices;
    bool cycleX, cycleY, cycleZ;

public:

    typedef float coord_t;

    directionalPointcloud(megamol::core::moldyn::DirectionalParticleDataCall *dat, std::vector<size_t> &indices)
        : dat(dat), indices(indices) {
        // intentionally empty
    }
    ~directionalPointcloud() {
        // intentionally empty
    }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const {
        return indices.size();
    }

    // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
    inline coord_t kdtree_distance(const coord_t *p1, const size_t idx_p2, size_t /*size*/) const {
        float const *p2 = get_position(idx_p2);

        const coord_t d0 = p1[0] - p2[0];
        const coord_t d1 = p1[1] - p2[1];
        const coord_t d2 = p1[2] - p2[2];

        return d0 * d0 + d1 * d1 + d2 * d2;
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline coord_t kdtree_get_pt(const size_t idx, int dim) const {
        assert((dim >= 0) && (dim < 3));
        return get_position(idx)[dim];
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& bb) const {
        //return false;

        assert(bb.size() == 3);
        const auto& cbox = dat->AccessBoundingBoxes().ObjectSpaceBBox();
        bb[0].low = cbox.Left();
        bb[0].high = cbox.Right();
        bb[1].low = cbox.Bottom();
        bb[1].high = cbox.Top();
        bb[2].low = cbox.Back();
        bb[2].high = cbox.Front();
        return true;
    }

    inline const coord_t* get_position(size_t index) const {
        using megamol::core::moldyn::DirectionalParticles;

        unsigned int plc = dat->GetParticleListCount();
        for (unsigned int pli = 0; pli < plc; pli++) {
            auto& pl = dat->AccessParticles(pli);
            if ((pl.GetVertexDataType() != DirectionalParticles::VERTDATA_FLOAT_XYZ)
                && (pl.GetVertexDataType() != DirectionalParticles::VERTDATA_FLOAT_XYZR)) {
                continue;
            }

            if (index < pl.GetCount()) {
                unsigned int vert_stride = 0;
                if (pl.GetVertexDataType() == DirectionalParticles::VERTDATA_FLOAT_XYZ) vert_stride = 12;
                else if (pl.GetVertexDataType() == DirectionalParticles::VERTDATA_FLOAT_XYZR) vert_stride = 16;
                else continue;
                vert_stride = std::max<unsigned int>(vert_stride, pl.GetVertexDataStride());
                const unsigned char *vert = static_cast<const unsigned char*>(pl.GetVertexData());

                return reinterpret_cast<const float *>(vert + (index * vert_stride));
            }

            index -= static_cast<size_t>(pl.GetCount());
        }

        return nullptr;
    }

    inline const coord_t* get_velocity(size_t index) const {
        using megamol::core::moldyn::DirectionalParticles;

        unsigned int plc = dat->GetParticleListCount();
        for (unsigned int pli = 0; pli < plc; pli++) {
            auto& pl = dat->AccessParticles(pli);
            if ((pl.GetVertexDataType() != DirectionalParticles::VERTDATA_FLOAT_XYZ)
                && (pl.GetVertexDataType() != DirectionalParticles::VERTDATA_FLOAT_XYZR)) {
                continue;
            }

            if (index < pl.GetCount()) {
                unsigned int dir_stride = 0;
                if (pl.GetDirDataType() == DirectionalParticles::DIRDATA_FLOAT_XYZ) dir_stride = 12;
                else continue;
                dir_stride = std::max<unsigned int>(dir_stride, pl.GetDirDataStride());
                const unsigned char *dir = static_cast<const unsigned char*>(pl.GetDirData());

                return reinterpret_cast<const float *>(dir + (index * dir_stride));
            }

            index -= static_cast<size_t>(pl.GetCount());
        }

        return nullptr;
    }

    inline const void get(size_t global_index, megamol::core::moldyn::DirectionalParticles& par_list, size_t& local_index) const {
        using megamol::core::moldyn::DirectionalParticles;

        unsigned int plc = dat->GetParticleListCount();
        for (unsigned int pli = 0; pli < plc; pli++) {
            auto& pl = dat->AccessParticles(pli);
            if ((pl.GetVertexDataType() != DirectionalParticles::VERTDATA_FLOAT_XYZ)
                && (pl.GetVertexDataType() != DirectionalParticles::VERTDATA_FLOAT_XYZR)) {
                continue;
            }

            if (global_index < pl.GetCount()) {
                local_index = global_index;
                par_list = pl;
                return;
            }

            global_index -= static_cast<size_t>(pl.GetCount());
        }
    }

};

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MMSTD_DATATOOLS_POINTCLOUDHELPERS_H_INCLUDED */
