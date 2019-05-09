/*
 * PointcloudHelpers.h
 *
 * Copyright (C) 2017 by MegaMol team
 * Alle Rechte vorbehalten.
 */

#ifndef MMSTD_DATATOOLS_POINTCLOUDHELPERS_H_INCLUDED
#define MMSTD_DATATOOLS_POINTCLOUDHELPERS_H_INCLUDED
#pragma once

#include <array>
#include <algorithm>
#include <vector>
#include "mmcore/moldyn/DirectionalParticleDataCall.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"

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
    megamol::core::moldyn::MultiParticleDataCall* dat;
    std::vector<size_t>& indices;
    bool cycleX, cycleY, cycleZ;

public:
    typedef float coord_t;

    simplePointcloud(megamol::core::moldyn::MultiParticleDataCall* dat, std::vector<size_t>& indices)
        : dat(dat), indices(indices) {
        // intentionally empty
    }
    ~simplePointcloud() {
        // intentionally empty
    }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return indices.size(); }

    // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the
    // class:
    inline coord_t kdtree_distance(const coord_t* p1, const size_t idx_p2, size_t /*size*/) const {
        float const* p2 = get_position(idx_p2);

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
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it
    //   again. Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX> bool kdtree_get_bbox(BBOX& bb) const {
        // return false;

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
            if ((pl.GetVertexDataType() != SimpleSphericalParticles::VERTDATA_FLOAT_XYZ) &&
                (pl.GetVertexDataType() != SimpleSphericalParticles::VERTDATA_FLOAT_XYZR)) {
                continue;
            }

            if (index < pl.GetCount()) {
                unsigned int vert_stride = 0;
                if (pl.GetVertexDataType() == SimpleSphericalParticles::VERTDATA_FLOAT_XYZ)
                    vert_stride = 12;
                else if (pl.GetVertexDataType() == SimpleSphericalParticles::VERTDATA_FLOAT_XYZR)
                    vert_stride = 16;
                else
                    continue;
                vert_stride = std::max<unsigned int>(vert_stride, pl.GetVertexDataStride());
                const unsigned char* vert = static_cast<const unsigned char*>(pl.GetVertexData());

                return reinterpret_cast<const float*>(vert + (index * vert_stride));
            }

            index -= static_cast<size_t>(pl.GetCount());
        }

        return nullptr;
    }
};


/**
 * Class that implements the interface nanoflann needs for simple spherical particles with icol.
 * The index vector addresses all of the particles across all lists, so its
 * range is (0,Sum(Allof(particleLists).Count)).
 */
class simpleIColPointcloud {
private:
    megamol::core::moldyn::SimpleSphericalParticles* dat;
    bool cycleX, cycleY, cycleZ;

public:
    typedef float coord_t;

    simpleIColPointcloud(megamol::core::moldyn::SimpleSphericalParticles* dat)
        : dat(dat) {
        // intentionally empty
    }
    ~simpleIColPointcloud() {
        // intentionally empty
    }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return dat->GetCount(); }

    // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the
    // class:
    inline coord_t kdtree_distance(const coord_t* p1, const size_t idx_p2, size_t /*size*/) const {
        auto const p2 = get_position(idx_p2);

        const coord_t d0 = p1[0] - p2[0];
        const coord_t d1 = p1[1] - p2[1];
        const coord_t d2 = p1[2] - p2[2];
        const coord_t d3 = p1[3] - p2[3];

        return d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline coord_t kdtree_get_pt(const size_t idx, int dim) const {
        assert((dim >= 0) && (dim < 4));
        return get_position(idx)[dim];
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it
    //   again. Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX> bool kdtree_get_bbox(BBOX& bb) const {
        // return false;

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

    inline std::array<coord_t, 4> get_position(size_t index) const {
        std::array<coord_t, 4> const ret{dat->GetParticleStore().GetXAcc()->Get_f(index),
            dat->GetParticleStore().GetYAcc()->Get_f(index), dat->GetParticleStore().GetZAcc()->Get_f(index),
            dat->GetParticleStore().GetCRAcc()->Get_f(index)};

        return ret;
    }
};


/**
 * Class that implements the interface nanoflann needs for directional particles.
 * The index vector addresses all of the particles across all lists, so its
 * range is (0,Sum(Allof(particleLists).Count)).
 */
class directionalPointcloud {
private:
    megamol::core::moldyn::DirectionalParticleDataCall* dat;
    std::vector<size_t>& indices;
    bool cycleX, cycleY, cycleZ;

public:
    typedef float coord_t;

    directionalPointcloud(megamol::core::moldyn::DirectionalParticleDataCall* dat, std::vector<size_t>& indices)
        : dat(dat), indices(indices) {
        // intentionally empty
    }
    ~directionalPointcloud() {
        // intentionally empty
    }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return indices.size(); }

    // TODO: this never seems to be used! why is that???
    // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the
    // class:
    inline coord_t kdtree_distance(const coord_t* p1, const size_t idx_p2, size_t /*size*/) const {
        float const* p2 = get_position(idx_p2);

        const coord_t d0 = p1[0] - p2[0];
        const coord_t d1 = p1[1] - p2[1];
        const coord_t d2 = p1[2] - p2[2];

        return sqrt(d0 * d0 + d1 * d1 + d2 * d2);
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline coord_t kdtree_get_pt(const size_t idx, int dim) const {
        assert((dim >= 0) && (dim < 3));
        return get_position(idx)[dim];
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it
    //   again. Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX> bool kdtree_get_bbox(BBOX& bb) const {
        // return false;

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
            if ((pl.GetVertexDataType() != DirectionalParticles::VERTDATA_FLOAT_XYZ) &&
                (pl.GetVertexDataType() != DirectionalParticles::VERTDATA_FLOAT_XYZR)) {
                continue;
            }

            if (index < pl.GetCount()) {
                unsigned int vert_stride = 0;
                if (pl.GetVertexDataType() == DirectionalParticles::VERTDATA_FLOAT_XYZ)
                    vert_stride = 12;
                else if (pl.GetVertexDataType() == DirectionalParticles::VERTDATA_FLOAT_XYZR)
                    vert_stride = 16;
                else
                    continue;
                vert_stride = std::max<unsigned int>(vert_stride, pl.GetVertexDataStride());
                const unsigned char* vert = static_cast<const unsigned char*>(pl.GetVertexData());

                return reinterpret_cast<const float*>(vert + (index * vert_stride));
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
            if ((pl.GetVertexDataType() != DirectionalParticles::VERTDATA_FLOAT_XYZ) &&
                (pl.GetVertexDataType() != DirectionalParticles::VERTDATA_FLOAT_XYZR)) {
                continue;
            }

            if (index < pl.GetCount()) {
                unsigned int dir_stride = 0;
                if (pl.GetDirDataType() == DirectionalParticles::DIRDATA_FLOAT_XYZ)
                    dir_stride = 12;
                else
                    continue;
                dir_stride = std::max<unsigned int>(dir_stride, pl.GetDirDataStride());
                const unsigned char* dir = static_cast<const unsigned char*>(pl.GetDirData());

                return reinterpret_cast<const float*>(dir + (index * dir_stride));
            }

            index -= static_cast<size_t>(pl.GetCount());
        }

        return nullptr;
    }
};


template <typename T> class genericPointCloud {
public:
    genericPointCloud() = default;

    genericPointCloud(size_t const numPts, size_t const stride, std::vector<T>& pts, vislib::math::Cuboid<float> bbox)
        : numPts_(numPts), stride_(stride), pts_(pts), bbox_(bbox) {}

    size_t kdtree_get_point_count() const { return numPts_; }

    T kdtree_get_pt(size_t const idx, int dim) const { return pts_[idx * stride_ + dim]; }

    template <typename BBOX> bool kdtree_get_bbox(BBOX& bb) const {
        bb[0].low = bbox_.Left();
        bb[0].high = bbox_.Right();
        bb[1].low = bbox_.Bottom();
        bb[1].high = bbox_.Top();
        bb[2].low = bbox_.Back();
        bb[2].high = bbox_.Front();
        return true;
    }

private:
    size_t numPts_;

    size_t stride_;

    std::vector<T> pts_;

    vislib::math::Cuboid<float> bbox_;
};


} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MMSTD_DATATOOLS_POINTCLOUDHELPERS_H_INCLUDED */
