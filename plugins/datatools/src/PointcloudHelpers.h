/*
 * PointcloudHelpers.h
 *
 * Copyright (C) 2017 by MegaMol team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "geometry_calls/MultiParticleDataCall.h"
#include <vector>

namespace megamol {
namespace datatools {

/**
 * Class that implements the interface nanoflann needs for simple spherical particles.
 * The index vector addresses all of the particles across all lists, so its
 * range is (0,Sum(Allof(particleLists).Count)).
 */
class simplePointcloud {
private:
    geocalls::MultiParticleDataCall* dat;
    std::vector<size_t>& indices;
    bool cycleX, cycleY, cycleZ;

public:
    typedef float coord_t;

    simplePointcloud(geocalls::MultiParticleDataCall* dat, std::vector<size_t>& indices) : dat(dat), indices(indices) {
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
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template<class BBOX>
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
        using geocalls::SimpleSphericalParticles;

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

    /// HAZARD BUG TODO this blows up spectacularly if you mix lists with and without velocities.
    /// alternatively, it wrongly associates velocities to particles that have none
    /// AND you cannot skip lists without velocity in get_position because that makes no sense either
    inline const coord_t* get_velocity(size_t index) const {
        using geocalls::SimpleSphericalParticles;

        unsigned int plc = dat->GetParticleListCount();
        for (unsigned int pli = 0; pli < plc; pli++) {
            auto& pl = dat->AccessParticles(pli);
            if ((pl.GetVertexDataType() != SimpleSphericalParticles::VERTDATA_FLOAT_XYZ) &&
                (pl.GetVertexDataType() != SimpleSphericalParticles::VERTDATA_FLOAT_XYZR)) {
                continue;
            }

            if (index < pl.GetCount()) {
                unsigned int dir_stride = 0;
                if (pl.GetDirDataType() == SimpleSphericalParticles::DIRDATA_FLOAT_XYZ)
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


template<typename T, int DIM>
class genericPointcloud {
public:
    using coord_t = T;

    genericPointcloud(std::vector<T> const& data, std::array<T, 2 * DIM> const& bbox, std::array<T, DIM> const& weights)
            : _point_data(data)
            , _bbox(bbox)
            , _weights(weights) {}

    ~genericPointcloud() = default;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const {
        return _point_data.size() / static_cast<std::size_t>(DIM);
    }

    // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the
    // class:
    inline coord_t kdtree_distance(const coord_t* p1, const size_t idx_p2, size_t /*size*/) const {
        coord_t const* p2 = get_position(idx_p2);

        coord_t res = 0;

        for (int d = 0; d < DIM; ++d) {
            res += _weights[d] * (p1[d] - p2[d]) * (p1[d] - p2[d]);
        }

        return res;
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline coord_t kdtree_get_pt(const size_t idx, int dim) const {
        return get_position(idx)[dim];
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it
    //   again. Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template<typename BBOX>
    bool kdtree_get_bbox(BBOX& bb) const {
        for (int d = 0; d < DIM; ++d) {
            bb[d].low = _bbox[d * 2];
            bb[d].high = _bbox[d * 2 + 1];
        }

        return true;
    }

    coord_t const* get_position(std::size_t idx) const {
        return &(_point_data[idx * static_cast<std::size_t>(DIM)]);
    }

    void normalize_data() {
        std::array<T, DIM> mins;
        std::array<T, DIM> divs;
        for (int d = 0; d < DIM; ++d) {
            mins[d] = _bbox[d * 2];
            divs[d] = static_cast<T>(1.0f) / (_bbox[d * 2 + 1] - _bbox[d * 2] + static_cast<T>(1e-8f));
        }

        auto const num_points = kdtree_get_point_count();
        for (std::size_t pidx = 0; pidx < num_points; ++pidx) {
            for (int d = 0; d < DIM; ++d) {
                _point_data[pidx * DIM + d] -= mins[d];
                _point_data[pidx * DIM + d] *= divs[d];
            }
        }

        for (int d = 0; d < DIM; ++d) {
            _bbox[d * 2] = 0.0f;
            _bbox[d * 2 + 1] = 1.0f;
        }
    }

private:
    std::vector<T> _point_data;

    std::array<T, 2 * DIM> _bbox;

    std::array<T, DIM> _weights;
};

} /* end namespace datatools */
} /* end namespace megamol */
