/*
 * ParticleColorSignedDistance.h
 *
 * Copyright (C) 2015 by S. Grottel
 * Alle Rechte vorbehalten.
 */
#include "ParticleColorSignedDistance.h"
#include "mmcore/param/BoolParam.h"
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <nanoflann.hpp>

using namespace megamol;


/*
 * datatools::ParticleColorSignedDistance::ParticleColorSignedDistance
 */
datatools::ParticleColorSignedDistance::ParticleColorSignedDistance(void)
        : AbstractParticleManipulator("outData", "indata")
        , enableSlot("enable", "Enables the color manipulation")
        , cyclXSlot("cyclX", "Considders cyclic boundary conditions in X direction")
        , cyclYSlot("cyclY", "Considders cyclic boundary conditions in Y direction")
        , cyclZSlot("cyclZ", "Considders cyclic boundary conditions in Z direction")
        , datahash(0)
        , time(0)
        , newColors()
        , minCol(0.0f)
        , maxCol(1.0f) {

    this->enableSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->enableSlot);

    this->cyclXSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclXSlot);

    this->cyclYSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclYSlot);

    this->cyclZSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclZSlot);
}


/*
 * datatools::ParticleColorSignedDistance::~ParticleColorSignedDistance
 */
datatools::ParticleColorSignedDistance::~ParticleColorSignedDistance(void) {
    this->Release();
}


/*
 * datatools::ParticleColorSignedDistance::manipulateData
 */
bool datatools::ParticleColorSignedDistance::manipulateData(
    geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) {
    using geocalls::MultiParticleDataCall;

    outData = inData;                   // also transfers the unlocker to 'outData'
    inData.SetUnlocker(nullptr, false); // keep original data locked
                                        // original data will be unlocked through outData

    if (!this->enableSlot.Param<core::param::BoolParam>()->Value())
        return true;

    if (this->cyclXSlot.IsDirty()) {
        this->cyclXSlot.ResetDirty();
        this->datahash = 0;
    }
    if (this->cyclYSlot.IsDirty()) {
        this->cyclYSlot.ResetDirty();
        this->datahash = 0;
    }
    if (this->cyclZSlot.IsDirty()) {
        this->cyclZSlot.ResetDirty();
        this->datahash = 0;
    }
    if ((this->datahash == 0) || (this->datahash != outData.DataHash()) || (this->time != outData.FrameID())) {
        this->datahash = outData.DataHash();
        this->time = outData.FrameID();
        this->compute_colors(outData);
    }

    if (this->newColors.size() > 0) {
        this->set_colors(outData);
    }

    return true;
}

namespace {

class pointcloud {
private:
    geocalls::MultiParticleDataCall& dat;
    std::vector<size_t>& indices;

public:
    typedef float coord_t;

    pointcloud(geocalls::MultiParticleDataCall& dat, std::vector<size_t>& indices) : dat(dat), indices(indices) {
        // intentionally empty
    }
    ~pointcloud() {
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
        const auto& cbox = dat.AccessBoundingBoxes().ObjectSpaceClipBox();
        bb[0].low = cbox.Left();
        bb[0].high = cbox.Right();
        bb[1].low = cbox.Bottom();
        bb[1].high = cbox.Top();
        bb[2].low = cbox.Back();
        bb[2].high = cbox.Front();
        return true;
    }

private:
    inline const coord_t* get_position(size_t index) const {
        using geocalls::SimpleSphericalParticles;

        unsigned int plc = dat.GetParticleListCount();
        for (unsigned int pli = 0; pli < plc; pli++) {
            auto& pl = dat.AccessParticles(pli);
            if (pl.GetColourDataType() != SimpleSphericalParticles::COLDATA_FLOAT_I)
                continue;
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

} // namespace


void datatools::ParticleColorSignedDistance::compute_colors(geocalls::MultiParticleDataCall& dat) {
    using geocalls::SimpleSphericalParticles;
    size_t allpartcnt = 0;
    size_t negpartcnt = 0;
    size_t nulpartcnt = 0;
    size_t pospartcnt = 0;
    const float border_epsilon = 0.001f;

    // count particles
    unsigned int plc = dat.GetParticleListCount();
    for (unsigned int pli = 0; pli < plc; pli++) {
        auto& pl = dat.AccessParticles(pli);
        if (pl.GetColourDataType() != SimpleSphericalParticles::COLDATA_FLOAT_I)
            continue;
        if ((pl.GetVertexDataType() != SimpleSphericalParticles::VERTDATA_FLOAT_XYZ) &&
            (pl.GetVertexDataType() != SimpleSphericalParticles::VERTDATA_FLOAT_XYZR)) {
            continue;
        }
        allpartcnt += static_cast<size_t>(pl.GetCount());
    }

    this->newColors.resize(allpartcnt);
    std::vector<size_t> posparts;
    std::vector<size_t> negparts;
    posparts.reserve(allpartcnt);
    negparts.reserve(allpartcnt);

    allpartcnt = 0;
    for (unsigned int pli = 0; pli < plc; pli++) {
        auto& pl = dat.AccessParticles(pli);
        if (pl.GetColourDataType() != SimpleSphericalParticles::COLDATA_FLOAT_I)
            continue;
        if ((pl.GetVertexDataType() != SimpleSphericalParticles::VERTDATA_FLOAT_XYZ) &&
            (pl.GetVertexDataType() != SimpleSphericalParticles::VERTDATA_FLOAT_XYZR)) {
            continue;
        }

        unsigned int vert_stride = 0;
        if (pl.GetVertexDataType() == SimpleSphericalParticles::VERTDATA_FLOAT_XYZ)
            vert_stride = 12;
        else if (pl.GetVertexDataType() == SimpleSphericalParticles::VERTDATA_FLOAT_XYZR)
            vert_stride = 16;
        else
            continue;
        vert_stride = std::max<unsigned int>(vert_stride, pl.GetVertexDataStride());
        const unsigned char* vert = static_cast<const unsigned char*>(pl.GetVertexData());

        int part_cnt = static_cast<int>(pl.GetCount());
        const unsigned char* col = static_cast<const unsigned char*>(pl.GetColourData());
        unsigned int stride = std::max<unsigned int>(pl.GetColourDataStride(), sizeof(float));

        for (int part_i = 0; part_i < part_cnt; ++part_i) {
            float c = *reinterpret_cast<const float*>(col + (part_i * stride));
            const float* v = reinterpret_cast<const float*>(vert + (part_i * vert_stride));

            if (c < -border_epsilon) {
                negpartcnt++;
                negparts.push_back(allpartcnt + part_i);
            } else if (c < border_epsilon) {
                nulpartcnt++;
                negparts.push_back(allpartcnt + part_i);
                posparts.push_back(allpartcnt + part_i);
            } else {
                pospartcnt++;
                posparts.push_back(allpartcnt + part_i);
            }
        }
        allpartcnt += static_cast<size_t>(pl.GetCount());
    }

    // allocate nanoflann data structures for border
    assert(pospartcnt + nulpartcnt == posparts.size());
    pointcloud posnulPts(dat, posparts);
    assert(negpartcnt + nulpartcnt == negparts.size());
    pointcloud negnulPts(dat, negparts);

    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, pointcloud>, pointcloud,
        3 /* dim */, std::size_t>
        my_kd_tree_t;

    my_kd_tree_t posTree(3 /* dim */, posnulPts, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    posTree.buildIndex();
    my_kd_tree_t negTree(3 /* dim */, negnulPts, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    negTree.buildIndex();

    // final computation
    bool cycl_x = this->cyclXSlot.Param<megamol::core::param::BoolParam>()->Value();
    bool cycl_y = this->cyclYSlot.Param<megamol::core::param::BoolParam>()->Value();
    bool cycl_z = this->cyclZSlot.Param<megamol::core::param::BoolParam>()->Value();
    auto bbox = dat.AccessBoundingBoxes().ObjectSpaceBBox();
    bbox.EnforcePositiveSize(); // paranoia
    auto bbox_cntr = bbox.CalcCenter();

    allpartcnt = 0;
    for (unsigned int pli = 0; pli < plc; pli++) {
        auto& pl = dat.AccessParticles(pli);
        if (pl.GetColourDataType() != SimpleSphericalParticles::COLDATA_FLOAT_I)
            continue;
        if ((pl.GetVertexDataType() != SimpleSphericalParticles::VERTDATA_FLOAT_XYZ) &&
            (pl.GetVertexDataType() != SimpleSphericalParticles::VERTDATA_FLOAT_XYZR)) {
            continue;
        }

        unsigned int vert_stride = 0;
        if (pl.GetVertexDataType() == SimpleSphericalParticles::VERTDATA_FLOAT_XYZ)
            vert_stride = 12;
        else if (pl.GetVertexDataType() == SimpleSphericalParticles::VERTDATA_FLOAT_XYZR)
            vert_stride = 16;
        else
            continue;
        vert_stride = std::max<unsigned int>(vert_stride, pl.GetVertexDataStride());
        const unsigned char* vert = static_cast<const unsigned char*>(pl.GetVertexData());

        int part_cnt = static_cast<int>(pl.GetCount());
        const unsigned char* col = static_cast<const unsigned char*>(pl.GetColourData());
        unsigned int col_stride = std::max<unsigned int>(pl.GetColourDataStride(), sizeof(float));

        for (int part_i = 0; part_i < part_cnt; ++part_i) {
            float c = *reinterpret_cast<const float*>(col + (part_i * col_stride));
            const float* v = reinterpret_cast<const float*>(vert + (part_i * vert_stride));

            if ((-border_epsilon < c) && (c < border_epsilon)) {
                c = 0.0f;
            } else {
                float q[3];
                float dist, distsq = static_cast<float>(DBL_MAX);
                my_kd_tree_t& tree = (c < 0.0f) ? posTree : negTree;

                for (int x_s = 0; x_s < (cycl_x ? 2 : 1); ++x_s) {
                    for (int y_s = 0; y_s < (cycl_y ? 2 : 1); ++y_s) {
                        for (int z_s = 0; z_s < (cycl_z ? 2 : 1); ++z_s) {

                            q[0] = v[0];
                            q[1] = v[1];
                            q[2] = v[2];
                            if (x_s > 0)
                                q[0] = v[0] + ((v[0] > bbox_cntr.X()) ? -bbox.Width() : bbox.Width());
                            if (y_s > 0)
                                q[1] = v[1] + ((v[1] > bbox_cntr.Y()) ? -bbox.Height() : bbox.Height());
                            if (z_s > 0)
                                q[2] = v[2] + ((v[2] > bbox_cntr.Z()) ? -bbox.Depth() : bbox.Depth());

                            size_t n_idx;
                            float n_distsq;
                            tree.knnSearch(q, 1, &n_idx, &n_distsq);
                            if (n_distsq < distsq)
                                distsq = n_distsq;
                        }
                    }
                }

                dist = sqrt(distsq);
                if (c < 0.0f)
                    dist = -dist;
                c = static_cast<float>(dist);
            }

            if (c < this->minCol)
                this->minCol = c;
            if (c > this->maxCol)
                this->maxCol = c;

            this->newColors[allpartcnt + part_i] = c;
        }

        allpartcnt += static_cast<size_t>(part_cnt);
    }
}


void datatools::ParticleColorSignedDistance::set_colors(geocalls::MultiParticleDataCall& dat) {
    size_t allpartcnt = 0;

    unsigned int plc = dat.GetParticleListCount();
    for (unsigned int pli = 0; pli < plc; pli++) {
        auto& pl = dat.AccessParticles(pli);
        if (pl.GetColourDataType() != geocalls::SimpleSphericalParticles::COLDATA_FLOAT_I)
            continue;

        pl.SetColourData(geocalls::SimpleSphericalParticles::COLDATA_FLOAT_I, this->newColors.data() + allpartcnt);
        pl.SetColourMapIndexValues(this->minCol, this->maxCol);

        allpartcnt += static_cast<size_t>(pl.GetCount());
    }
}
