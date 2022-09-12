/*
 * ParticleIColGradientField.h
 *
 * Copyright (C) 2016 by S. Grottel
 * Alle Rechte vorbehalten.
 */
#include "ParticleIColGradientField.h"
#include "datatools/MultiParticleDataAdaptor.h"

#include "mmcore/param/FloatParam.h"
#include "vislib/math/ShallowPoint.h"
#include "vislib/math/Vector.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <nanoflann.hpp>
#include <utility>

using namespace megamol;


/*
 * datatools::ParticleIColGradientField::ParticleIColGradientField
 */
datatools::ParticleIColGradientField::ParticleIColGradientField(void)
        : AbstractParticleManipulator("outData", "indata")
        , radiusSlot("radius", "The neighbourhood radius size")
        , datahash(0)
        , time(0)
        , newColors() {

    this->radiusSlot.SetParameter(new core::param::FloatParam(0.05f, 0.000001f));
    this->MakeSlotAvailable(&this->radiusSlot);
}


/*
 * datatools::ParticleIColGradientField::~ParticleIColGradientField
 */
datatools::ParticleIColGradientField::~ParticleIColGradientField(void) {
    this->Release();
}


/*
 * datatools::ParticleIColGradientField::manipulateData
 */
bool datatools::ParticleIColGradientField::manipulateData(
    geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) {
    using geocalls::MultiParticleDataCall;

    outData = inData;                   // also transfers the unlocker to 'outData'
    inData.SetUnlocker(nullptr, false); // keep original data locked
                                        // original data will be unlocked through outData

    if (this->radiusSlot.IsDirty()) {
        this->radiusSlot.ResetDirty();
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

/**
 * Utility class forming a data adapter for nanoflann
 */
class DataAdapter : public datatools::MultiParticleDataAdaptor {
public:
    DataAdapter(geocalls::MultiParticleDataCall& dat) : datatools::MultiParticleDataAdaptor(dat) {
        // intentionally empty
    }
    ~DataAdapter() {
        // intentionally empty
    }
    // Must return the number of data points
    inline size_t kdtree_get_point_count() const {
        return get_count();
    }

    typedef float coord_t;

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
        const auto& cbox = data.GetBoundingBoxes().ObjectSpaceClipBox();
        bb[0].low = cbox.Left();
        bb[0].high = cbox.Right();
        bb[1].low = cbox.Bottom();
        bb[1].high = cbox.Top();
        bb[2].low = cbox.Back();
        bb[2].high = cbox.Front();
        return true;
    }
};
} // namespace

void datatools::ParticleIColGradientField::compute_colors(geocalls::MultiParticleDataCall& dat) {
    DataAdapter data(dat);

    this->newColors.resize(data.kdtree_get_point_count() /* * 3*/);
    float rad = this->radiusSlot.Param<core::param::FloatParam>()->Value();

    // construct a kd-tree index:
    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, DataAdapter>, DataAdapter,
        3 /* dim */, std::size_t>
        my_kd_tree_t;

    my_kd_tree_t index(3 /*dim*/, data, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    index.buildIndex();

    double maxLen = 0.0;

    std::vector<std::pair<size_t, float>> res;
    res.reserve(100);

    //#pragma omp parallel for
    for (int part_i = 0; part_i < static_cast<int>(data.kdtree_get_point_count()); ++part_i) {
        // compute gradient vector for point i

        vislib::math::ShallowPoint<const float, 3> query_pos(data.get_position(part_i));
        const float* query_col = data.get_color(part_i);

        res.clear();
        index.radiusSearch(query_pos.PeekCoordinates(), rad, res, nanoflann::SearchParams(10, 0.01f, false));

        vislib::math::Vector<double, 3> gradient;

        for (std::pair<size_t, float>& p : res) {
            vislib::math::Vector<double, 3> dir(
                vislib::math::ShallowPoint<float, 3>(const_cast<float*>(data.get_position(p.first))) - query_pos);
            dir.Normalise();
            double colDiff = static_cast<double>(*data.get_color(p.first)) - static_cast<double>(*query_col);
            //double weight = static_cast<double>(rad - p.second) / static_cast<double>(rad);

            dir *= colDiff;
            //dir *= weight;

            gradient += dir;
        }

        gradient /= static_cast<double>(res.size());
        double len = gradient.Length();
        if (len > maxLen)
            maxLen = len;

        newColors[part_i] = static_cast<float>(len);

        //newColors[part_i * 3 + 0] = static_cast<float>(gradient[0]);
        //newColors[part_i * 3 + 1] = static_cast<float>(gradient[1]);
        //newColors[part_i * 3 + 2] = static_cast<float>(gradient[2]);
    }

    maxColor = static_cast<float>(maxLen);
    //printf("%f\n", maxLen);
}


void datatools::ParticleIColGradientField::set_colors(geocalls::MultiParticleDataCall& dat) {
    size_t allpartcnt = 0;

    unsigned int plc = dat.GetParticleListCount();
    for (unsigned int pli = 0; pli < plc; pli++) {
        auto& pl = dat.AccessParticles(pli);
        if ((pl.GetColourDataType() != geocalls::SimpleSphericalParticles::COLDATA_FLOAT_I) ||
            ((pl.GetVertexDataType() != geocalls::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ) &&
                (pl.GetVertexDataType() != geocalls::SimpleSphericalParticles::VERTDATA_FLOAT_XYZR)))
            continue;

        //pl.SetColourData(geocalls::SimpleSphericalParticles::COLDATA_FLOAT_RGB, this->newColors.data() + allpartcnt * 3);
        //pl.SetColourMapIndexValues(-1.0f, 1.0f);
        pl.SetColourData(geocalls::SimpleSphericalParticles::COLDATA_FLOAT_I, this->newColors.data() + allpartcnt);
        pl.SetColourMapIndexValues(0.0f, maxColor);

        allpartcnt += static_cast<size_t>(pl.GetCount());
    }
}
