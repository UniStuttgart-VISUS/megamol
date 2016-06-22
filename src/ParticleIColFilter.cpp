/*
 * ParticleIColFilter.h
 *
 * Copyright (C) 2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "ParticleIColFilter.h"
#include "mmcore/param/FloatParam.h"
#include <algorithm>

using namespace megamol;
using namespace megamol::stdplugin;

datatools::ParticleIColFilter::ParticleIColFilter() : AbstractParticleManipulator("outData", "inData"),
        minValSlot("minVal", "The minimal color value of particles to be passed on"),
        maxValSlot("maxVal", "The maximal color value of particles to be passed on"),
        staifHackDistSlot("staifHackDist", "Distance to the bounding box to include particles"),
        dataHash(0), frameId(0), parts(), data() {
    minValSlot.SetParameter(new core::param::FloatParam(0.0f));
    minValSlot.SetUpdateCallback(&ParticleIColFilter::reset);
    MakeSlotAvailable(&minValSlot);
    maxValSlot.SetParameter(new core::param::FloatParam(1.0f));
    maxValSlot.SetUpdateCallback(&ParticleIColFilter::reset);
    MakeSlotAvailable(&maxValSlot);
    staifHackDistSlot.SetParameter(new core::param::FloatParam(0.0f, 0.0f));
    staifHackDistSlot.SetUpdateCallback(&ParticleIColFilter::reset);
    MakeSlotAvailable(&staifHackDistSlot);
}

datatools::ParticleIColFilter::~ParticleIColFilter() {
    this->Release();
}

bool datatools::ParticleIColFilter::manipulateData(
        megamol::core::moldyn::MultiParticleDataCall& outData,
        megamol::core::moldyn::MultiParticleDataCall& inData) {

    if ((frameId != inData.FrameID()) || (dataHash != inData.DataHash()) || (inData.DataHash() == 0)) {
        frameId = inData.FrameID();
        dataHash = inData.DataHash();
        setData(inData);
    }
    inData.Unlock();

    outData.SetDataHash(dataHash);
    outData.SetFrameID(frameId);
    outData.SetParticleListCount(static_cast<unsigned int>(parts.size()));
    for (size_t i = 0; i < parts.size(); ++i) outData.AccessParticles(static_cast<unsigned int>(i)) = parts[i];
    outData.SetUnlocker(nullptr); // HAZARD: we could have one ...

    return true;
}

bool datatools::ParticleIColFilter::reset(core::param::ParamSlot&) {
    dataHash = 0;
    return true;
}

void datatools::ParticleIColFilter::setData(core::moldyn::MultiParticleDataCall& inDat) {
    unsigned int cnt = inDat.GetParticleListCount();
    parts.resize(cnt);
    data.resize(cnt);

    for (unsigned int i = 0; i < cnt; ++i) {
        setData(parts[i], data[i], inDat.AccessParticles(i), inDat.AccessBoundingBoxes().ObjectSpaceBBox());
    }

}

//namespace {
//
//    inline bool partTest(const float *ci, const float *vi, float minC, float maxC, float d, const vislib::math::Cuboid<float>& bbox) {
//        return ((minC <= *ci) && (*ci <= maxC))
//            && ((d < minStaifDist)
//            || ())
//    }
//}

void datatools::ParticleIColFilter::setData(core::moldyn::MultiParticleDataCall::Particles& p, vislib::RawStorage& d, const core::moldyn::SimpleSphericalParticles& s, vislib::math::Cuboid<float> bbox) {
    using core::moldyn::MultiParticleDataCall;
    using core::moldyn::SimpleSphericalParticles;
    using vislib::RawStorage;

    p.SetCount(0);
    p.SetVertexData(SimpleSphericalParticles::VERTDATA_NONE, nullptr);
    p.SetColourData(SimpleSphericalParticles::COLDATA_NONE, nullptr);

    if (s.GetVertexDataType() == SimpleSphericalParticles::VERTDATA_NONE) return; // No data is no data
    if (s.GetColourDataType() != SimpleSphericalParticles::COLDATA_FLOAT_I) return; // for now, wrongly formated data is simply removed

    const uint64_t cnt = s.GetCount();

    const uint8_t* vp = reinterpret_cast<const uint8_t*>(s.GetVertexData());
    int v_size = 0;
    int v_step = s.GetVertexDataStride();
    switch (s.GetVertexDataType()) {
    case SimpleSphericalParticles::VERTDATA_FLOAT_XYZ: v_size = 12; break;
    case SimpleSphericalParticles::VERTDATA_FLOAT_XYZR: v_size = 16; break;
    case SimpleSphericalParticles::VERTDATA_SHORT_XYZ: v_size = 6; break;
    default: assert(false);
    }
    if (v_step < v_size) v_step = v_size;

    const uint8_t* cp = reinterpret_cast<const uint8_t*>(s.GetColourData());
    int c_size = 4;
    int c_step = s.GetColourDataStride();
    if (c_step < c_size) c_step = c_size;

    float minVal = minValSlot.Param<core::param::FloatParam>()->Value();
    float maxVal = maxValSlot.Param<core::param::FloatParam>()->Value();
    if (maxVal < minVal) std::swap(minVal, maxVal);

    //const float minStaifDist = 0.0001f;
    //float staifDist = staifHackDistSlot.Param<core::param::FloatParam>()->Value();
    //if (staifDist > minStaifDist) {
    //    bbox.Grow(-staifDist);
    //} else {
    //    bbox.Set(-1000.0f, -1000.0f, -1000.0f, -1000.0f, -1000.0f, -1000.0f);
    //}

    // now count particles surviving
    uint64_t r_cnt = 0;
    for (uint64_t i = 0; i < cnt; ++i) {
        const float *ci = reinterpret_cast<const float*>(cp + c_step * i);
        const float *vi = reinterpret_cast<const float*>(vp + v_step * i);
        if ((minVal <= *ci) && (*ci <= maxVal)) { // || !bbox.Contains(vislib::math::Point<float, 3>(vi))) {
            r_cnt++;
        }
    }

    // now copying particles
    d.AssertSize(static_cast<size_t>(r_cnt * (v_size + c_size)));
    const size_t c_off = static_cast<size_t>(r_cnt * v_size);
    p.SetCount(r_cnt);
    p.SetGlobalRadius(s.GetGlobalRadius());
    p.SetGlobalColour(s.GetGlobalColour()[0], s.GetGlobalColour()[1], s.GetGlobalColour()[2], s.GetGlobalColour()[3]);
    p.SetVertexData(s.GetVertexDataType(), d);
    p.SetColourData(SimpleSphericalParticles::COLDATA_FLOAT_I, d.At(c_off));
    p.SetColourMapIndexValues(s.GetMinColourIndexValue(), s.GetMaxColourIndexValue());

    r_cnt = 0;
    for (size_t i = 0; i < cnt; ++i) {
        const float *ci = reinterpret_cast<const float*>(cp + c_step * i);
        const float *vi = reinterpret_cast<const float*>(vp + v_step * i);
        if ((minVal <= *ci) && (*ci <= maxVal)) { // || !bbox.Contains(vislib::math::Point<float, 3>(vi))) {
            ::memcpy(d.At(static_cast<size_t>(r_cnt * v_size)), vi, static_cast<size_t>(v_size));
            ::memcpy(d.At(static_cast<size_t>(c_off + r_cnt * c_size)), ci, static_cast<size_t>(c_size));
            r_cnt++;
        }
    }

}
