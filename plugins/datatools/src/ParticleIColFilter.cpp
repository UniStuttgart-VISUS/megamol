/*
 * ParticleIColFilter.h
 *
 * Copyright (C) 2015-2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "ParticleIColFilter.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector2fParam.h"
#include <algorithm>

using namespace megamol;

datatools::ParticleIColFilter::ParticleIColFilter()
        : AbstractParticleManipulator("outData", "inData")
        , particleMapSlot("outParticleMap", "Publishes the particle map data")
        , minValSlot("minVal", "The minimal color value of particles to be passed on")
        , maxValSlot("maxVal", "The maximal color value of particles to be passed on")
        , staifHackDistSlot("staifHackDist", "Distance to the bounding box to include particles")
        , dataHash(0)
        , frameId(0)
        , parts()
        , data()
        , mapIndex()
        , inValRangeSlot("inValRange", "Displays the value range of the input color values") {

    minValSlot.SetParameter(new core::param::FloatParam(0.0f));
    minValSlot.SetUpdateCallback(&ParticleIColFilter::reset);
    MakeSlotAvailable(&minValSlot);

    maxValSlot.SetParameter(new core::param::FloatParam(1.0f));
    maxValSlot.SetUpdateCallback(&ParticleIColFilter::reset);
    MakeSlotAvailable(&maxValSlot);

    // staifHackDistSlot.SetParameter(new core::param::FloatParam(0.0f, 0.0f));
    // staifHackDistSlot.SetUpdateCallback(&ParticleIColFilter::reset);
    // MakeSlotAvailable(&staifHackDistSlot);

    inValRangeSlot.SetParameter(new core::param::Vector2fParam(vislib::math::Vector<float, 2>(0.0f, 1.0f)));
    MakeSlotAvailable(&inValRangeSlot);

    particleMapSlot.SetCallback(ParticleFilterMapDataCall::ClassName(),
        ParticleFilterMapDataCall::FunctionName(ParticleFilterMapDataCall::GET_DATA),
        &ParticleIColFilter::getParticleMapData);
    particleMapSlot.SetCallback(ParticleFilterMapDataCall::ClassName(),
        ParticleFilterMapDataCall::FunctionName(ParticleFilterMapDataCall::GET_EXTENT),
        &ParticleIColFilter::getParticleMapExtent);
    particleMapSlot.SetCallback(ParticleFilterMapDataCall::ClassName(),
        ParticleFilterMapDataCall::FunctionName(ParticleFilterMapDataCall::GET_HASH),
        &ParticleIColFilter::getParticleMapHash);
    MakeSlotAvailable(&particleMapSlot);
}

datatools::ParticleIColFilter::~ParticleIColFilter() {
    this->Release();
}

bool datatools::ParticleIColFilter::manipulateData(
    geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) {

    if ((frameId != inData.FrameID()) || (dataHash != inData.DataHash()) || (inData.DataHash() == 0) || isDirty()) {
        frameId = inData.FrameID();
        dataHash = inData.DataHash();
        setData(inData);
        resetDirty();
        ++outDataHash;
    }
    inData.Unlock();

    outData.SetDataHash(outDataHash);
    outData.SetFrameID(frameId);
    outData.SetParticleListCount(static_cast<unsigned int>(parts.size()));
    for (size_t i = 0; i < parts.size(); ++i)
        outData.AccessParticles(static_cast<unsigned int>(i)) = parts[i];
    outData.SetUnlocker(nullptr); // HAZARD: we could have one ...

    return true;
}

bool datatools::ParticleIColFilter::reset(core::param::ParamSlot&) {
    dataHash = 0;
    return true;
}

void datatools::ParticleIColFilter::setData(geocalls::MultiParticleDataCall& inDat) {
    unsigned int cnt = inDat.GetParticleListCount();
    parts.resize(cnt);
    data.resize(cnt);
    mapIndex.clear();
    ParticleFilterMapDataCall::index_t mapOffset = 0;

    float minV, maxV;
    for (unsigned int i = 0; i < cnt; ++i) {
        setData(parts[i], data[i], inDat.AccessParticles(i), inDat.AccessBoundingBoxes().ObjectSpaceBBox(), mapOffset);
        if (i == 0) {
            minV = inDat.AccessParticles(i).GetMinColourIndexValue();
            maxV = inDat.AccessParticles(i).GetMaxColourIndexValue();
        } else {
            minV = std::min<float>(minV, inDat.AccessParticles(i).GetMinColourIndexValue());
            maxV = std::max<float>(maxV, inDat.AccessParticles(i).GetMaxColourIndexValue());
        }
    }

    inValRangeSlot.Param<core::param::Vector2fParam>()->SetValue(vislib::math::Vector<float, 2>(minV, maxV), false);
}

// namespace {
//
//    inline bool partTest(const float *ci, const float *vi, float minC, float maxC, float d, const
//    vislib::math::Cuboid<float>& bbox) {
//        return ((minC <= *ci) && (*ci <= maxC))
//            && ((d < minStaifDist)
//            || ())
//    }
//}

void datatools::ParticleIColFilter::setData(geocalls::MultiParticleDataCall::Particles& p, vislib::RawStorage& d,
    const geocalls::SimpleSphericalParticles& s, vislib::math::Cuboid<float> bbox,
    ParticleFilterMapDataCall::index_t& mapOffset) {
    using geocalls::MultiParticleDataCall;
    using geocalls::SimpleSphericalParticles;
    using vislib::RawStorage;

    p.SetCount(0);
    p.SetVertexData(SimpleSphericalParticles::VERTDATA_NONE, nullptr);
    p.SetColourData(SimpleSphericalParticles::COLDATA_NONE, nullptr);

    if (s.GetVertexDataType() == SimpleSphericalParticles::VERTDATA_NONE)
        return; // No data is no data
    if (s.GetColourDataType() != SimpleSphericalParticles::COLDATA_FLOAT_I)
        return; // for now, wrongly formated data is simply removed

    const uint64_t cnt = s.GetCount();

    uint8_t const* ip = reinterpret_cast<uint8_t const*>(s.GetIDData());
    int i_size = 0;
    int i_step = s.GetIDDataStride();
    switch (s.GetIDDataType()) {
    case SimpleSphericalParticles::IDDATA_UINT32:
        i_size = 4;
        break;
    case SimpleSphericalParticles::IDDATA_UINT64:
        i_size = 8;
        break;
    default:
        break;
    }
    if (i_step < i_size)
        i_step = i_size;

    const uint8_t* vp = reinterpret_cast<const uint8_t*>(s.GetVertexData());
    int v_size = 0;
    int v_step = s.GetVertexDataStride();
    switch (s.GetVertexDataType()) {
    case SimpleSphericalParticles::VERTDATA_FLOAT_XYZ:
        v_size = 12;
        break;
    case SimpleSphericalParticles::VERTDATA_FLOAT_XYZR:
        v_size = 16;
        break;
    case SimpleSphericalParticles::VERTDATA_SHORT_XYZ:
        v_size = 6;
        break;
    default:
        assert(false);
    }
    if (v_step < v_size)
        v_step = v_size;

    const uint8_t* cp = reinterpret_cast<const uint8_t*>(s.GetColourData());
    int c_size = 4;
    int c_step = s.GetColourDataStride();
    if (c_step < c_size)
        c_step = c_size;

    float minVal = minValSlot.Param<core::param::FloatParam>()->Value();
    float maxVal = maxValSlot.Param<core::param::FloatParam>()->Value();
    if (maxVal < minVal)
        std::swap(minVal, maxVal);

    uint8_t const* dp = reinterpret_cast<uint8_t const*>(s.GetDirData());
    int d_size = 0;
    int d_step = s.GetDirDataStride();
    switch (s.GetDirDataType()) {
    case SimpleSphericalParticles::DIRDATA_FLOAT_XYZ:
        d_size = 12;
        break;
    default:
        break;
    }
    if (d_step < d_size)
        d_step = d_size;


    // const float minStaifDist = 0.0001f;
    // float staifDist = staifHackDistSlot.Param<core::param::FloatParam>()->Value();
    // if (staifDist > minStaifDist) {
    //    bbox.Grow(-staifDist);
    //} else {
    //    bbox.Set(-1000.0f, -1000.0f, -1000.0f, -1000.0f, -1000.0f, -1000.0f);
    //}

    // now count particles surviving
    uint64_t r_cnt = 0;
    for (uint64_t i = 0; i < cnt; ++i) {
        const float* ci = reinterpret_cast<const float*>(cp + c_step * i);
        // const float *vi = reinterpret_cast<const float*>(vp + v_step * i);
        if ((minVal <= *ci) && (*ci <= maxVal)) { // || !bbox.Contains(vislib::math::Point<float, 3>(vi))) {
            r_cnt++;
        }
    }

    if (r_cnt == 0) {
        p.SetCount(0);
        p.SetGlobalRadius(s.GetGlobalRadius());
        p.SetGlobalColour(
            s.GetGlobalColour()[0], s.GetGlobalColour()[1], s.GetGlobalColour()[2], s.GetGlobalColour()[3]);
        p.SetVertexData(SimpleSphericalParticles::VERTDATA_NONE, nullptr);
        p.SetColourData(SimpleSphericalParticles::COLDATA_NONE, nullptr);
        p.SetDirData(SimpleSphericalParticles::DIRDATA_NONE, nullptr);
        p.SetIDData(SimpleSphericalParticles::IDDATA_NONE, nullptr);
        p.SetColourMapIndexValues(s.GetMinColourIndexValue(), s.GetMaxColourIndexValue());
        return;
    }

    // now copying particles
    mapIndex.reserve(mapIndex.size() + static_cast<size_t>(r_cnt));
    d.AssertSize(static_cast<size_t>(r_cnt * (v_size + c_size + d_size + i_size)));
    const size_t v_off = static_cast<size_t>(r_cnt * i_size);
    const size_t c_off = static_cast<size_t>(v_off + r_cnt * v_size);
    const size_t d_off = static_cast<size_t>(c_off + r_cnt * c_size);
    p.SetCount(r_cnt);
    p.SetGlobalRadius(s.GetGlobalRadius());
    p.SetGlobalColour(s.GetGlobalColour()[0], s.GetGlobalColour()[1], s.GetGlobalColour()[2], s.GetGlobalColour()[3]);
    if (i_size != 0)
        p.SetIDData(s.GetIDDataType(), d);
    else
        p.SetIDData(SimpleSphericalParticles::IDDATA_NONE, nullptr);
    p.SetVertexData(s.GetVertexDataType(), d.At(v_off));
    p.SetColourData(SimpleSphericalParticles::COLDATA_FLOAT_I, d.At(c_off));
    if (d_size != 0)
        p.SetDirData(SimpleSphericalParticles::DIRDATA_FLOAT_XYZ, d.At(d_off));
    else
        p.SetDirData(SimpleSphericalParticles::DIRDATA_NONE, nullptr);
    p.SetColourMapIndexValues(s.GetMinColourIndexValue(), s.GetMaxColourIndexValue());

    r_cnt = 0;
    for (size_t i = 0; i < cnt; ++i) {
        const float* ci = reinterpret_cast<const float*>(cp + c_step * i);
        const float* vi = reinterpret_cast<const float*>(vp + v_step * i);
        const float* di = reinterpret_cast<const float*>(dp + d_step * i);
        const float* ii = reinterpret_cast<const float*>(ip + i_step * i);
        if ((minVal <= *ci) && (*ci <= maxVal)) { // || !bbox.Contains(vislib::math::Point<float, 3>(vi))) {
            if (i_size != 0)
                memcpy(d.At(static_cast<size_t>(r_cnt * i_size)), ii, static_cast<size_t>(i_size));
            ::memcpy(d.At(static_cast<size_t>(v_off + r_cnt * v_size)), vi, static_cast<size_t>(v_size));
            ::memcpy(d.At(static_cast<size_t>(c_off + r_cnt * c_size)), ci, static_cast<size_t>(c_size));
            if (d_size != 0)
                memcpy(d.At(static_cast<size_t>(d_off + r_cnt * d_size)), di, static_cast<size_t>(d_size));
            r_cnt++;
            mapIndex.push_back(static_cast<ParticleFilterMapDataCall::index_t>(mapOffset + i));
        }
    }

    mapOffset += static_cast<ParticleFilterMapDataCall::index_t>(cnt);
}

bool datatools::ParticleIColFilter::getParticleMapData(core::Call& c) {
    ParticleFilterMapDataCall* mapCall = dynamic_cast<ParticleFilterMapDataCall*>(&c);
    if (mapCall == nullptr)
        return false;

    mapCall->Set(mapIndex.data(), mapIndex.size());
    mapCall->SetUnlocker(nullptr);

    return true;
}

bool datatools::ParticleIColFilter::getParticleMapExtent(core::Call& c) {
    ParticleFilterMapDataCall* mapCall = dynamic_cast<ParticleFilterMapDataCall*>(&c);
    if (mapCall == nullptr)
        return false;

    mapCall->SetFrameCount(0); // not supported by this module :-/
    mapCall->SetUnlocker(nullptr);

    return true;
}

bool datatools::ParticleIColFilter::getParticleMapHash(core::Call& c) {
    ParticleFilterMapDataCall* mapCall = dynamic_cast<ParticleFilterMapDataCall*>(&c);
    if (mapCall == nullptr)
        return false;

    mapCall->SetDataHash(dataHash);
    mapCall->SetUnlocker(nullptr);

    return true;
}
