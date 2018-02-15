/*
 * ParticleThermometer.cpp
 *
 * Copyright (C) 2017 by MegaMol team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "ParticleThermometer.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/IntParam.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/ConsoleProgressBar.h"
#include <cstdint>
#include <algorithm>
#include <cfloat>
#include <cassert>
#include <limits>
#include <omp.h>

using namespace megamol;
using namespace megamol::stdplugin;

/*
 * datatools::ParticleThermometer::ParticleThermometer
 */
datatools::ParticleThermometer::ParticleThermometer(void)
        : cyclXSlot("cyclX", "Considers cyclic boundary conditions in X direction"),
        cyclYSlot("cyclY", "Considers cyclic boundary conditions in Y direction"),
        cyclZSlot("cyclZ", "Considers cyclic boundary conditions in Z direction"),
        radiusSlot("radius", "the radius in which to look for neighbors"),
        numNeighborSlot("numNeighbors", "how many neighbors to collect"),
        searchTypeSlot("searchType", "num of neighbors or radius"),
        minTempSlot("minTemp", "the detected minimum temperature"),
        maxTempSlot("maxTemp", "the detected maximum temperature"),
        massSlot("mass", "the mass of the particles"),
        freedomSlot("freedomFactor", "factor reducing T* based on degrees of freedom of the molecular model"),
        toggleNewColorSlot("newColor", "toggles between output of new color and default color"),
        toggleNewVelocitySlot("newVelocities", "toggles between output of new and old velocities"),
        outDataSlot("outData", "Provides colors based on local particle temperature"),
        inDataSlot("inData", "Takes the directional particle data"),
        maxDist(0.0f),
        datahash(0), lastTime(-1), newColors(), allParts(), particleTree(nullptr), myPts(nullptr) {

    this->cyclXSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclXSlot);

    this->cyclYSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclYSlot);

    this->cyclZSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclZSlot);

    this->radiusSlot.SetParameter(new core::param::FloatParam(2.0));
    this->MakeSlotAvailable(&this->radiusSlot);

    this->numNeighborSlot.SetParameter(new core::param::IntParam(10));
    this->MakeSlotAvailable(&this->numNeighborSlot);

    core::param::EnumParam *st = new core::param::EnumParam(searchTypeEnum::NUM_NEIGHBORS);
    st->SetTypePair(searchTypeEnum::RADIUS, "Radius");
    st->SetTypePair(searchTypeEnum::NUM_NEIGHBORS, "Num. Neighbors");
    this->searchTypeSlot << st;
    this->MakeSlotAvailable(&this->searchTypeSlot);

    this->minTempSlot.SetParameter(new core::param::FloatParam(0));
    this->MakeSlotAvailable(&this->minTempSlot);

    this->maxTempSlot.SetParameter(new core::param::FloatParam(0));
    this->MakeSlotAvailable(&this->maxTempSlot);

    this->massSlot.SetParameter(new core::param::FloatParam(1.0f));
    this->MakeSlotAvailable(&this->massSlot);

    this->freedomSlot.SetParameter(new core::param::FloatParam(1.5f)); // works for single-center models. 3 degrees of freedom -> 3/2
    this->MakeSlotAvailable(&this->freedomSlot);

    this->toggleNewColorSlot << new megamol::core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->toggleNewColorSlot);

    this->toggleNewVelocitySlot << new megamol::core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->toggleNewVelocitySlot);

    this->outDataSlot.SetCallback(megamol::core::moldyn::DirectionalParticleDataCall::ClassName(), "GetData", &ParticleThermometer::getDataCallback);
    this->outDataSlot.SetCallback(megamol::core::moldyn::DirectionalParticleDataCall::ClassName(), "GetExtent", &ParticleThermometer::getExtentCallback);
    this->outDataSlot.SetCallback(megamol::core::moldyn::MultiParticleDataCall::ClassName(), "GetData", &ParticleThermometer::getDataCallback);
    this->outDataSlot.SetCallback(megamol::core::moldyn::MultiParticleDataCall::ClassName(), "GetExtent", &ParticleThermometer::getExtentCallback);
    this->MakeSlotAvailable(&this->outDataSlot);

    this->inDataSlot.SetCompatibleCall<megamol::core::moldyn::DirectionalParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);
}


/*
 * datatools::ParticleColorSignedDistance::~ParticleColorSignedDistance
 */
datatools::ParticleThermometer::~ParticleThermometer(void) {
    this->Release();
}

/*
* datatools::ParticleThermometer::create
*/
bool datatools::ParticleThermometer::create(void) {
    return true;
}


bool isListOK(megamol::core::moldyn::DirectionalParticleDataCall *in, unsigned int i) {
    using megamol::core::moldyn::DirectionalParticleDataCall;
    auto& pl = in->AccessParticles(i);
    return pl.GetVertexDataType() == DirectionalParticleDataCall::Particles::VertexDataType::VERTDATA_FLOAT_XYZ
        || DirectionalParticleDataCall::Particles::VertexDataType::VERTDATA_FLOAT_XYZR;
}


/*
* datatools::ParticleThermometer::release
*/
void datatools::ParticleThermometer::release(void) {
}

bool datatools::ParticleThermometer::assertData(core::moldyn::DirectionalParticleDataCall *in,
    core::moldyn::MultiParticleDataCall *outMPDC, core::moldyn::DirectionalParticleDataCall *outDPDC) {

    using megamol::core::moldyn::DirectionalParticleDataCall;
    using megamol::core::moldyn::MultiParticleDataCall;

    megamol::core::AbstractGetData3DCall *out;
    if (outMPDC != nullptr) out = outMPDC;
    if (outDPDC != nullptr) out = outDPDC;

    unsigned int time = out->FrameID();
    unsigned int plc = in->GetParticleListCount();
    float theRadius = this->radiusSlot.Param<core::param::FloatParam>()->Value();
    theRadius = theRadius * theRadius;
    float theMass = this->massSlot.Param<core::param::FloatParam>()->Value();
    float theFreedom = this->freedomSlot.Param<core::param::FloatParam>()->Value();
    int theNumber = this->numNeighborSlot.Param<core::param::IntParam>()->Value();
    auto theSearchType = this->searchTypeSlot.Param<core::param::EnumParam>()->Value();
    size_t allpartcnt = 0;

    if (this->lastTime != time || this->datahash != in->DataHash()) {
        in->SetFrameID(time, true);

        if (!(*in)(0)) {
            vislib::sys::Log::DefaultLog.WriteError("ParticleThermometer: could not get frame (%u)", time);
            return false;
        }

        size_t totalParts = 0;
        plc = in->GetParticleListCount();

        for (unsigned int i = 0; i < plc; i++) {
            if (isListOK(in, i))
                totalParts += in->AccessParticles(i).GetCount();
        }

        if (theSearchType == searchTypeEnum::RADIUS) {
            this->newColors.resize(totalParts, theRadius);
        } else {
            this->newColors.resize(totalParts);
        }

        this->newVelocities.resize(totalParts * 3);

        allParts.clear();
        allParts.reserve(totalParts);

        // we could now filter particles according to something. but currently we need not.
        allpartcnt = 0;
        for (unsigned int pli = 0; pli < plc; pli++) {
            auto& pl = in->AccessParticles(pli);
            if (!isListOK(in, pli)) {
                continue;
            }

            //unsigned int vert_stride = 0;
            //if (pl.GetVertexDataType() == DirectionalParticleDataCall::Particles::VertexDataType::VERTDATA_FLOAT_XYZ) vert_stride = 12;
            //else if (pl.GetVertexDataType() == DirectionalParticleDataCall::Particles::VertexDataType::VERTDATA_FLOAT_XYZR) vert_stride = 16;
            //else continue;
            //vert_stride = std::max<unsigned int>(vert_stride, pl.GetVertexDataStride());
            //const unsigned char *vert = static_cast<const unsigned char*>(pl.GetVertexData());

            UINT64 part_cnt = pl.GetCount();

            for (int part_i = 0; part_i < part_cnt; ++part_i) {
                allParts.push_back(allpartcnt + part_i);
            }
            allpartcnt += pl.GetCount();
        }

        // allocate nanoflann data structures for border
        assert(allpartcnt == totalParts);
        this->myPts = std::make_shared<directionalPointcloud>(in, allParts);

        vislib::sys::Log::DefaultLog.WriteInfo("ParticleThermometer: building acceleration structure...");
        particleTree = std::make_shared<my_kd_tree_t>(3 /* dim */, *myPts, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
        particleTree->buildIndex();
        vislib::sys::Log::DefaultLog.WriteInfo("ParticleThermometer: done.");

        this->datahash = in->DataHash();
        this->lastTime = time;
        this->radiusSlot.ForceSetDirty();
    }

    if (this->radiusSlot.IsDirty() || this->cyclXSlot.IsDirty() || this->cyclYSlot.IsDirty() || this->cyclZSlot.IsDirty()
        || this->numNeighborSlot.IsDirty() || this->searchTypeSlot.IsDirty()) {
        size_t allpartcnt = 0;

        // final computation
        bool cycl_x = this->cyclXSlot.Param<megamol::core::param::BoolParam>()->Value();
        bool cycl_y = this->cyclYSlot.Param<megamol::core::param::BoolParam>()->Value();
        bool cycl_z = this->cyclZSlot.Param<megamol::core::param::BoolParam>()->Value();
        auto bbox = in->AccessBoundingBoxes().ObjectSpaceBBox();
        //bbox.EnforcePositiveSize(); // paranoia
        auto bbox_cntr = bbox.CalcCenter();

        vislib::sys::ConsoleProgressBar cpb;
        const int progressDivider = 100;
        cpb.Start("measuring temperature", static_cast<vislib::sys::ConsoleProgressBar::Size>(newColors.size() / progressDivider));

        float theMinTemp = FLT_MAX;
        float theMaxTemp = 0.0f;

        const bool remove_self = false;

        allpartcnt = 0;
        for (unsigned int pli = 0; pli < plc; pli++) {
            auto& pl = in->AccessParticles(pli);
            if (!isListOK(in, pli)) {
                continue;
            }

            int num_thr = omp_get_max_threads();
            INT64 counter = 0;

            std::vector<float> minTemp(num_thr, FLT_MAX);
            std::vector<float> maxTemp(num_thr, 0.0f);

#pragma omp parallel num_threads(num_thr)
            {
                float theVertex[3];
                float theTemperature[3];
                std::vector<std::pair<size_t, float> > ret_matches;
                std::vector<std::pair<size_t, float> > ret_localMatches;
                std::vector<size_t> ret_index(theNumber);
                std::vector<float> out_dist_sqr(theNumber);
                nanoflann::KNNResultSet<float> resultSet(theNumber);
                nanoflann::SearchParams params;
                params.sorted = false;
                ret_matches.reserve(100);
                ret_localMatches.reserve(100);
                int threadIdx = omp_get_thread_num();

                INT64 part_cnt = pl.GetCount();
#pragma omp for
                for (INT64 part_i = 0; part_i < part_cnt; ++part_i) {

                    INT64 myIndex = part_i + allpartcnt;
                    ret_matches.clear();
                    const float *vertexBase = this->myPts->get_position(myIndex);
                    const float *velocityBase = this->myPts->get_velocity(myIndex);

                    for (int x_s = 0; x_s < (cycl_x ? 2 : 1); ++x_s) {
                        for (int y_s = 0; y_s < (cycl_y ? 2 : 1); ++y_s) {
                            for (int z_s = 0; z_s < (cycl_z ? 2 : 1); ++z_s) {

                                theVertex[0] = vertexBase[0];
                                theVertex[1] = vertexBase[1];
                                theVertex[2] = vertexBase[2];
                                if (x_s > 0) theVertex[0] = theVertex[0] + ((theVertex[0] > bbox_cntr.X()) ? -bbox.Width() : bbox.Width());
                                if (y_s > 0) theVertex[1] = theVertex[1] + ((theVertex[1] > bbox_cntr.Y()) ? -bbox.Height() : bbox.Height());
                                if (z_s > 0) theVertex[2] = theVertex[2] + ((theVertex[2] > bbox_cntr.Z()) ? -bbox.Depth() : bbox.Depth());

                                if (theSearchType == searchTypeEnum::RADIUS) {
                                    particleTree->radiusSearch(theVertex, theRadius, ret_localMatches, params);
                                    if (remove_self) {
                                        ret_localMatches.erase(std::remove_if(ret_localMatches.begin(), ret_localMatches.end(),
                                            [&](decltype(ret_localMatches)::value_type &elem) {return elem.first == myIndex; }), ret_localMatches.end());
                                    }
                                    ret_matches.insert(ret_matches.end(), ret_localMatches.begin(), ret_localMatches.end());
                                } else {
                                    resultSet.init(ret_index.data(), out_dist_sqr.data());
                                    particleTree->findNeighbors(resultSet, theVertex, params);
                                    for (size_t i = 0; i < resultSet.size(); ++i) {
                                        if (!remove_self || ret_index[i] != myIndex) {
                                            ret_matches.push_back(std::pair<size_t, float>(ret_index[i], out_dist_sqr[i]));
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // no neighbor should count twice!
                    ret_matches.erase(unique(ret_matches.begin(), ret_matches.end()), ret_matches.end());

                    size_t num_matches = 0;
                    if (theSearchType == searchTypeEnum::RADIUS) {
                        maxDist = theRadius;
                        num_matches = ret_matches.size();
                    } else {
                        // find overall closest! we did search around periodic boundary conditions, so there will be huge distances!
                        sort(ret_matches.begin(), ret_matches.end(),
                            [](const decltype(ret_matches)::value_type &left, const decltype(ret_matches)::value_type &right) {return left.second < right.second; });
                        // the furthest is theNumber closest or the last one if fewer.
                        num_matches = ret_matches.size() >= theNumber ? theNumber : ret_matches.size();
                        maxDist = ret_matches[num_matches - 1].second;
                    }

                    std::vector<float> sum(3, 0);
                    std::vector<float> sqsum(3, 0);
                    for (size_t i = 0; i < num_matches; ++i) {
                        const float *velo = myPts->get_velocity(ret_matches[i].first);
                        for (int c = 0; c < 3; ++c) {
                            float v = velo[c];
                            sum[c] += v;
                            sqsum[c] += v * v;
                        }
                    }
                    for (int c = 0; c < 3; ++c) {
                        float vd = sum[c] / num_matches;
                        // this is ... I don't know.
                        theTemperature[c] = (theMass / 2) * (sqsum[c] - num_matches * vd * vd);
                        // this would be local velocity compared to velocity of surrounding (vd)
                        //theTemperature[c] = (theMass / 2) * (velocityBase[c] - vd) * (velocityBase[c] - vd);
                        this->newVelocities[myIndex * 3 + c] = vd;
                    }

                    // no square root, so actually kinetic energy
                    float tempMag = theTemperature[0] + theTemperature[1] + theTemperature[2];
                    //tempMag /= (num_matches * num_matches * 4.0f) / 9.0f;
                    tempMag /= num_matches * theFreedom;
                    //tempMag /= theFreedom;
                    newColors[myIndex] = tempMag;
                    if (tempMag < minTemp[threadIdx]) minTemp[threadIdx] = tempMag;
                    if (tempMag > maxTemp[threadIdx]) maxTemp[threadIdx] = tempMag;
#pragma omp atomic
                    ++counter;
                    if ((counter % progressDivider) == 0) cpb.Set(static_cast<vislib::sys::ConsoleProgressBar::Size>(counter / progressDivider));
                }
            } // end #pragma omp parallel num_threads(num_thr)
            for (auto i = 0; i < num_thr; ++i) {
                if (minTemp[i] < theMinTemp) theMinTemp = minTemp[i];
                if (maxTemp[i] > theMaxTemp) theMaxTemp = maxTemp[i];
            }
            allpartcnt += pl.GetCount();
        }
        cpb.Stop();

        this->minTempSlot.Param<core::param::FloatParam>()->SetValue(theMinTemp);
        this->maxTempSlot.Param<core::param::FloatParam>()->SetValue(theMaxTemp);
        vislib::sys::Log::DefaultLog.WriteInfo("Thermometer: min temp: %f max temp: %f", theMinTemp, theMaxTemp);

        this->radiusSlot.ResetDirty();
        this->cyclXSlot.ResetDirty();
        this->cyclYSlot.ResetDirty();
        this->cyclZSlot.ResetDirty();
        this->numNeighborSlot.ResetDirty();
        this->searchTypeSlot.ResetDirty();
    }

    // now the colors are known, inject them
    //in->SetUnlocker(nullptr, false);
    //in->Unlock();

    //vislib::sys::Log::DefaultLog.WriteInfo("ParticleThermometer: found temperatures between %f and %f", minTemp, maxTemp);

    allpartcnt = 0;
    if (outMPDC != nullptr) {
        outMPDC->SetParticleListCount(in->GetParticleListCount());
        for (unsigned int i = 0; i < in->GetParticleListCount(); ++i) {
            auto &pl = in->AccessParticles(i);
            if (!isListOK(in, i)) {
                outMPDC->AccessParticles(i).SetCount(0);
                continue;
            }
            outMPDC->AccessParticles(i).SetCount(pl.GetCount());
            outMPDC->AccessParticles(i).SetVertexData(pl.GetVertexDataType(), pl.GetVertexData(), pl.GetVertexDataStride());
            if (this->toggleNewColorSlot.Param < megamol::core::param::BoolParam>()->Value()) {
                outMPDC->AccessParticles(i).SetColourData(core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I,
                    this->newColors.data() + allpartcnt, 0);
                outMPDC->AccessParticles(i).SetColourMapIndexValues(this->minTempSlot.Param<core::param::FloatParam>()->Value(),
                    this->maxTempSlot.Param<core::param::FloatParam>()->Value());
            } else {
                outMPDC->AccessParticles(i).SetGlobalColour(255, 255, 255);
            }
            allpartcnt += pl.GetCount();
        }
    } else if (outDPDC != nullptr) {
        outDPDC->SetParticleListCount(in->GetParticleListCount());
        for (unsigned int i = 0; i < in->GetParticleListCount(); ++i) {
            auto &pl = in->AccessParticles(i);
            if (!isListOK(in, i)) {
                outDPDC->AccessParticles(i).SetCount(0);
                continue;
            }
            outDPDC->AccessParticles(i).SetCount(pl.GetCount());
            outDPDC->AccessParticles(i).SetVertexData(pl.GetVertexDataType(), pl.GetVertexData(), pl.GetVertexDataStride());
            if (this->toggleNewColorSlot.Param < megamol::core::param::BoolParam>()->Value()) {
                outDPDC->AccessParticles(i).SetColourData(core::moldyn::DirectionalParticleDataCall::Particles::COLDATA_FLOAT_I,
                    this->newColors.data() + allpartcnt, 0);
                outDPDC->AccessParticles(i).SetColourMapIndexValues(this->minTempSlot.Param<core::param::FloatParam>()->Value(),
                    this->maxTempSlot.Param<core::param::FloatParam>()->Value());
            } else {
                outDPDC->AccessParticles(i).SetGlobalColour(255, 255, 255);
            }
            if (this->toggleNewVelocitySlot.Param<megamol::core::param::BoolParam>()->Value()) {
                outDPDC->AccessParticles(i).SetDirData(megamol::core::moldyn::DirectionalParticles::DIRDATA_FLOAT_XYZ,
                    this->newVelocities.data() + allpartcnt * 3, 0);
            } else {
                outDPDC->AccessParticles(i).SetDirData(pl.GetDirDataType(), pl.GetDirData(), pl.GetDirDataStride());
            }
            allpartcnt += pl.GetCount();
        }
    }
    out->SetUnlocker(in->GetUnlocker());
    return true;
}


bool datatools::ParticleThermometer::getExtentCallback(megamol::core::Call& c) {
    using megamol::core::moldyn::MultiParticleDataCall;
    using megamol::core::moldyn::DirectionalParticleDataCall;

    DirectionalParticleDataCall *outDpdc = dynamic_cast<DirectionalParticleDataCall*>(&c);
    MultiParticleDataCall *outMpdc = dynamic_cast<MultiParticleDataCall*>(&c);
    if (outMpdc == nullptr && outDpdc == nullptr) return false;

    DirectionalParticleDataCall *inDpdc = this->inDataSlot.CallAs<DirectionalParticleDataCall>();
    if (inDpdc == nullptr) return false;

    megamol::core::AbstractGetData3DCall *out;
    if (outMpdc != nullptr) out = outMpdc;
    if (outDpdc != nullptr) out = outDpdc;

    inDpdc->SetFrameID(out->FrameID(), true);
    if (!(*inDpdc)(1)) {
        vislib::sys::Log::DefaultLog.WriteError("ParticleThermometer: could not get current frame extents (%u)", out->FrameID());
        return false;
    }
    out->AccessBoundingBoxes().SetObjectSpaceBBox(inDpdc->GetBoundingBoxes().ObjectSpaceBBox());
    out->AccessBoundingBoxes().SetObjectSpaceClipBox(inDpdc->GetBoundingBoxes().ObjectSpaceClipBox());
    if (inDpdc->FrameCount() < 1) {
        vislib::sys::Log::DefaultLog.WriteError("ParticleThermometer: no frame data!");
        return false;
    }
    out->SetFrameCount(inDpdc->FrameCount());
    // TODO: what am I actually doing here
    inDpdc->SetUnlocker(nullptr, false);
    inDpdc->Unlock();

    return true;
}

bool datatools::ParticleThermometer::getDataCallback(megamol::core::Call& c) {
    using megamol::core::moldyn::MultiParticleDataCall;
    using megamol::core::moldyn::DirectionalParticleDataCall;

    DirectionalParticleDataCall *outDpdc = dynamic_cast<DirectionalParticleDataCall*>(&c);
    MultiParticleDataCall *outMpdc = dynamic_cast<MultiParticleDataCall*>(&c);
    if (outMpdc == nullptr && outDpdc == nullptr) return false;

    DirectionalParticleDataCall *inDpdc = this->inDataSlot.CallAs<DirectionalParticleDataCall>();
    if (inDpdc == nullptr) return false;

    if (!this->assertData(inDpdc, outMpdc, outDpdc)) return false;

    //inMpdc->Unlock();

    return true;
}