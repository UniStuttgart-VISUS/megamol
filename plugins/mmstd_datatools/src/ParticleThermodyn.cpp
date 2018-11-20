/*
 * ParticleThermodyn.cpp
 *
 * Copyright (C) 2017 by MegaMol team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "ParticleThermodyn.h"
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
#include <array>

using namespace megamol;
using namespace megamol::stdplugin;

/*
 * datatools::ParticleThermodyn::ParticleThermodyn
 */
datatools::ParticleThermodyn::ParticleThermodyn(void)
        : cyclXSlot("cyclX", "Considers cyclic boundary conditions in X direction"),
        cyclYSlot("cyclY", "Considers cyclic boundary conditions in Y direction"),
        cyclZSlot("cyclZ", "Considers cyclic boundary conditions in Z direction"),
        radiusSlot("radius", "the radius in which to look for neighbors"),
        numNeighborSlot("numNeighbors", "how many neighbors to collect"),
        searchTypeSlot("searchType", "num of neighbors or radius"),
        minMetricSlot("minMetric", "the detected minimum of the selected metric"),
        maxMetricSlot("maxMetric", "the detected maximum of the selected metric"),
        massSlot("mass", "the mass of the particles"),
        freedomSlot("freedomFactor", "factor reducing T* based on degrees of freedom of the molecular model"),
        metricsSlot("metrics", "the metrics you want to computer from the neighborhood"),
        datahash(0),
        lastTime(-1),
        newColors(),
        allParts(), maxDist(0.0f), particleTree(nullptr), myPts(nullptr),
        outDataSlot("outData", "Provides intensities based on a local particle metric"),
        inDataSlot("inData", "Takes the directional particle data") {

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

    core::param::EnumParam *mt = new core::param::EnumParam(metricsEnum::TEMPERATURE);
    mt->SetTypePair(metricsEnum::TEMPERATURE, "Temperature");
    mt->SetTypePair(metricsEnum::DENSITY, "Density");
    mt->SetTypePair(metricsEnum::FRACTIONAL_ANISOTROPY, "Fractional Anisotropy");
    mt->SetTypePair(metricsEnum::PRESSURE, "Pressure");
    this->metricsSlot << mt;
    this->MakeSlotAvailable(&this->metricsSlot);

    this->minMetricSlot.SetParameter(new core::param::FloatParam(0));
    this->MakeSlotAvailable(&this->minMetricSlot);

    this->maxMetricSlot.SetParameter(new core::param::FloatParam(0));
    this->MakeSlotAvailable(&this->maxMetricSlot);

    this->massSlot.SetParameter(new core::param::FloatParam(1.0f));
    this->MakeSlotAvailable(&this->massSlot);

    this->freedomSlot.SetParameter(new core::param::FloatParam(1.5f)); // works for single-center models. 3 degrees of freedom -> 3/2
    this->MakeSlotAvailable(&this->freedomSlot);

    this->outDataSlot.SetCallback(megamol::core::moldyn::DirectionalParticleDataCall::ClassName(), "GetData", &ParticleThermodyn::getDataCallback);
    this->outDataSlot.SetCallback(megamol::core::moldyn::DirectionalParticleDataCall::ClassName(), "GetExtent", &ParticleThermodyn::getExtentCallback);
    this->outDataSlot.SetCallback(megamol::core::moldyn::MultiParticleDataCall::ClassName(), "GetData", &ParticleThermodyn::getDataCallback);
    this->outDataSlot.SetCallback(megamol::core::moldyn::MultiParticleDataCall::ClassName(), "GetExtent", &ParticleThermodyn::getExtentCallback);
    this->MakeSlotAvailable(&this->outDataSlot);

    this->inDataSlot.SetCompatibleCall<megamol::core::moldyn::DirectionalParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);
}


/*
 * datatools::ParticleColorSignedDistance::~ParticleColorSignedDistance
 */
datatools::ParticleThermodyn::~ParticleThermodyn(void) {
    this->Release();
}

/*
* datatools::ParticleThermodyn::create
*/
bool datatools::ParticleThermodyn::create(void) {
    return true;
}


bool isListOK(megamol::core::moldyn::DirectionalParticleDataCall *in, const unsigned int i) {
    using megamol::core::moldyn::DirectionalParticleDataCall;
    auto& pl = in->AccessParticles(i);
    return ((pl.GetVertexDataType() == DirectionalParticleDataCall::Particles::VertexDataType::VERTDATA_FLOAT_XYZ)
        || (pl.GetVertexDataType() == DirectionalParticleDataCall::Particles::VertexDataType::VERTDATA_FLOAT_XYZR))
    && pl.GetDirDataType() == DirectionalParticleDataCall::Particles::DirDataType::DIRDATA_FLOAT_XYZ;
}


/*
* datatools::ParticleThermodyn::release
*/
void datatools::ParticleThermodyn::release(void) {
}


bool datatools::ParticleThermodyn::assertData(core::moldyn::DirectionalParticleDataCall *in,
    core::moldyn::MultiParticleDataCall *outMPDC, core::moldyn::DirectionalParticleDataCall *outDPDC) {

    using megamol::core::moldyn::DirectionalParticleDataCall;
    using megamol::core::moldyn::MultiParticleDataCall;

    megamol::core::AbstractGetData3DCall *out;
    if (outMPDC != nullptr) out = outMPDC;
    if (outDPDC != nullptr) out = outDPDC;

    const unsigned int time = out->FrameID();
    unsigned int plc = in->GetParticleListCount();
    float theRadius = this->radiusSlot.Param<core::param::FloatParam>()->Value();
    theRadius = theRadius * theRadius;
    const float theMass = this->massSlot.Param<core::param::FloatParam>()->Value();
    const float theFreedom = this->freedomSlot.Param<core::param::FloatParam>()->Value();
    const int theNumber = this->numNeighborSlot.Param<core::param::IntParam>()->Value();
    const auto theSearchType = this->searchTypeSlot.Param<core::param::EnumParam>()->Value();
    const auto theMetrics = this->metricsSlot.Param<core::param::EnumParam>()->Value();
    size_t allpartcnt = 0;

    if (this->lastTime != time || this->datahash != in->DataHash()) {
        in->SetFrameID(time, true);

        if (!(*in)(0)) {
            vislib::sys::Log::DefaultLog.WriteError("ParticleThermodyn: could not get frame (%u)", time);
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

        allParts.clear();
        allParts.reserve(totalParts);

        // we could now filter particles according to something. but currently we need not.
        allpartcnt = 0;
        for (unsigned int pli = 0; pli < plc; pli++) {
            auto& pl = in->AccessParticles(pli);
            if (!isListOK(in, pli)) {
                vislib::sys::Log::DefaultLog.WriteWarn("ParticleThermodyn: ignoring list %d because it either has no proper positions or no velocity", pli);
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

        vislib::sys::Log::DefaultLog.WriteInfo("ParticleThermodyn: building acceleration structure...");
        particleTree = std::make_shared<my_kd_tree_t>(3 /* dim */, *myPts, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
        particleTree->buildIndex();
        vislib::sys::Log::DefaultLog.WriteInfo("ParticleThermodyn: done.");

        this->datahash = in->DataHash();
        this->lastTime = time;
        this->radiusSlot.ForceSetDirty();
    }

    if (this->radiusSlot.IsDirty() || this->cyclXSlot.IsDirty() || this->cyclYSlot.IsDirty() || this->cyclZSlot.IsDirty()
        || this->numNeighborSlot.IsDirty() || this->searchTypeSlot.IsDirty() || this->metricsSlot.IsDirty()) {
        allpartcnt = 0;

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

            std::vector<float> metricMin(num_thr, FLT_MAX);
            std::vector<float> metricMax(num_thr, 0.0f);

#pragma omp parallel num_threads(num_thr)
//#pragma omp parallel num_threads(1)
            {
                float theVertex[3];
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
                    //const float *velocityBase = this->myPts->get_velocity(myIndex);

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

                    float magnitude = 0.0f;

                    switch (theMetrics) {
                    case metricsEnum::TEMPERATURE:
                        magnitude = computeTemperature(ret_matches, num_matches, theMass, theFreedom);
                        break;
                    case metricsEnum::DENSITY:
                        vislib::sys::Log::DefaultLog.WriteWarn("ParticleThermodyn: cannot compute density yet!");
                        break;
                    case metricsEnum::FRACTIONAL_ANISOTROPY:
                        magnitude = computeFractionalAnisotropy(ret_matches, num_matches);
                        break;
                    case metricsEnum::PRESSURE:
                        vislib::sys::Log::DefaultLog.WriteWarn("ParticleThermodyn: cannot compute pressure yet!");
                        break;
                    default:
                        vislib::sys::Log::DefaultLog.WriteError("ParticleThermodyn: unknown metric");
                        break;
                    }
                    newColors[myIndex] = magnitude;

                    if (magnitude < metricMin[threadIdx]) metricMin[threadIdx] = magnitude;
                    if (magnitude > metricMax[threadIdx]) metricMax[threadIdx] = magnitude;
#pragma omp atomic
                    ++counter;
                    if ((counter % progressDivider) == 0) cpb.Set(static_cast<vislib::sys::ConsoleProgressBar::Size>(counter / progressDivider));
                }
            } // end #pragma omp parallel num_threads(num_thr)
            for (auto i = 0; i < num_thr; ++i) {
                if (metricMin[i] < theMinTemp) theMinTemp = metricMin[i];
                if (metricMax[i] > theMaxTemp) theMaxTemp = metricMax[i];
            }
            allpartcnt += pl.GetCount();
        }
        cpb.Stop();

        this->minMetricSlot.Param<core::param::FloatParam>()->SetValue(theMinTemp);
        this->maxMetricSlot.Param<core::param::FloatParam>()->SetValue(theMaxTemp);
        vislib::sys::Log::DefaultLog.WriteInfo("ParticleThermodyn: min metric: %f max metric: %f", theMinTemp, theMaxTemp);

        this->radiusSlot.ResetDirty();
        this->cyclXSlot.ResetDirty();
        this->cyclYSlot.ResetDirty();
        this->cyclZSlot.ResetDirty();
        this->numNeighborSlot.ResetDirty();
        this->searchTypeSlot.ResetDirty();
        this->metricsSlot.ResetDirty();
    }

    // now the colors are known, inject them
    //in->SetUnlocker(nullptr, false);
    //in->Unlock();

    //vislib::sys::Log::DefaultLog.WriteInfo("ParticleThermodyn: found temperatures between %f and %f", minTemp, maxTemp);

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
            outMPDC->AccessParticles(i).SetColourData(core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I, 
                this->newColors.data() + allpartcnt, 0);
            outMPDC->AccessParticles(i).SetColourMapIndexValues(this->minMetricSlot.Param<core::param::FloatParam>()->Value(),
                this->maxMetricSlot.Param<core::param::FloatParam>()->Value());
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
            outDPDC->AccessParticles(i).SetColourData(core::moldyn::DirectionalParticleDataCall::Particles::COLDATA_FLOAT_I,
                this->newColors.data() + allpartcnt, 0);
            outDPDC->AccessParticles(i).SetDirData(pl.GetDirDataType(), pl.GetDirData(), pl.GetDirDataStride());
            outDPDC->AccessParticles(i).SetColourMapIndexValues(this->minMetricSlot.Param<core::param::FloatParam>()->Value(),
                this->maxMetricSlot.Param<core::param::FloatParam>()->Value());
            allpartcnt += pl.GetCount();
        }
    }
    out->SetUnlocker(in->GetUnlocker());
    in->SetUnlocker(nullptr, false);
    return true;
}

float megamol::stdplugin::datatools::ParticleThermodyn::computeTemperature(
    std::vector<std::pair<size_t, float>>& matches, const size_t num_matches, const float mass, const float freedom) {
    std::array<float, 3> sum = {0, 0, 0};
    std::array<float, 3> sq_sum = {0, 0, 0};
    std::array<float, 3> the_temperature = {0, 0, 0};
    for (size_t i = 0; i < num_matches; ++i) {
        const float* velo = myPts->get_velocity(matches[i].first);
        for (int c = 0; c < 3; ++c) {
            float v = velo[c];
            sum[c] += v;
            sq_sum[c] += v * v;
        }
    }
    for (int c = 0; c < 3; ++c) {
        float vd = sum[c] / num_matches;
        the_temperature[c] = (mass / 2) * (sq_sum[c] - num_matches * vd * vd);
        // this would be local velocity compared to velocity of surrounding (vd)
        // theTemperature[c] = (theMass / 2) * (velocityBase[c] - vd) * (velocityBase[c] - vd);
    }

    // no square root, so actually kinetic energy
    float magnitude = the_temperature[0] + the_temperature[1] + the_temperature[2];
    // tempMag /= (num_matches * num_matches * 4.0f) / 9.0f;
    magnitude /= num_matches * freedom;
    return magnitude;
}

float megamol::stdplugin::datatools::ParticleThermodyn::computeFractionalAnisotropy(
    std::vector<std::pair<size_t, float>>& matches, const size_t num_matches) {

    Eigen::Matrix3f mat;
    mat.fill(0.0f);

    for (size_t i = 0; i < num_matches; ++i) {
        const float* velo = myPts->get_velocity(matches[i].first);
        for (int x = 0; x < 3; ++x) 
            for (int y = 0; y < 3; ++y)
                mat(x,y) += velo[x] * velo[y];
    }
    mat /= static_cast<float>(num_matches);

    eigensolver.computeDirect(mat, Eigen::EigenvaluesOnly);
    auto& ev = eigensolver.eigenvalues();
    float evMean = ev[0] + ev[1] + ev[2];
    evMean /= 3.0f;

    float FA = sqrt((ev[0] - evMean) * (ev[0] - evMean) + (ev[1] - evMean) * (ev[1] - evMean) +
                    (ev[2] - evMean) * (ev[2] - evMean));
    float scale = sqrt(1.5f) / sqrt(ev[0] * ev[0] + ev[1] * ev[1] + ev[2] * ev[2]);

    return FA * scale;
}


bool datatools::ParticleThermodyn::getExtentCallback(megamol::core::Call& c) {
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
        vislib::sys::Log::DefaultLog.WriteError("ParticleThermodyn: could not get current frame extents (%u)", out->FrameID());
        return false;
    }
    out->AccessBoundingBoxes().SetObjectSpaceBBox(inDpdc->GetBoundingBoxes().ObjectSpaceBBox());
    out->AccessBoundingBoxes().SetObjectSpaceClipBox(inDpdc->GetBoundingBoxes().ObjectSpaceClipBox());
    if (inDpdc->FrameCount() < 1) {
        vislib::sys::Log::DefaultLog.WriteError("ParticleThermodyn: no frame data!");
        return false;
    }
    out->SetFrameCount(inDpdc->FrameCount());
    // TODO: what am I actually doing here
    inDpdc->SetUnlocker(nullptr, false);
    inDpdc->Unlock();

    return true;
}

bool datatools::ParticleThermodyn::getDataCallback(megamol::core::Call& c) {
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