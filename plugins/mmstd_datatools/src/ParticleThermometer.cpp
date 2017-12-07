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
#include "vislib/sys/Log.h"
#include "vislib/sys/ConsoleProgressBar.h"
#include <cstdint>
#include <algorithm>
#include <cfloat>
#include <cassert>
#include <limits>

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
        minTempSlot("minTemp", "the detected minimum temperature"),
        maxTempSlot("maxTemp", "the detected maximum temperature"),
        outDataSlot("outData", "Provides colors based on local particle temperature"),
        inDataSlot("inData", "Takes the directional particle data"),
        datahash(0), lastTime(-1), newColors(), allParts(), particleTree(nullptr), myPts(nullptr) {

    this->cyclXSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclXSlot);

    this->cyclYSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclYSlot);

    this->cyclZSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclZSlot);

    this->radiusSlot.SetParameter(new core::param::FloatParam(2.0));
    this->MakeSlotAvailable(&this->radiusSlot);

    this->minTempSlot.SetParameter(new core::param::FloatParam(0));
    this->MakeSlotAvailable(&this->minTempSlot);

    this->maxTempSlot.SetParameter(new core::param::FloatParam(0));
    this->MakeSlotAvailable(&this->maxTempSlot);

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

        this->newColors.resize(totalParts, theRadius);

        allParts.clear();
        allParts.reserve(totalParts);

        // we could now filter particles according to something. but currently we need not.
        allpartcnt = 0;
        for (unsigned int pli = 0; pli < plc; pli++) {
            auto& pl = in->AccessParticles(pli);
            if (!isListOK(in, pli)) {
                continue;
            }

            unsigned int vert_stride = 0;
            if (pl.GetVertexDataType() == DirectionalParticleDataCall::Particles::VertexDataType::VERTDATA_FLOAT_XYZ) vert_stride = 12;
            else if (pl.GetVertexDataType() == DirectionalParticleDataCall::Particles::VertexDataType::VERTDATA_FLOAT_XYZR) vert_stride = 16;
            else continue;
            vert_stride = std::max<unsigned int>(vert_stride, pl.GetVertexDataStride());
            const unsigned char *vert = static_cast<const unsigned char*>(pl.GetVertexData());

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

    if (this->radiusSlot.IsDirty() || this->cyclXSlot.IsDirty() || this->cyclYSlot.IsDirty() || this->cyclZSlot.IsDirty()) {
        size_t allpartcnt = 0;
        size_t cursor = 0;
        float theVertex[3];
        float theTemperature[3];

        std::vector<std::pair<size_t, float> > ret_matches;
        std::vector<std::pair<size_t, float> > ret_localMatches;
        nanoflann::SearchParams params;
        params.sorted = false;

        // final computation
        bool cycl_x = this->cyclXSlot.Param<megamol::core::param::BoolParam>()->Value();
        bool cycl_y = this->cyclYSlot.Param<megamol::core::param::BoolParam>()->Value();
        bool cycl_z = this->cyclZSlot.Param<megamol::core::param::BoolParam>()->Value();
        auto bbox = in->AccessBoundingBoxes().ObjectSpaceBBox();
        //bbox.EnforcePositiveSize(); // paranoia
        auto bbox_cntr = bbox.CalcCenter();

        ret_matches.reserve(100);

        vislib::sys::ConsoleProgressBar cpb;
        const int progressDivider = 100;
        cpb.Start("measuring temperature", static_cast<vislib::sys::ConsoleProgressBar::Size>(newColors.size() / progressDivider));

        float minTemp = FLT_MAX;
        float maxTemp = 0;

        allpartcnt = 0;
        for (unsigned int pli = 0; pli < plc; pli++) {
            auto& pl = in->AccessParticles(pli);
            if (!isListOK(in, pli)) {
                continue;
            }

            size_t part_cnt = pl.GetCount();
            for (size_t part_i = 0; part_i < part_cnt; ++part_i) {

                size_t myIndex = part_i + allpartcnt;
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

                            particleTree->radiusSearch(theVertex, theRadius, ret_localMatches, params);
                            ret_localMatches.erase(std::remove_if(ret_localMatches.begin(), ret_localMatches.end(), 
                                [&](decltype(ret_localMatches)::value_type &elem) {return elem.first == myIndex; }), ret_localMatches.end());
                            ret_matches.insert(ret_matches.end(), ret_localMatches.begin(), ret_localMatches.end());
                        }
                    }
                }

                //sort(ret_matches.begin(), ret_matches.end());
                ret_matches.erase(unique(ret_matches.begin(), ret_matches.end()), ret_matches.end());

                int n = 1;
                float averageX = 0;
                float averageY = 0;
                float averageZ = 0;
                for (auto &m : ret_matches) {
                    const float *velo = myPts->get_velocity(m.first);
                    averageX += (velo[0] - averageX) / n;
                    averageY += (velo[1] - averageY) / n;
                    averageZ += (velo[2] - averageZ) / n;
                    ++n;
                }
                theTemperature[0] = averageX - velocityBase[0];
                theTemperature[1] = averageY - velocityBase[1];
                theTemperature[2] = averageZ - velocityBase[2];
                float tempMag = sqrtf(theTemperature[0] * theTemperature[0] + theTemperature[1] * theTemperature[1] + theTemperature[2] * theTemperature[2]);
                newColors[myIndex] = tempMag;
                if (tempMag < minTemp) minTemp = tempMag;
                if (tempMag > maxTemp) maxTemp = tempMag;
                if ((myIndex % progressDivider) == 0) cpb.Set(static_cast<vislib::sys::ConsoleProgressBar::Size>(myIndex / progressDivider));
            }
            allpartcnt += pl.GetCount();
        }
        cpb.Stop();

        this->minTempSlot.Param<core::param::FloatParam>()->SetValue(minTemp);
        this->maxTempSlot.Param<core::param::FloatParam>()->SetValue(maxTemp);

        this->radiusSlot.ResetDirty();
        this->cyclXSlot.ResetDirty();
        this->cyclYSlot.ResetDirty();
        this->cyclZSlot.ResetDirty();
    }

    // now the colors are known, inject them
    in->SetUnlocker(nullptr, false);
    in->Unlock();

#pragma region oldstuff
        //// final computation
        //bool cycl_x = this->cyclXSlot.Param<megamol::core::param::BoolParam>()->Value();
        //bool cycl_y = this->cyclYSlot.Param<megamol::core::param::BoolParam>()->Value();
        //bool cycl_z = this->cyclZSlot.Param<megamol::core::param::BoolParam>()->Value();
        //auto bbox = dat.AccessBoundingBoxes().ObjectSpaceBBox();
        //bbox.EnforcePositiveSize(); // paranoia
        //auto bbox_cntr = bbox.CalcCenter();

        //allpartcnt = 0;
        //for (unsigned int pli = 0; pli < plc; pli++) {
        //    auto& pl = dat.AccessParticles(pli);
        //    if (pl.GetColourDataType() != SimpleSphericalParticles::COLDATA_FLOAT_I) continue;
        //    if ((pl.GetVertexDataType() != SimpleSphericalParticles::VERTDATA_FLOAT_XYZ)
        //        && (pl.GetVertexDataType() != SimpleSphericalParticles::VERTDATA_FLOAT_XYZR)) {
        //        continue;
        //    }

        //    unsigned int vert_stride = 0;
        //    if (pl.GetVertexDataType() == SimpleSphericalParticles::VERTDATA_FLOAT_XYZ) vert_stride = 12;
        //    else if (pl.GetVertexDataType() == SimpleSphericalParticles::VERTDATA_FLOAT_XYZR) vert_stride = 16;
        //    else continue;
        //    vert_stride = std::max<unsigned int>(vert_stride, pl.GetVertexDataStride());
        //    const unsigned char *vert = static_cast<const unsigned char*>(pl.GetVertexData());

        //    int part_cnt = static_cast<int>(pl.GetCount());
        //    const unsigned char *col = static_cast<const unsigned char*>(pl.GetColourData());
        //    unsigned int col_stride = std::max<unsigned int>(pl.GetColourDataStride(), sizeof(float));

        //    for (int part_i = 0; part_i < part_cnt; ++part_i) {
        //        float c = *reinterpret_cast<const float *>(col + (part_i * col_stride));
        //        const float *v = reinterpret_cast<const float *>(vert + (part_i * vert_stride));

        //        if ((-border_epsilon < c) && (c < border_epsilon)) {
        //            c = 0.0f;
        //        } else {
        //            float q[3];
        //            float dist, distsq = static_cast<float>(DBL_MAX);
        //            my_kd_tree_t& tree = (c < 0.0f) ? posTree : negTree;

        //            for (int x_s = 0; x_s < (cycl_x ? 2 : 1); ++x_s) {
        //                for (int y_s = 0; y_s < (cycl_y ? 2 : 1); ++y_s) {
        //                    for (int z_s = 0; z_s < (cycl_z ? 2 : 1); ++z_s) {

        //                        q[0] = v[0];
        //                        q[1] = v[1];
        //                        q[2] = v[2];
        //                        if (x_s > 0) q[0] = v[0] + ((v[0] > bbox_cntr.X()) ? -bbox.Width() : bbox.Width());
        //                        if (y_s > 0) q[1] = v[1] + ((v[1] > bbox_cntr.Y()) ? -bbox.Height() : bbox.Height());
        //                        if (z_s > 0) q[2] = v[2] + ((v[2] > bbox_cntr.Z()) ? -bbox.Depth() : bbox.Depth());

        //                        size_t n_idx;
        //                        float n_distsq;
        //                        tree.knnSearch(q, 1, &n_idx, &n_distsq);
        //                        if (n_distsq < distsq) distsq = n_distsq;

        //                    }
        //                }
        //            }

        //            dist = sqrt(distsq);
        //            if (c < 0.0f) dist = -dist;
        //            c = static_cast<float>(dist);
        //        }

        //        if (c < this->minCol) this->minCol = c;
        //        if (c > this->maxCol) this->maxCol = c;

        //        this->newColors[allpartcnt + part_i] = c;
        //    }

        //    allpartcnt += static_cast<size_t>(part_cnt);
        //}
#pragma endregion

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
            outMPDC->AccessParticles(i).SetColourData(core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I, 
                this->newColors.data() + allpartcnt, 0);
            outMPDC->AccessParticles(i).SetColourMapIndexValues(this->minTempSlot.Param<core::param::FloatParam>()->Value(),
                this->maxTempSlot.Param<core::param::FloatParam>()->Value());
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
            outDPDC->AccessParticles(i).SetColourMapIndexValues(this->minTempSlot.Param<core::param::FloatParam>()->Value(),
                this->maxTempSlot.Param<core::param::FloatParam>()->Value());
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