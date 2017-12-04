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
#include <cstdint>
#include <algorithm>
#include <cfloat>
#include <cassert>

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
        outDataSlot("outData", "Provides colors based on local particle temperature"),
        inDataSlot("inData", "Takes the directional particle data"),
        datahash(0), lastTime(-1), newColors(), minCol(0.0f), maxCol(1.0f), radius(-1.0),
        allParts(), particleTree(nullptr), myPts(nullptr) {

    this->cyclXSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclXSlot);

    this->cyclYSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclYSlot);

    this->cyclZSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclZSlot);

    this->radiusSlot.SetParameter(new core::param::FloatParam(2.0));
    this->MakeSlotAvailable(&this->radiusSlot);

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

    if (this->lastTime != time || this->datahash != in->DataHash()) {
        // load previous Frame
        in->SetFrameID(time, true);

        if (!(*in)(0)) {
            vislib::sys::Log::DefaultLog.WriteError("ParticleThermometer: could not get frame (%u)", time);
            return false;
        }

        size_t totalParts = 0;
        size_t plc = in->GetParticleListCount();
        for (unsigned int i = 0; i < plc; i++) {
            if (in->AccessParticles(i).GetVertexDataType() == DirectionalParticleDataCall::Particles::VertexDataType::VERTDATA_FLOAT_XYZ
                || in->AccessParticles(i).GetVertexDataType() == DirectionalParticleDataCall::Particles::VertexDataType::VERTDATA_FLOAT_XYZR)
                totalParts += in->AccessParticles(i).GetCount();
        }

        this->newColors.resize(totalParts);

        allParts.reserve(totalParts);

        // we could now filter particles according to something. but currently we need not.
        size_t allpartcnt = 0;
        for (unsigned int pli = 0; pli < plc; pli++) {
            auto& pl = in->AccessParticles(pli);
            if ((pl.GetVertexDataType() != DirectionalParticleDataCall::Particles::VERTDATA_FLOAT_XYZ)
                && (pl.GetVertexDataType() != DirectionalParticleDataCall::Particles::VERTDATA_FLOAT_XYZR)) {
                continue;
            }

            unsigned int vert_stride = 0;
            if (pl.GetVertexDataType() == DirectionalParticleDataCall::Particles::VertexDataType::VERTDATA_FLOAT_XYZ) vert_stride = 12;
            else if (pl.GetVertexDataType() == DirectionalParticleDataCall::Particles::VertexDataType::VERTDATA_FLOAT_XYZR) vert_stride = 16;
            else continue;
            vert_stride = std::max<unsigned int>(vert_stride, pl.GetVertexDataStride());
            const unsigned char *vert = static_cast<const unsigned char*>(pl.GetVertexData());

            int part_cnt = static_cast<int>(pl.GetCount());

            for (int part_i = 0; part_i < part_cnt; ++part_i) {
                const float *v = reinterpret_cast<const float *>(vert + (part_i * vert_stride));
                allParts.push_back(allpartcnt + part_i);
            }
            allpartcnt += static_cast<size_t>(pl.GetCount());
        }

        // allocate nanoflann data structures for border
        assert(allpartcnt == totalParts);
        // TODO: can I really keep the Kd-Tree when cyclicness is changed? I'd say why not. But who knows the implementation!
        this->myPts = std::make_shared<directionalPointcloud>(in, allParts,
            this->cyclXSlot.Param<core::param::BoolParam>()->Value(), this->cyclYSlot.Param<core::param::BoolParam>()->Value(), this->cyclZSlot.Param<core::param::BoolParam>()->Value());

        particleTree = std::make_shared<my_kd_tree_t>(3 /* dim */, *myPts, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
        particleTree->buildIndex();
    }
    float theRadius = this->radiusSlot.Param<core::param::FloatParam>()->Value();
    if (radius != theRadius) {
        size_t allpartcnt = 0;
        size_t cursor = 0;
        float v[3];
        float v2[3];
        float vel[3];
        float vel2[3];
        size_t plc = in->GetParticleListCount();
        std::vector<std::pair<size_t, float> > ret_matches;
        nanoflann::SearchParams params;
        params.sorted = false;
        // TODO: can I really keep the Kd-Tree when cyclicness is changed? I'd say why not. But who knows the implementation! see above.
        this->myPts->SetCyclicBoundary(this->cyclXSlot.Param<core::param::BoolParam>()->Value(),
            this->cyclYSlot.Param<core::param::BoolParam>()->Value(), this->cyclZSlot.Param<core::param::BoolParam>()->Value());

        for (unsigned int pli = 0; pli < plc; pli++) {
            auto& pl = in->AccessParticles(pli);
            if ((pl.GetVertexDataType() != DirectionalParticleDataCall::Particles::VERTDATA_FLOAT_XYZ)
                && (pl.GetVertexDataType() != DirectionalParticleDataCall::Particles::VERTDATA_FLOAT_XYZR)) {
                continue;
            }
            unsigned int vert_stride = 0;
            if (pl.GetVertexDataType() == DirectionalParticleDataCall::Particles::VERTDATA_FLOAT_XYZ) vert_stride = 12;
            else if (pl.GetVertexDataType() == DirectionalParticleDataCall::Particles::VERTDATA_FLOAT_XYZR) vert_stride = 16;
            else continue;
            vert_stride = std::max<unsigned int>(vert_stride, pl.GetVertexDataStride());
            const unsigned char *vert = static_cast<const unsigned char*>(pl.GetVertexData());

            size_t part_cnt = pl.GetCount();
            for (size_t part_i = 0; part_i < part_cnt; ++part_i) {
                // did we consider this particle?
                while (allParts[cursor] < allpartcnt + part_i) cursor++;
                if (allParts[cursor] == allpartcnt + part_i) {
                    const float *v = reinterpret_cast<const float *>(vert + (part_i * vert_stride));
                    const float *v2 = myPts->get_position(allpartcnt + part_i);

                    particleTree->radiusSearch(v, theRadius, ret_matches, params);
                    // does the query return the point itself? YES!
                    for (int i = 0; i < ret_matches.size(); ++i) {
                        if (ret_matches[i].first == allpartcnt + part_i) {
                            // this is me
                        } else {
                            // these are others
                            // add to the running average of velocities
                        }
                    }
                } else {
                    newColors[allpartcnt + part_i] = 0.0f;
                }
            }
            allpartcnt += part_cnt;
        }


        radius = theRadius;
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

    if (outMPDC != nullptr) {
        outMPDC->SetParticleListCount(in->GetParticleListCount());
        for (unsigned int i = 0; i < in->GetParticleListCount(); ++i) {
            auto &pl = in->AccessParticles(i);
            outMPDC->AccessParticles(i).SetCount(pl.GetCount());
            outMPDC->AccessParticles(i).SetVertexData(pl.GetVertexDataType(), pl.GetVertexData(), pl.GetVertexDataStride());
            outMPDC->AccessParticles(i).SetColourData(core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I, 
                this->newColors.data(), 0);
        }
    } else if (outDPDC != nullptr) {
        outDPDC->SetParticleListCount(in->GetParticleListCount());
        for (unsigned int i = 0; i < in->GetParticleListCount(); ++i) {
            auto &pl = in->AccessParticles(i);
            outDPDC->AccessParticles(i).SetCount(pl.GetCount());
            outDPDC->AccessParticles(i).SetVertexData(pl.GetVertexDataType(), pl.GetVertexData(), pl.GetVertexDataStride());
            outDPDC->AccessParticles(i).SetColourData(core::moldyn::DirectionalParticleDataCall::Particles::COLDATA_FLOAT_I,
                this->newColors.data(), 0);
            outDPDC->AccessParticles(i).SetDirData(pl.GetDirDataType(), pl.GetDirData(), pl.GetDirDataStride());
        }
    }
    this->datahash = in->DataHash();
    this->lastTime = time;
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

    //if (!this->assertData(inMpdc, outDpdc)) return false;
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