/*
 * ParticleNeighborhood.cpp
 *
 * Copyright (C) 2017 by MegaMol team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "ParticleNeighborhood.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/EnumParam.h"
#include "vislib/sys/Log.h"
#include <cstdint>
#include <algorithm>
#include <cfloat>
#include <cassert>

using namespace megamol;
using namespace megamol::stdplugin;

/*
 * datatools::ParticleNeighborhood::ParticleNeighborhood
 */
datatools::ParticleNeighborhood::ParticleNeighborhood(void)
        : cyclXSlot("cyclX", "Considers cyclic boundary conditions in X direction"),
        cyclYSlot("cyclY", "Considers cyclic boundary conditions in Y direction"),
        cyclZSlot("cyclZ", "Considers cyclic boundary conditions in Z direction"),
        radiusSlot("radius", "the radius in which to look for neighbors"),
        numNeighborSlot("numNeighbors", "how many neighbors to collect"),
        searchTypeSlot("searchType", "num of neighbors or radius"),
        particleNumberSlot("idx", "the particle to track"),
        outDataSlot("outData", "Provides colors based on local particle temperature"),
        inDataSlot("inData", "Takes the directional particle data"),
        datahash(0), lastTime(-1), newColors(), maxDist(0),
        allParts(), particleTree(nullptr), dirParticleTree(nullptr), myPts(nullptr), myDirPts(nullptr) {

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

    this->particleNumberSlot.SetParameter(new core::param::IntParam(-1));
    this->MakeSlotAvailable(&this->particleNumberSlot);

    this->outDataSlot.SetCallback(megamol::core::moldyn::DirectionalParticleDataCall::ClassName(), "GetData", &ParticleNeighborhood::getDataCallback);
    this->outDataSlot.SetCallback(megamol::core::moldyn::DirectionalParticleDataCall::ClassName(), "GetExtent", &ParticleNeighborhood::getExtentCallback);
    this->outDataSlot.SetCallback(megamol::core::moldyn::MultiParticleDataCall::ClassName(), "GetData", &ParticleNeighborhood::getDataCallback);
    this->outDataSlot.SetCallback(megamol::core::moldyn::MultiParticleDataCall::ClassName(), "GetExtent", &ParticleNeighborhood::getExtentCallback);
    this->MakeSlotAvailable(&this->outDataSlot);

    this->inDataSlot.SetCompatibleCall<megamol::core::moldyn::DirectionalParticleDataCallDescription>();
    this->inDataSlot.SetCompatibleCall<megamol::core::moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);
}


/*
 * datatools::ParticleNeighborhood::~ParticleNeighborhood
 */
datatools::ParticleNeighborhood::~ParticleNeighborhood(void) {
    this->Release();
}

/*
* datatools::ParticleNeighborhood::create
*/
bool datatools::ParticleNeighborhood::create(void) {
    return true;
}


/*
* datatools::ParticleNeighborhood::release
*/
void datatools::ParticleNeighborhood::release(void) {
}

bool isListOK(megamol::core::AbstractGetData3DCall *c, unsigned int i) {
    using megamol::core::moldyn::DirectionalParticleDataCall;
    using megamol::core::moldyn::MultiParticleDataCall;

    DirectionalParticleDataCall *dpdc = dynamic_cast<DirectionalParticleDataCall*>(c);
    MultiParticleDataCall *mpdc = dynamic_cast<MultiParticleDataCall*>(c);
    if (dpdc != nullptr) {
        auto& pl = dpdc->AccessParticles(i);
        return pl.GetVertexDataType() == DirectionalParticleDataCall::Particles::VertexDataType::VERTDATA_FLOAT_XYZ
            || DirectionalParticleDataCall::Particles::VertexDataType::VERTDATA_FLOAT_XYZR;
    }
    if (mpdc != nullptr) {
        auto& pl = mpdc->AccessParticles(i);
        return pl.GetVertexDataType() == MultiParticleDataCall::Particles::VertexDataType::VERTDATA_FLOAT_XYZ
            || MultiParticleDataCall::Particles::VertexDataType::VERTDATA_FLOAT_XYZR;
    }
    return false;
}

UINT64 getListCount(megamol::core::AbstractGetData3DCall *c, unsigned int i) {
    using megamol::core::moldyn::DirectionalParticleDataCall;
    using megamol::core::moldyn::MultiParticleDataCall;

    DirectionalParticleDataCall *dpdc = dynamic_cast<DirectionalParticleDataCall*>(c);
    MultiParticleDataCall *mpdc = dynamic_cast<MultiParticleDataCall*>(c);
    if (dpdc != nullptr) {
        return dpdc->AccessParticles(i).GetCount();
    }
    if (mpdc != nullptr) {
        return mpdc->AccessParticles(i).GetCount();
    }
    return 0;
}

bool datatools::ParticleNeighborhood::assertData(megamol::core::AbstractGetData3DCall *in,
        megamol::core::AbstractGetData3DCall *out) {

    using megamol::core::moldyn::DirectionalParticleDataCall;
    using megamol::core::moldyn::MultiParticleDataCall;

    DirectionalParticleDataCall *outDpdc = dynamic_cast<DirectionalParticleDataCall*>(out);
    MultiParticleDataCall *outMpdc = dynamic_cast<MultiParticleDataCall*>(out);
    DirectionalParticleDataCall *inDpdc = dynamic_cast<DirectionalParticleDataCall*>(in);
    MultiParticleDataCall *inMpdc = dynamic_cast<MultiParticleDataCall*>(in);

    unsigned int time = out->FrameID();

    unsigned int plc = (inMpdc == nullptr) ? inDpdc->GetParticleListCount() : inMpdc->GetParticleListCount();

    float theRadius = this->radiusSlot.Param<core::param::FloatParam>()->Value();
    theRadius = theRadius * theRadius;
    int theNumber = this->numNeighborSlot.Param<core::param::IntParam>()->Value();
    auto theSearchType = this->searchTypeSlot.Param<core::param::EnumParam>()->Value();
    int thePart = this->particleNumberSlot.Param<core::param::IntParam>()->Value();

    if (this->lastTime != time || this->datahash != in->DataHash()) {
        in->SetFrameID(time, true);

        if (!(*in)(0)) {
            vislib::sys::Log::DefaultLog.WriteError("ParticleNeighborhood: could not get frame (%u)", time);
            return false;
        }

        plc = (inMpdc == nullptr) ? inDpdc->GetParticleListCount() : inMpdc->GetParticleListCount();

        size_t totalParts = 0;
        for (unsigned int i = 0; i < plc; i++) {
            if (isListOK(in, i))
                totalParts += getListCount(in, i);
        }

        if (theSearchType == searchTypeEnum::RADIUS) {
            this->newColors.resize(totalParts, theRadius);
        } else {
            this->newColors.resize(totalParts);
        }

        allParts.clear();
        allParts.reserve(totalParts);

        // we could now filter particles according to something. but currently we need not.
        size_t allpartcnt = 0;
        for (unsigned int pli = 0; pli < plc; pli++) {
            if (!isListOK(in, pli)) {
                continue;
            }

            UINT64 part_cnt = getListCount(in, pli);

            for (int part_i = 0; part_i < part_cnt; ++part_i) {
                //const float *v = reinterpret_cast<const float *>(vert + (part_i * vert_stride));
                allParts.push_back(allpartcnt + part_i);
            }
            allpartcnt += getListCount(in, pli);
        }

        // allocate nanoflann data structures for border
        assert(allpartcnt == totalParts);

        if (inMpdc != nullptr) {
            this->myPts = std::make_shared<simplePointcloud>(inMpdc, allParts);
            particleTree = std::make_shared<my_kd_tree_t>(3 /* dim */, *myPts, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
            particleTree->buildIndex();
            this->myDirPts = nullptr;
            dirParticleTree = nullptr;
        } else {
            this->myPts = nullptr;
            particleTree = nullptr;
            this->myDirPts = std::make_shared<directionalPointcloud>(inDpdc, allParts);
            dirParticleTree = std::make_shared<my_dir_kd_tree_t>(3 /* dim */, *myDirPts, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));;
            dirParticleTree->buildIndex();
        }
        this->datahash = in->DataHash();
        this->lastTime = time;
        this->radiusSlot.ForceSetDirty();
    }

    if (this->radiusSlot.IsDirty() || this->particleNumberSlot.IsDirty()
        || this->cyclXSlot.IsDirty() || this->cyclYSlot.IsDirty() || this->cyclZSlot.IsDirty()
        || this->numNeighborSlot.IsDirty() || this->searchTypeSlot.IsDirty()) {

        if (thePart >= 0) {

            if (thePart >= newColors.size()) {
                if (newColors.size() > 0) {
                    this->particleNumberSlot.Param<core::param::IntParam>()->SetValue(0);
                    thePart = 0;
                } else {
                    this->particleNumberSlot.Param<core::param::IntParam>()->SetValue(-1);
                    return true;
                }
            }

            const float *vbase;
            if (inDpdc != nullptr) {
                vbase = myDirPts->get_position(thePart);
            } else {
                vbase = myPts->get_position(thePart);
            }
            float theVertex[3];
            maxDist = 0.0f;
            std::vector<std::pair<size_t, float> > ret_matches;
            std::vector<std::pair<size_t, float> > ret_localMatches;
            std::vector<size_t> ret_index(theNumber);
            std::vector<float> out_dist_sqr(theNumber);
            nanoflann::KNNResultSet<float> resultSet(theNumber);
            nanoflann::SearchParams params;
            params.sorted = false;

            // final computation
            bool cycl_x = this->cyclXSlot.Param<megamol::core::param::BoolParam>()->Value();
            bool cycl_y = this->cyclYSlot.Param<megamol::core::param::BoolParam>()->Value();
            bool cycl_z = this->cyclZSlot.Param<megamol::core::param::BoolParam>()->Value();
            auto bbox = in->AccessBoundingBoxes().ObjectSpaceBBox();
            //bbox.EnforcePositiveSize(); // paranoia
            auto bbox_cntr = bbox.CalcCenter();

            ret_matches.clear();
            ret_matches.reserve(100);

            for (int x_s = 0; x_s < (cycl_x ? 2 : 1); ++x_s) {
                for (int y_s = 0; y_s < (cycl_y ? 2 : 1); ++y_s) {
                    for (int z_s = 0; z_s < (cycl_z ? 2 : 1); ++z_s) {

                        theVertex[0] = vbase[0];
                        theVertex[1] = vbase[1];
                        theVertex[2] = vbase[2];
                        if (x_s > 0) theVertex[0] = theVertex[0] + ((theVertex[0] > bbox_cntr.X()) ? -bbox.Width() : bbox.Width());
                        if (y_s > 0) theVertex[1] = theVertex[1] + ((theVertex[1] > bbox_cntr.Y()) ? -bbox.Height() : bbox.Height());
                        if (z_s > 0) theVertex[2] = theVertex[2] + ((theVertex[2] > bbox_cntr.Z()) ? -bbox.Depth() : bbox.Depth());

                        if (inMpdc != nullptr) {
                            if (theSearchType == searchTypeEnum::RADIUS) {
                                particleTree->radiusSearch(theVertex, theRadius, ret_localMatches, params);
                                ret_matches.insert(ret_matches.end(), ret_localMatches.begin(), ret_localMatches.end());
                            } else {
                                resultSet.init(ret_index.data(), out_dist_sqr.data());
                                particleTree->findNeighbors(resultSet, theVertex, params);
                                for (size_t i = 0; i < resultSet.size(); ++i) {
                                    ret_matches.push_back(std::pair<size_t, float>(ret_index[i], out_dist_sqr[i]));
                                }
                            }
                        } else {
                            if (theSearchType == searchTypeEnum::RADIUS) {
                                dirParticleTree->radiusSearch(theVertex, theRadius, ret_localMatches, params);
                                ret_matches.insert(ret_matches.end(), ret_localMatches.begin(), ret_localMatches.end());
                            } else {
                                resultSet.init(ret_index.data(), out_dist_sqr.data());
                                dirParticleTree->findNeighbors(resultSet, theVertex, params);
                                for (size_t i = 0; i < resultSet.size(); ++i) {
                                    ret_matches.push_back(std::pair<size_t, float>(ret_index[i], out_dist_sqr[i]));
                                }
                            }
                        }
                    }
                }
            }

            size_t num_matches = 0;

            // TODO this probably is not even worth it, writing something several times is probably cheaper.
            //sort(ret_matches.begin(), ret_matches.end());
            //ret_matches.erase(unique(ret_matches.begin(), ret_matches.end()), ret_matches.end());

            // reset all colors
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
            std::fill(newColors.begin(), newColors.end(), maxDist);

            for (size_t i = 0; i < num_matches; ++i) {
                this->newColors[ret_matches[i].first] = ret_matches[i].second;
            }
        } else {
        // reset all colors
            std::fill(newColors.begin(), newColors.end(), 0.0f);
        }
        this->radiusSlot.ResetDirty();
        this->particleNumberSlot.ResetDirty();
        this->cyclXSlot.ResetDirty();
        this->cyclYSlot.ResetDirty();
        this->cyclZSlot.ResetDirty();
        this->numNeighborSlot.ResetDirty();
        this->searchTypeSlot.ResetDirty();
    }
    in->SetUnlocker(nullptr, false);
    in->Unlock();

    size_t allpartcnt = 0;
    if (outMpdc != nullptr) {
        outMpdc->SetParticleListCount(plc);
        for (unsigned int i = 0; i < plc; ++i) {
            if (!isListOK(in, i)) {
                outMpdc->AccessParticles(i).SetCount(0);
                continue;
            }
            auto theCount = getListCount(in, i);
            outMpdc->AccessParticles(i).SetCount(theCount);
            if (inMpdc != nullptr) {
                outMpdc->AccessParticles(i).SetVertexData(inMpdc->AccessParticles(i).GetVertexDataType(),
                    inMpdc->AccessParticles(i).GetVertexData(), inMpdc->AccessParticles(i).GetVertexDataStride());
            } else {
                outMpdc->AccessParticles(i).SetVertexData(inDpdc->AccessParticles(i).GetVertexDataType(),
                    inDpdc->AccessParticles(i).GetVertexData(), inDpdc->AccessParticles(i).GetVertexDataStride());
            }
            outMpdc->AccessParticles(i).SetColourData(core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I,
                this->newColors.data() + allpartcnt, 0);
            outMpdc->AccessParticles(i).SetColourMapIndexValues(0.0f, maxDist);
            allpartcnt += theCount;
        }
    } else if (outDpdc != nullptr) {
        outDpdc->SetParticleListCount(plc);
        for (unsigned int i = 0; i < plc; ++i) {
            if (!isListOK(in, i)) {
                outDpdc->AccessParticles(i).SetCount(0);
                continue;
            }
            auto theCount = getListCount(in, i);
            outDpdc->AccessParticles(i).SetCount(theCount);
            if (inMpdc != nullptr) {
                outDpdc->AccessParticles(i).SetVertexData(inMpdc->AccessParticles(i).GetVertexDataType(), 
                    inMpdc->AccessParticles(i).GetVertexData(), inMpdc->AccessParticles(i).GetVertexDataStride());
                outDpdc->AccessParticles(i).SetDirData(DirectionalParticleDataCall::Particles::DirDataType::DIRDATA_NONE, nullptr, 0);
            } else {
                outDpdc->AccessParticles(i).SetVertexData(inDpdc->AccessParticles(i).GetVertexDataType(),
                    inDpdc->AccessParticles(i).GetVertexData(), inDpdc->AccessParticles(i).GetVertexDataStride());
                outDpdc->AccessParticles(i).SetDirData(inDpdc->AccessParticles(i).GetDirDataType(),
                    inDpdc->AccessParticles(i).GetDirData(), inDpdc->AccessParticles(i).GetDirDataStride());
            }
            outDpdc->AccessParticles(i).SetColourData(core::moldyn::DirectionalParticleDataCall::Particles::COLDATA_FLOAT_I,
                this->newColors.data() + allpartcnt, 0);
            outDpdc->AccessParticles(i).SetColourMapIndexValues(0.0f, maxDist);
            allpartcnt += theCount;
        }
    }
    out->SetUnlocker(in->GetUnlocker());
    return true;
}


bool datatools::ParticleNeighborhood::getExtentCallback(megamol::core::Call& c) {
    using megamol::core::moldyn::MultiParticleDataCall;
    using megamol::core::moldyn::DirectionalParticleDataCall;

    DirectionalParticleDataCall *outDpdc = dynamic_cast<DirectionalParticleDataCall*>(&c);
    MultiParticleDataCall *outMpdc = dynamic_cast<MultiParticleDataCall*>(&c);
    if (outMpdc == nullptr && outDpdc == nullptr) return false;

    DirectionalParticleDataCall *inDpdc = this->inDataSlot.CallAs<DirectionalParticleDataCall>();
    MultiParticleDataCall *inMpdc = this->inDataSlot.CallAs<MultiParticleDataCall>();
    if (inDpdc == nullptr && inMpdc == nullptr) return false;

    megamol::core::AbstractGetData3DCall *out;
    if (outMpdc != nullptr) out = outMpdc;
    if (outDpdc != nullptr) out = outDpdc;
    megamol::core::AbstractGetData3DCall *in;
    if (inDpdc != nullptr) in = inDpdc;
    if (inMpdc != nullptr) in = inMpdc;


    //if (!this->assertData(inMpdc, outDpdc)) return false;
    in->SetFrameID(out->FrameID(), true);
    if (!(*in)(1)) {
        vislib::sys::Log::DefaultLog.WriteError("ParticleNeighborhood: could not get current frame extents (%u)", out->FrameID());
        return false;
    }
    out->AccessBoundingBoxes().SetObjectSpaceBBox(in->GetBoundingBoxes().ObjectSpaceBBox());
    out->AccessBoundingBoxes().SetObjectSpaceClipBox(in->GetBoundingBoxes().ObjectSpaceClipBox());
    if (in->FrameCount() < 1) {
        vislib::sys::Log::DefaultLog.WriteError("ParticleNeighborhood: no frame data!");
        return false;
    }
    out->SetFrameCount(in->FrameCount());
    // TODO: what am I actually doing here
    in->SetUnlocker(nullptr, false);
    in->Unlock();

    return true;
}

bool datatools::ParticleNeighborhood::getDataCallback(megamol::core::Call& c) {
    using megamol::core::moldyn::MultiParticleDataCall;
    using megamol::core::moldyn::DirectionalParticleDataCall;

    DirectionalParticleDataCall *outDpdc = dynamic_cast<DirectionalParticleDataCall*>(&c);
    MultiParticleDataCall *outMpdc = dynamic_cast<MultiParticleDataCall*>(&c);
    if (outMpdc == nullptr && outDpdc == nullptr) return false;

    DirectionalParticleDataCall *inDpdc = this->inDataSlot.CallAs<DirectionalParticleDataCall>();
    MultiParticleDataCall *inMpdc = this->inDataSlot.CallAs<MultiParticleDataCall>();
    if (inDpdc == nullptr && inMpdc == nullptr) return false;

    core::AbstractGetData3DCall *in, *out;
    if (inDpdc == nullptr) {
        in = inMpdc;
    } else {
        in = inDpdc;
    }
    if (outDpdc == nullptr) {
        out = outMpdc;
    } else {
        out = outDpdc;
    }

    if (!this->assertData(in, out)) return false;

    //inMpdc->Unlock();

    return true;
}