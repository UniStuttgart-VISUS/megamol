/*
 * ParticleListFilter.cpp
 *
 * Copyright (C) 2013 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/moldyn/ParticleListFilter.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/BoolParam.h"
#include <climits>
#include <cfloat>
#include "vislib/StringTokeniser.h"

using namespace megamol::core;


/*
 * moldyn::ParticleListFilter::ParticleListFilter
 */
moldyn::ParticleListFilter::ParticleListFilter(void) : Module(),
        inParticlesDataSlot("inPartData", "Input for (oriented) particle data"),
        outParticlesDataSlot("outPartData", "Output of (oriented) particle data"),
        includedListsSlot("includedTypes", "Comma-separated list of particle types to include"),
        includeAllSlot("includeAll", "Button to populate includedLists from available data"),
        globalColorMapComputationSlot("globalMap", "Compute color map min/max from all particle lists (false: compute them per list)"),
        includeHiddenInColorMapSlot("includeHidden", "Include hidden particle lists in color map min/max computation"),
        datahashParticlesIn(0), datahashParticlesOut(0),
        frameID(0) {

    //this->inParticlesDataSlot.SetCompatibleCall<moldyn::DirectionalParticleDataCallDescription>();
    this->inParticlesDataSlot.SetCompatibleCall<moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->inParticlesDataSlot);

    this->outParticlesDataSlot.SetCallback("MultiParticleDataCall", "GetData", &ParticleListFilter::getDataCallback);
    this->outParticlesDataSlot.SetCallback("MultiParticleDataCall", "GetExtent", &ParticleListFilter::getExtentCallback);
    this->MakeSlotAvailable(&this->outParticlesDataSlot);

    this->includedListsSlot << new param::StringParam("");
    this->MakeSlotAvailable(&this->includedListsSlot);

    this->includeAllSlot << new param::ButtonParam();
    this->MakeSlotAvailable(&this->includeAllSlot);

    this->globalColorMapComputationSlot << new param::BoolParam(true);
    this->MakeSlotAvailable(&this->globalColorMapComputationSlot);
    this->includeHiddenInColorMapSlot << new param::BoolParam(true);
    this->MakeSlotAvailable(&this->includeHiddenInColorMapSlot);
}


/*
 * moldyn::ParticleListFilter::~ParticleListFilter
 */
moldyn::ParticleListFilter::~ParticleListFilter(void) {
    this->Release(); // implicitly calls 'release'
}


/*
 * moldyn::ParticleListFilter::create
 */
bool moldyn::ParticleListFilter::create(void) {
    // intentionally empty
    return true;
}


/*
 * moldyn::ParticleListFilter::release
 */
void moldyn::ParticleListFilter::release(void) {
    // intentionally empty
}

vislib::Array<unsigned int> moldyn::ParticleListFilter::getSelectedLists() {
    vislib::StringA str = this->includedListsSlot.Param<megamol::core::param::StringParam>()->Value();
    vislib::StringTokeniserA sta(str, ',');
    vislib::Array<unsigned int> result;
    while(sta.HasNext()) {
        vislib::StringA t = sta.Next();
        if (t.IsEmpty()) {
            continue;
        }
        result.Add(static_cast<unsigned int>(vislib::CharTraitsA::ParseInt64(t)));
    }
    return result;
}

/*
 * moldyn::ParticleListFilter::getData
 */
bool moldyn::ParticleListFilter::getDataCallback(Call& call) {

    MultiParticleDataCall *outMpdc = dynamic_cast<MultiParticleDataCall*>(&call);
    if (outMpdc == NULL) return false;

    bool doStuff = false;
    if (this->globalColorMapComputationSlot.IsDirty()) {
        this->globalColorMapComputationSlot.ResetDirty();
        doStuff = true;
    }
    if (this->includeHiddenInColorMapSlot.IsDirty()) { 
        this->includeHiddenInColorMapSlot.ResetDirty();
        doStuff = true;
    }

    if (outMpdc != NULL) {
        MultiParticleDataCall *inMpdc = this->inParticlesDataSlot.CallAs<MultiParticleDataCall>();
        if (inMpdc == NULL) return false;

        if (this->includeAllSlot.IsDirty()) {
            this->includeAllSlot.ResetDirty();
            vislib::StringA str, str2;
            unsigned int cnt = inMpdc->GetParticleListCount();
            for (unsigned int i = 0; i < cnt; i++) {
                str2.Format("%u%s", inMpdc->AccessParticles(i).GetGlobalType(), (i == cnt - 1) ? "" : ", ");
                str.Append(str2);
            }
            this->includedListsSlot.Param<megamol::core::param::StringParam>()->SetValue(str);
        }
        vislib::Array<unsigned int> included = this->getSelectedLists();
        if (this->includedListsSlot.IsDirty()) {
            this->includedListsSlot.ResetDirty();
            doStuff = true;
        }

        // make a deep copy (also of content pointers)
        //*inDpdc = *outDpdc;
        // call DataCallback, updating content
        if (!(*inMpdc)(0)) {
            return false;
        }
        // copy diffs back (maybe)
        //*outDpdc = *inDpdc;

        if (inMpdc->DataHash() != this->datahashParticlesIn) {
            doStuff = true;
            this->datahashParticlesIn = inMpdc->DataHash();
        }
        // not sure
        if (outMpdc->FrameID() != this->frameID) {
            doStuff = true;
        }

        if (!doStuff) {
            return true;
        }

        unsigned int cnt = inMpdc->GetParticleListCount();
        unsigned int outCnt = 0;
        if (included.Count() == 0) {
            outCnt = cnt;
        } else {
            for (unsigned int i = 0; i < cnt; i++) {
                if (included.Contains(inMpdc->AccessParticles(i).GetGlobalType())) {
                    outCnt++;
                }
            }
        }
        outMpdc->SetParticleListCount(outCnt);
        outCnt = 0;
        float globalMin = FLT_MAX;
        float globalMax = -FLT_MAX;
        for (unsigned int i = 0; i < cnt; i++) {
            if (this->includeHiddenInColorMapSlot.Param<megamol::core::param::BoolParam>()->Value()
                || included.Count() == 0
                || (included.Count() > 0 && included.Contains(inMpdc->AccessParticles(i).GetGlobalType()))) {
                if (inMpdc->AccessParticles(i).GetMinColourIndexValue() < globalMin) globalMin = inMpdc->AccessParticles(i).GetMinColourIndexValue();
                if (inMpdc->AccessParticles(i).GetMaxColourIndexValue() > globalMax) globalMax = inMpdc->AccessParticles(i).GetMaxColourIndexValue();
            }
            if (included.Count() > 0 && !included.Contains(inMpdc->AccessParticles(i).GetGlobalType())) {
                continue;
            }
            outMpdc->AccessParticles(outCnt).SetCount(inMpdc->AccessParticles(i).GetCount());
            outMpdc->AccessParticles(outCnt).SetColourData(inMpdc->AccessParticles(i).GetColourDataType(),
                inMpdc->AccessParticles(i).GetColourData(), inMpdc->AccessParticles(i).GetColourDataStride());
            outMpdc->AccessParticles(outCnt).SetVertexData(inMpdc->AccessParticles(i).GetVertexDataType(),
                inMpdc->AccessParticles(i).GetVertexData(), inMpdc->AccessParticles(i).GetVertexDataStride());
            outMpdc->AccessParticles(outCnt).SetDirData(inMpdc->AccessParticles(i).GetDirDataType(),
                inMpdc->AccessParticles(i).GetDirData(), inMpdc->AccessParticles(i).GetDirDataStride());
            // TODO BUG HAZARD this is most probably wrong, as different list subsets have a different dynamic range :(
            // probably still loop over all...
            //outMpdc->AccessParticles(outCnt).SetColourMapIndexValues(inMpdc->AccessParticles(i).GetMinColourIndexValue(),
            //    inMpdc->AccessParticles(i).GetMaxColourIndexValue());

            const unsigned char *col = inMpdc->AccessParticles(i).GetGlobalColour();
            outMpdc->AccessParticles(outCnt).SetGlobalColour(col[0], col[1], col[2], col[3]);
            outMpdc->AccessParticles(outCnt).SetGlobalRadius(inMpdc->AccessParticles(i).GetGlobalRadius());
            outMpdc->AccessParticles(outCnt).SetGlobalType(inMpdc->AccessParticles(i).GetGlobalType());
            outCnt++;
        }
        outCnt = 0;
        for (unsigned int i = 0; i < cnt; i++) {
            if (included.Count() > 0 && !included.Contains(inMpdc->AccessParticles(i).GetGlobalType())) {
                continue;
            }
            if (this->globalColorMapComputationSlot.Param<megamol::core::param::BoolParam>()->Value()) {
                outMpdc->AccessParticles(outCnt).SetColourMapIndexValues(globalMin, globalMax);
            } else {
                outMpdc->AccessParticles(outCnt).SetColourMapIndexValues(inMpdc->AccessParticles(i).GetMinColourIndexValue(),
                    inMpdc->AccessParticles(i).GetMaxColourIndexValue());
            }
            outCnt++;
        }
        this->datahashParticlesOut++;
        outMpdc->SetDataHash(this->datahashParticlesOut);

    }
#if 0
    else if (outDpdc != NULL) {
        DirectionalParticleDataCall *inDpdc = this->inParticlesDataSlot.CallAs<DirectionalParticleDataCall>();
        if (inDpdc == NULL) return false;

        if (this->includeAllSlot.IsDirty()) {
            this->includeAllSlot.ResetDirty();
            vislib::StringA str, str2;
            unsigned int cnt = inDpdc->GetParticleListCount();
            for (unsigned int i = 0; i < cnt; i++) {
                str2.Format("%u%s", inDpdc->AccessParticles(i).GetGlobalType(), (i == cnt - 1) ? "" : ", ");
                str.Append(str2);
            }
            this->includedListsSlot.Param<megamol::core::param::StringParam>()->SetValue(str);
        }
        vislib::Array<unsigned int> included = this->getSelectedLists();
        if (this->includedListsSlot.IsDirty()) {
            this->includedListsSlot.ResetDirty();
            doStuff = true;
        }

        // TODO this is wrong
        // make a deep copy (also of content pointers)
        //*inDpdc = *outDpdc;
        //inDpdc->AbstractGetDataCall::operator =(*outDpdc);
        // call DataCallback, updating content
        if (!(*inDpdc)(0)) {
            return false;
        }
        // copy diffs back (maybe)
        //*outDpdc = *inDpdc;
        //outDpdc->AbstractGetData3DCall::operator =(*inDpdc);

        if (inDpdc->DataHash() != this->datahashParticlesIn) {
            doStuff = true;
            this->datahashParticlesIn = inDpdc->DataHash();
        }
        // not sure
        if (outDpdc->FrameID() != this->frameID) {
            doStuff = true;
        }

        if (!doStuff) {
            //return true;
        }

        unsigned int cnt = inDpdc->GetParticleListCount();
        unsigned int outCnt = 0;
        if (included.Count() == 0) {
            outCnt = cnt;
        } else {
            for (unsigned int i = 0; i < cnt; i++) {
                if (included.Contains(inDpdc->AccessParticles(i).GetGlobalType())) {
                    outCnt++;
                }
            }
        }
        outDpdc->SetParticleListCount(outCnt);
        outCnt = 0;
        float globalMin = FLT_MAX;
        float globalMax = -FLT_MAX;
        for (unsigned int i = 0; i < cnt; i++) {
            if (this->includeHiddenInColorMapSlot.Param<megamol::core::param::BoolParam>()->Value()
                || included.Count() == 0
                || (included.Count() > 0 && included.Contains(inDpdc->AccessParticles(i).GetGlobalType()))) {
                if (inDpdc->AccessParticles(i).GetMinColourIndexValue() < globalMin) globalMin = inDpdc->AccessParticles(i).GetMinColourIndexValue();
                if (inDpdc->AccessParticles(i).GetMaxColourIndexValue() > globalMax) globalMax = inDpdc->AccessParticles(i).GetMaxColourIndexValue();
            }
            if (included.Count() > 0 && !included.Contains(inDpdc->AccessParticles(i).GetGlobalType())) {
                continue;
            }
            outDpdc->AccessParticles(outCnt).SetCount(inDpdc->AccessParticles(i).GetCount());
            outDpdc->AccessParticles(outCnt).SetColourData(inDpdc->AccessParticles(i).GetColourDataType(),
                inDpdc->AccessParticles(i).GetColourData(), inDpdc->AccessParticles(i).GetColourDataStride());
            outDpdc->AccessParticles(outCnt).SetVertexData(inDpdc->AccessParticles(i).GetVertexDataType(),
                inDpdc->AccessParticles(i).GetVertexData(), inDpdc->AccessParticles(i).GetVertexDataStride());
            // TODO BUG HAZARD this is most probably wrong, as different list subsets have a different dynamic range :(
            // probably still loop over all...
            //outDpdc->AccessParticles(outCnt).SetColourMapIndexValues(inDpdc->AccessParticles(i).GetMinColourIndexValue(),
            //    inDpdc->AccessParticles(i).GetMaxColourIndexValue());
            const unsigned char *col = inDpdc->AccessParticles(i).GetGlobalColour();
            outDpdc->AccessParticles(outCnt).SetGlobalColour(col[0], col[1], col[2], col[3]);
            outDpdc->AccessParticles(outCnt).SetGlobalRadius(inDpdc->AccessParticles(i).GetGlobalRadius());
            outDpdc->AccessParticles(outCnt).SetGlobalType(inDpdc->AccessParticles(i).GetGlobalType());
            outDpdc->AccessParticles(outCnt).SetDirData(inDpdc->AccessParticles(i).GetDirDataType(),
                inDpdc->AccessParticles(i).GetDirData(), inDpdc->AccessParticles(i).GetDirDataStride());
            outCnt++;
        }
        outCnt = 0;
        for (unsigned int i = 0; i < cnt; i++) {
            if (included.Count() > 0 && !included.Contains(inDpdc->AccessParticles(i).GetGlobalType())) {
                continue;
            }
            if (this->globalColorMapComputationSlot.Param<megamol::core::param::BoolParam>()->Value()) {
                outDpdc->AccessParticles(outCnt).SetColourMapIndexValues(globalMin, globalMax);
            } else {
                outDpdc->AccessParticles(outCnt).SetColourMapIndexValues(inDpdc->AccessParticles(i).GetMinColourIndexValue(),
                    inDpdc->AccessParticles(i).GetMaxColourIndexValue());
            }
            outCnt++;
        }
        this->datahashParticlesOut++;
        outDpdc->SetDataHash(this->datahashParticlesOut);
    }
#endif
    return true;
}


/*
 * moldyn::DirPartColModulate::getExtent
 */
bool moldyn::ParticleListFilter::getExtentCallback(Call& call) {
    MultiParticleDataCall *outMpdc = dynamic_cast<MultiParticleDataCall*>(&call);
    if (outMpdc == NULL) return false;

    if (outMpdc != NULL) {
        MultiParticleDataCall *inMpdc = this->inParticlesDataSlot.CallAs<MultiParticleDataCall>();
        if (inMpdc == NULL) return false;
        // this is the devil. don't do it.
        //*inMpdc = *outMpdc;
        // this is better but still not good.
        static_cast<AbstractGetData3DCall*>(inMpdc)->operator=(*outMpdc);
        if ((*inMpdc)(1)) {
            //*outMpdc = *inMpdc;
            static_cast<AbstractGetData3DCall*>(outMpdc)->operator=(*inMpdc);
            outMpdc->SetDataHash(this->datahashParticlesOut);
            return true;
        }
    }
#if 0
    else if (outDpdc != NULL) {
        DirectionalParticleDataCall *inDpdc = this->inParticlesDataSlot.CallAs<DirectionalParticleDataCall>();
        if (inDpdc == NULL) return false;
        // this is the devil. don't do it.
        //*inDpdc = *outDpdc;
        // this is better but still not good.
        static_cast<AbstractGetData3DCall*>(inDpdc)->operator=(*outDpdc);
        // call ExtentCallback, updating content
        if ((*inDpdc)(1)) {
            // copy diffs back (maybe)
            //*outDpdc = *inDpdc;
            // WTF??????
            static_cast<AbstractGetData3DCall*>(outDpdc)->operator=(*inDpdc);
            outDpdc->SetDataHash(this->datahashParticlesOut);
            return true;
        }
    }
#endif
    return false;
}
