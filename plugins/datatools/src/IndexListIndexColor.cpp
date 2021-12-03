/*
 * IndexListIndexColor.cpp
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "IndexListIndexColor.h"
#include "datatools/MultiIndexListDataCall.h"
#include "datatools/MultiParticleDataAdaptor.h"
#include "stdafx.h"
#include <algorithm>

using namespace megamol;
using namespace megamol::datatools;


IndexListIndexColor::IndexListIndexColor()
        : datatools::AbstractParticleManipulator("outData", "inData")
        , inIndexListDataSlot("inIndexListData", "Fetches the second ICol value stream")
        , inPartsHash(0)
        , inIndexHash(0)
        , outHash(0)
        , frameID(0)
        , colors()
        , minCol(0.0f)
        , maxCol(1.0f) {

    inIndexListDataSlot.SetCompatibleCall<MultiIndexListDataCallDescription>();
    MakeSlotAvailable(&inIndexListDataSlot);
}

IndexListIndexColor::~IndexListIndexColor() {
    Release();
}

bool IndexListIndexColor::manipulateData(
    geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) {

    MultiIndexListDataCall* inListsPtr = inIndexListDataSlot.CallAs<MultiIndexListDataCall>();
    if (inListsPtr == nullptr)
        return false;

    inListsPtr->SetFrameID(inData.FrameID());
    if (!(*inListsPtr)(MultiIndexListDataCall::GET_HASH))
        return false;

    if ((inPartsHash != inData.DataHash()) || (inData.DataHash() == 0) || (inIndexHash != inListsPtr->DataHash()) ||
        (inListsPtr->DataHash() == 0) || (frameID != inData.FrameID())) {
        // Update data
        inPartsHash = inData.DataHash();
        inIndexHash = inListsPtr->DataHash();
        outHash++;
        frameID = inData.FrameID();

        inListsPtr->Unlock();
        inListsPtr->SetFrameID(inData.FrameID());
        if (!(*inListsPtr)(MultiIndexListDataCall::GET_DATA))
            return false;

        datatools::MultiParticleDataAdaptor p(inData);
        size_t pCnt = p.get_count();
        colors.resize(pCnt);

        if (pCnt == 0) {
            minCol = maxCol = 0.0f;

        } else {
            // initialize as "unreferenced"
            std::fill(colors.begin(), colors.end(), -1.0f);
            size_t ptsSet = 0;      // numbers of point-references set
            bool warnIndex = false; // should warn about illegal indices?

            maxCol = -1.0f; // if no particle color is ever set, this will be max

            // iterate through all references
            size_t listCnt = inListsPtr->Count();
            for (size_t listI = 0; listI < listCnt; ++listI) {
                MultiIndexListDataCall::index_list_t const& list = inListsPtr->Lists()[listI];
                size_t oldPtsSet = ptsSet;
                for (MultiIndexListDataCall::index_t const index : list) {
                    if (index >= pCnt) {
                        warnIndex = true;
                        continue;
                    }
                    if (colors[index] < -0.9f) {
                        ptsSet++;
                        colors[index] = static_cast<float>(listI);
                    }
                }
                // if at least one point was set, listI exists as color, thus is the current max value.
                if (oldPtsSet < ptsSet)
                    maxCol = static_cast<float>(listI);
            }

            // set min, based if all points have been referenced
            assert(ptsSet <= pCnt);
            minCol = (ptsSet == pCnt) ? 0.0f : -1.0f;
        }
    }

    inListsPtr->Unlock();

    outData = inData;
    outData.SetDataHash(outHash);
    outData.SetFrameID(frameID);
    inData.SetUnlocker(nullptr, false);

    const float* data = colors.data();
    for (unsigned int list = 0; list < outData.GetParticleListCount(); ++list) {
        auto& plist = outData.AccessParticles(list);
        plist.SetColourData(geocalls::SimpleSphericalParticles::COLDATA_FLOAT_I, data, 0);
        plist.SetColourMapIndexValues(minCol, maxCol);
        data += plist.GetCount();
    }

    return true;
}
