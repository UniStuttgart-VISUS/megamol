/*
 * MultiParticleRelister.cpp
 *
 * Copyright (C) 2015 by MegaMol Team (TU Dresden)
 * Alle Rechte vorbehalten.
 */
#include "MultiParticleRelister.h"

using namespace megamol;
using namespace megamol::datatools;

MultiParticleRelister::MultiParticleRelister()
        : AbstractParticleManipulator("outData", "indata")
        , getRelistInfoSlot("getRelistInfo", "Connects to the provider of the relist info")
        , inDataHash(0)
        , inFrameId(0)
        , inRelistHash(0)
        , outRelistHash(0)
        , colDatTyp(geocalls::SimpleSphericalParticles::COLDATA_NONE)
        , globCol(192, 192, 192, 255)
        , verDatTyp(geocalls::SimpleSphericalParticles::VERTDATA_NONE)
        , globRad(0.5f)
        , partSize(0)
        , colOffset(0)
        , data() {
    getRelistInfoSlot.SetCompatibleCall<geocalls::ParticleRelistCallDescription>();
    MakeSlotAvailable(&getRelistInfoSlot);
}

MultiParticleRelister::~MultiParticleRelister() {
    Release();
}

bool MultiParticleRelister::manipulateData(
    megamol::geocalls::MultiParticleDataCall& outData, megamol::geocalls::MultiParticleDataCall& inData) {
    geocalls::ParticleRelistCall* prc = getRelistInfoSlot.CallAs<geocalls::ParticleRelistCall>();
    if (prc == nullptr) {
        outData = inData;                   // also transfers the unlocker to 'outData'
        inData.SetUnlocker(nullptr, false); // keep original data locked
                                            // original data will be unlocked through outData
        return true;
    }
    prc->SetFrameID(inData.FrameID(), inData.IsFrameForced());
    if (!(*prc)()) {
        outData = inData;                   // also transfers the unlocker to 'outData'
        inData.SetUnlocker(nullptr, false); // keep original data locked
                                            // original data will be unlocked through outData
        return true;
    }

    if ((inData.DataHash() != inDataHash) || (inDataHash == 0) || (inData.FrameID() != inFrameId) ||
        (prc->DataHash() != inRelistHash) || (inRelistHash == 0)) {
        // need to update the internal data
        inDataHash = inData.DataHash();
        inFrameId = inData.FrameID();
        inRelistHash = prc->DataHash();

        outRelistHash++;

        data.clear();
        colDatTyp = geocalls::SimpleSphericalParticles::COLDATA_NONE;
        verDatTyp = geocalls::SimpleSphericalParticles::VERTDATA_NONE;
        partSize = 0;
        colOffset = 0;

        unsigned int inPLC = inData.GetParticleListCount();
        uint64_t inPC =
            (inPLC >= 0) ? inData.AccessParticles(0).GetCount() : 0; // HAZARD: I want all particles in the first list
        if (inPC == prc->SourceParticleCount()) {
            copyData(inData.AccessParticles(0), *prc);

        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "(MultiParticleRelister) Cannot combine data from MultiParticleDataCall and ParticleRelistCall: "
                "Particle counts not equal");
        }
    }

    if (data.size() == 0) {
        outData = inData;                   // also transfers the unlocker to 'outData'
        inData.SetUnlocker(nullptr, false); // keep original data locked
                                            // original data will be unlocked through outData
        return true;
    } else {
        outData.SetDataHash(outRelistHash);
        outData.SetFrameID(inFrameId);
        outData.SetParticleListCount(static_cast<unsigned int>(data.size()));
        for (size_t i = 0; i < data.size(); ++i) {
            auto& p = outData.AccessParticles(static_cast<unsigned int>(i));
            assert((data[i].size() % partSize) == 0);
            p.SetCount(data[i].size() / partSize);
            p.SetGlobalColour(globCol.R(), globCol.G(), globCol.B(), globCol.A());
            p.SetGlobalRadius(globRad);
            p.SetColourData(colDatTyp, data[i].data() + colOffset);
            p.SetVertexData(verDatTyp, data[i].data());
        }
        outData.SetUnlocker(nullptr);
    }

    return true;
}

void MultiParticleRelister::copyData(
    const geocalls::SimpleSphericalParticles& inData, const geocalls::ParticleRelistCall& relist) {
    colDatTyp = inData.GetColourDataType();
    globCol.Set(inData.GetGlobalColour()[0], inData.GetGlobalColour()[1], inData.GetGlobalColour()[2],
        inData.GetGlobalColour()[3]);
    verDatTyp = inData.GetVertexDataType();
    globRad = inData.GetGlobalRadius();

    size_t colSize = 0;
    switch (colDatTyp) {
    case geocalls::SimpleSphericalParticles::COLDATA_NONE:
        colSize = 0;
        break;
    case geocalls::SimpleSphericalParticles::COLDATA_UINT8_RGB:
        colSize = 3;
        break;
    case geocalls::SimpleSphericalParticles::COLDATA_UINT8_RGBA:
        colSize = 4;
        break;
    case geocalls::SimpleSphericalParticles::COLDATA_FLOAT_RGB:
        colSize = 12;
        break;
    case geocalls::SimpleSphericalParticles::COLDATA_FLOAT_RGBA:
        colSize = 16;
        break;
    case geocalls::SimpleSphericalParticles::COLDATA_FLOAT_I:
        colSize = 4;
        break;
    default:
        colSize = 0;
    }
    size_t colStep = std::max<size_t>(colSize, inData.GetColourDataStride());
    const uint8_t* colDat = reinterpret_cast<const uint8_t*>(inData.GetColourData());

    size_t verSize = 0;
    switch (verDatTyp) {
    case geocalls::SimpleSphericalParticles::VERTDATA_NONE:
        verSize = 0;
        break;
    case geocalls::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ:
        verSize = 12;
        break;
    case geocalls::SimpleSphericalParticles::VERTDATA_FLOAT_XYZR:
        verSize = 16;
        break;
    case geocalls::SimpleSphericalParticles::VERTDATA_SHORT_XYZ:
        verSize = 6;
        break;
    default:
        verSize = 0;
        break;
    }
    size_t verStep = std::max<size_t>(verSize, inData.GetVertexDataStride());
    const uint8_t* verDat = reinterpret_cast<const uint8_t*>(inData.GetVertexData());

    partSize = verSize + colSize;
    colOffset = verSize;

    const geocalls::ParticleRelistCall::ListIDType* relistData = relist.SourceParticleTargetLists();
    data.resize(relist.TargetListCount());

    for (size_t pi = 0; pi < inData.GetCount(); ++pi, colDat += colStep, verDat += verStep, relistData++) {
        for (size_t i = 0; i < verSize; ++i)
            data[*relistData].push_back(verDat[i]);
        for (size_t i = 0; i < colSize; ++i)
            data[*relistData].push_back(colDat[i]);
    }
}
