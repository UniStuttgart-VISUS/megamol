#include "stdafx.h"
#include "RemapIColValues.h"
#include "mmstd_datatools/ParticleFilterMapDataCall.h"
#include "mmstd_datatools/MultiParticleDataAdaptor.h"

using namespace megamol;
using namespace megamol::stdplugin;
using namespace megamol::stdplugin::datatools;

RemapIColValues::RemapIColValues() : AbstractParticleManipulator("outData", "inData"),
        inIColValuesSlot("inIColData", "The particles holding the ICol data to be used"),
        inParticleMapSlot("inMapData", "The particle index mapping data"),
        outDataHash(0), frameId(0), col(), minCol(0.0f), maxCol(1.0f) {

    inIColValuesSlot.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&inIColValuesSlot);

    inParticleMapSlot.SetCompatibleCall<ParticleFilterMapDataCallDescription>();
    MakeSlotAvailable(&inParticleMapSlot);
}

RemapIColValues::~RemapIColValues() {
    Release();
}

bool RemapIColValues::manipulateData(
        core::moldyn::MultiParticleDataCall& outData,
        core::moldyn::MultiParticleDataCall& inData) {
    core::moldyn::MultiParticleDataCall *inIColData = inIColValuesSlot.CallAs<core::moldyn::MultiParticleDataCall>();
    if (inIColData == nullptr) return false;
    ParticleFilterMapDataCall *inMapData = inParticleMapSlot.CallAs<ParticleFilterMapDataCall>();
    if (inMapData == nullptr) return false;

    if (!(*inMapData)(ParticleFilterMapDataCall::GET_HASH)) return false;
    size_t nimh = inMapData->DataHash();
    inMapData->Unlock();

    // get color data
    inIColData->SetFrameID(inData.FrameID(), true);
    if (!(*inIColData)(1))
        return false;
    if (!(*inIColData)(0)) return false;

    if ((dataHash != inData.DataHash()) 
            || (inIColHash != inIColData->DataHash())
            || (inMapHash != nimh)
            || (frameId != inData.FrameID())) {
        // Update data
        outDataHash++;
        dataHash = inData.DataHash();
        inIColHash = inIColData->DataHash();
        inMapHash = nimh;
        frameId = inData.FrameID();

        MultiParticleDataAdaptor tarD(inData);
        MultiParticleDataAdaptor srcD(*inIColData);

        col.resize(tarD.get_count());

        if ((tarD.get_count() > 0) && ((*inMapData)(ParticleFilterMapDataCall::GET_DATA))) {
            if (inMapData->Size() != tarD.get_count()) {
                megamol::core::utility::log::Log::DefaultLog.WriteError("Filtered particle data and map data not compatible");
                inMapData->Unlock();
                inIColData->Unlock();
                return false;
            }

            for (size_t i = 0; i < tarD.get_count(); ++i) {
                if (inMapData->Data()[i] >= srcD.get_count()) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError("Unfiltered particle data and map data not compatible");
                    inMapData->Unlock();
                    inIColData->Unlock();
                    return false;
                }
                col[i] = *srcD.get_color(inMapData->Data()[i]);
            }

            minCol = maxCol = col[0];
            for (size_t i = 1; i < tarD.get_count(); ++i) {
                if (minCol > col[i]) minCol = col[i];
                if (maxCol < col[i]) maxCol = col[i];
            }

            inMapData->Unlock();
        } else {
            std::fill(col.begin(), col.end(), 0.0f);
            minCol = 0.0f;
            maxCol = 1.0f;
        }
    }

    inIColData->Unlock();

    outData = inData;
    inData.SetUnlocker(nullptr, false);
    outData.SetDataHash(outDataHash);
    outData.SetFrameID(frameId);

    size_t off = 0;
    for (unsigned int pli = 0; pli < outData.GetParticleListCount(); ++pli) {
        auto& pl = outData.AccessParticles(pli);
        if (pl.GetCount() <= 0) continue;
        assert(col.size() >= off + pl.GetCount());
        pl.SetColourData(core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I, col.data() + off, 0);
        pl.SetColourMapIndexValues(minCol, maxCol);
        off += static_cast<size_t>(pl.GetCount());
    }

    return true;
}
