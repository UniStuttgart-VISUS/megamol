#include "stdafx.h"
#include "ParticlesToPaths.h"

#include "mmcore/moldyn/DirectionalParticleDataCall.h"

#include "vislib/sys/Log.h"

#include "thermodyn/PathLineDataCall.h"


megamol::thermodyn::ParticlesToPaths::ParticlesToPaths()
    : dataInSlot_("dataIn", "Input of particle data"), dataOutSlot_("dataOut", "Output of particle pathlines") {
    dataInSlot_.SetCompatibleCall<core::moldyn::DirectionalParticleDataCallDescription>();
    MakeSlotAvailable(&dataInSlot_);

    dataOutSlot_.SetCallback(
        PathLineDataCall::ClassName(), PathLineDataCall::FunctionName(0), &ParticlesToPaths::getDataCallback);
    dataOutSlot_.SetCallback(
        PathLineDataCall::ClassName(), PathLineDataCall::FunctionName(1), &ParticlesToPaths::getExtentCallback);
    MakeSlotAvailable(&dataOutSlot_);
}


megamol::thermodyn::ParticlesToPaths::~ParticlesToPaths() { this->Release(); }


bool megamol::thermodyn::ParticlesToPaths::create() { return true; }


void megamol::thermodyn::ParticlesToPaths::release() {}


bool megamol::thermodyn::ParticlesToPaths::getDataCallback(core::Call& c) {
    auto outCall = dynamic_cast<PathLineDataCall*>(&c);
    if (outCall == nullptr) return false;

    auto inCall = dataInSlot_.CallAs<core::moldyn::DirectionalParticleDataCall>();
    if (inCall == nullptr) return false;

    auto const plc = inCall->GetParticleListCount();
    auto const frameCount = inCall->FrameCount();

    if (inCall->DataHash() != inDataHash_) {
        inDataHash_ = inCall->DataHash();

        pathStore_.resize(plc);

        for (unsigned int plidx = 0; plidx < plc; ++plidx) {
            // step over all time frames to create complete pathline
            auto const& part = inCall->AccessParticles(plidx);
            auto const& pCount = part.GetCount();
            auto& storeEntry = pathStore_[plidx];
            storeEntry.reserve(pCount);

            auto const idDT = part.GetIDDataType();
            if (idDT == core::moldyn::SimpleSphericalParticles::IDDATA_NONE) {
                vislib::sys::Log::DefaultLog.WriteWarn(
                    "ParticlesToPaths: Particlelist entry %d does not contain particle IDs ... Skipping entry\n");
                continue;
            }

            auto const vertDT = part.GetVertexDataType();
            if (vertDT == core::moldyn::SimpleSphericalParticles::VERTDATA_NONE ||
                vertDT == core::moldyn::SimpleSphericalParticles::VERTDATA_SHORT_XYZ) {
                vislib::sys::Log::DefaultLog.WriteWarn("ParticlesToPaths: Particlelist entry %d does not contain "
                                                       "compatible particle positions ... Skipping entry\n");
                continue;
            }

            bool dirPresent = false;
            auto const dirDT = part.GetDirDataType();
            if (dirDT != core::moldyn::DirectionalParticles::DIRDATA_NONE) dirPresent = true;
            bool colPresent = false;
            auto const colDT = part.GetColourDataType();
            if (colDT != core::moldyn::SimpleSphericalParticles::COLDATA_NONE) colPresent = true;

            auto& xAcc = part.GetParticleStore().GetXAcc();
            auto& yAcc = part.GetParticleStore().GetYAcc();
            auto& zAcc = part.GetParticleStore().GetZAcc();
            auto& dxAcc = part.GetParticleStore().GetDXAcc();
            auto& dyAcc = part.GetParticleStore().GetDYAcc();
            auto& dzAcc = part.GetParticleStore().GetDZAcc();
            auto& rAcc = part.GetParticleStore().GetCRAcc();
            auto& gAcc = part.GetParticleStore().GetCGAcc();
            auto& bAcc = part.GetParticleStore().GetCBAcc();
            auto& aAcc = part.GetParticleStore().GetCAAcc();
            auto& idAcc = part.GetParticleStore().GetIDAcc();

            entrySize_ = 3;
            if (dirPresent) entrySize_ += 3;

            for (size_t pidx = 0; pidx < pCount; ++pidx) {
                storeEntry[idAcc->Get_u64(pidx)].reserve(frameCount * entrySize_);
            }

            for (unsigned int fidx = 0; fidx < frameCount; ++fidx) {
                do {
                    inCall->SetFrameID(fidx, true);
                    (*inCall)(1);
                } while (fidx != inCall->FrameID());

                if (!(*inCall)(0)) return false;

                for (size_t pidx = 0; pidx < pCount; ++pidx) {
                    auto idx = idAcc->Get_u64(pidx);
                    auto& entry = storeEntry[idx];
                    entry.push_back(xAcc->Get_f(pidx));
                    entry.push_back(yAcc->Get_f(pidx));
                    entry.push_back(zAcc->Get_f(pidx));
                    if (dirPresent) {
                        entry.push_back(dxAcc->Get_f(pidx));
                        entry.push_back(dyAcc->Get_f(pidx));
                        entry.push_back(dzAcc->Get_f(pidx));
                    }
                    if (colPresent) {
                        entry.push_back(rAcc->Get_f(pidx));
                        entry.push_back(gAcc->Get_f(pidx));
                        entry.push_back(bAcc->Get_f(pidx));
                        entry.push_back(aAcc->Get_f(pidx));
                    }
                }
            }
        }
    }

    // all data is serialized
    outCall->SetEntrySize(entrySize_);
    outCall->SetPathStore(&pathStore_);

    return true;
}


bool megamol::thermodyn::ParticlesToPaths::getExtentCallback(core::Call& c) {
    auto outCall = dynamic_cast<PathLineDataCall*>(&c);
    if (outCall == nullptr) return false;

    auto inCall = dataInSlot_.CallAs<core::moldyn::DirectionalParticleDataCall>();
    if (inCall == nullptr) return false;

    if (!(*inCall)(1)) return false;

    outCall->AccessBoundingBoxes().SetObjectSpaceBBox(inCall->AccessBoundingBoxes().ObjectSpaceBBox());
    outCall->AccessBoundingBoxes().SetObjectSpaceClipBox(inCall->AccessBoundingBoxes().ObjectSpaceClipBox());
    outCall->AccessBoundingBoxes().MakeScaledWorld(1.0f);

    outCall->SetFrameCount(1);

    outCall->SetDataHash(inDataHash_);

    return true;
}
