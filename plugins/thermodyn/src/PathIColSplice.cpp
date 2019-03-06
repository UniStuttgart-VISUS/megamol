#include "stdafx.h"
#include "PathIColSplice.h"

#include "mmcore/moldyn/MultiParticleDataCall.h"

#include "thermodyn/PathLineDataCall.h"


megamol::thermodyn::PathIColSplice::PathIColSplice()
    : pathsInSlot_("pathsIn", "Input of particle pathlines")
    , icolInSlot_("icolIn", "Input of particles with ICol")
    , dataOutSlot_("dataOut", "Output of particle pathlines") {
    pathsInSlot_.SetCompatibleCall<PathLineDataCallDescription>();
    MakeSlotAvailable(&pathsInSlot_);

    icolInSlot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&icolInSlot_);

    dataOutSlot_.SetCallback(PathLineDataCall::ClassName(), PathLineDataCall::FunctionName(0), &PathIColSplice::getDataCallback);
    dataOutSlot_.SetCallback(PathLineDataCall::ClassName(), PathLineDataCall::FunctionName(1), &PathIColSplice::getExtentCallback);
    MakeSlotAvailable(&dataOutSlot_);
}


megamol::thermodyn::PathIColSplice::~PathIColSplice() {
    this->Release();
}


bool megamol::thermodyn::PathIColSplice::create() {
    return true;
}


void megamol::thermodyn::PathIColSplice::release() {

}


bool megamol::thermodyn::PathIColSplice::getDataCallback(core::Call& c) {
    auto inPathsCall = pathsInSlot_.CallAs<PathLineDataCall>();
    if (inPathsCall == nullptr) return false;

    auto inIColCall = icolInSlot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (inIColCall == nullptr) return false;

    auto outCall = dynamic_cast<PathLineDataCall*>(&c);
    if (outCall == nullptr) return false;

    if (!(*inPathsCall)(0)) return false;
    if (!(*inIColCall)(0)) return false;

    if (inPathsCall->DataHash() != inPathsHash_ /* || inIColCall->DataHash() != inIColHash_*/) {
        inPathsHash_ = inPathsCall->DataHash();
        inIColHash_ = inIColCall->DataHash();
        ++outDataHash_;

        auto const pathsFrameCount = inPathsCall->GetTimeSteps();
        auto const icolFrameCount = inIColCall->FrameCount();

        size_t frameSkip = 1;

        if (pathsFrameCount != icolFrameCount) {
            //vislib::sys::Log::DefaultLog.WriteError("PathIColSplice: Framecounts of inputs do not match\n");
            //return false;
            frameSkip = icolFrameCount/pathsFrameCount;
        }

        outPathStore_             = *inPathsCall->GetPathStore();
        auto const& entrySizes    = inPathsCall->GetEntrySize();
        auto const& inDirsPresent = inPathsCall->HasDirections();
        auto const& inColsPresent = inPathsCall->HasColors();

        if (outPathStore_.size() != inIColCall->GetParticleListCount()) {
            vislib::sys::Log::DefaultLog.WriteError("PathIColSplice: Particlelistcounts of inputs do not match\n");
            return false;
        }

        outEntrySizes_.resize(outPathStore_.size());
        outDirsPresent_.resize(outPathStore_.size());
        outColsPresent_.resize(outPathStore_.size());

        for (size_t plidx = 0; plidx < outPathStore_.size(); ++plidx) {

            outEntrySizes_[plidx] = entrySizes[plidx]+1;
            outDirsPresent_[plidx] = inDirsPresent[plidx];
            outColsPresent_[plidx] = inColsPresent[plidx];

            auto const& particlestore = inIColCall->AccessParticles(plidx).GetParticleStore();
            auto& idAcc = particlestore.GetIDAcc();
            auto& icolAcc = particlestore.GetCRAcc();
            auto const parCount = inIColCall->AccessParticles(plidx).GetCount();

            auto stride = 3;
            if (inColsPresent[plidx]) stride += 4;
            if (inDirsPresent[plidx]) stride += 3;

            auto& paths = outPathStore_[plidx];

            for (auto& el : paths) {
                el.second = enlargeVector(el.second, stride);
            }

            for (size_t fidx = 0; fidx < pathsFrameCount; ++fidx) {
                do {
                    inIColCall->SetFrameID(fidx * frameSkip, true);
                    (*inIColCall)(1);
                } while (fidx * frameSkip != inIColCall->FrameID());

                if (!(*inIColCall)(0)) return false;

                for (size_t pidx = 0; pidx < parCount; ++pidx) {
                    auto const idx = idAcc->Get_u64(pidx);
                    auto const temp = icolAcc->Get_f(pidx);
                    PathLineDataCall::pathline_store_t::iterator it;
                    if ((it = paths.find(idx)) != paths.end()) {
                        //auto& pathline = paths[idx];
                        auto& pathline = it->second;
                        pathline[fidx * (stride + 1) + stride] = temp;
                    }
                }
            }
        }
    }

    outCall->SetPathStore(&outPathStore_);
    outCall->SetColorFlags(outColsPresent_);
    outCall->SetDirFlags(outDirsPresent_);
    outCall->SetEntrySizes(outEntrySizes_);
    outCall->SetTimeSteps(inPathsCall->GetTimeSteps());

    outCall->AccessBoundingBoxes().SetObjectSpaceBBox(inPathsCall->AccessBoundingBoxes().ObjectSpaceBBox());
    outCall->AccessBoundingBoxes().SetObjectSpaceClipBox(inPathsCall->AccessBoundingBoxes().ObjectSpaceClipBox());
    outCall->AccessBoundingBoxes().MakeScaledWorld(1.0f);

    return true;
}


bool megamol::thermodyn::PathIColSplice::getExtentCallback(core::Call& c) {
    auto inPathsCall = pathsInSlot_.CallAs<PathLineDataCall>();
    if (inPathsCall == nullptr) return false;

    auto inIColCall = icolInSlot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (inIColCall == nullptr) return false;

    auto outCall = dynamic_cast<PathLineDataCall*>(&c);
    if (outCall == nullptr) return false;

    if (!(*inPathsCall)(1)) return false;
    if (!(*inIColCall)(1)) return false;


    outCall->AccessBoundingBoxes().SetObjectSpaceBBox(inPathsCall->AccessBoundingBoxes().ObjectSpaceBBox());
    outCall->AccessBoundingBoxes().SetObjectSpaceClipBox(inPathsCall->AccessBoundingBoxes().ObjectSpaceClipBox());
    /*outCall->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox_);
    outCall->AccessBoundingBoxes().SetObjectSpaceClipBox(this->bbox_);*/
    outCall->AccessBoundingBoxes().MakeScaledWorld(1.0f);


    outCall->SetFrameCount(1);
    outCall->SetFrameID(0);

    outCall->SetDataHash(outDataHash_);

    return true;
}
