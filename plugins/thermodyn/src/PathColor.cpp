#include "stdafx.h"
#include "PathColor.h"

#include "thermodyn/PathLineDataCall.h"


megamol::thermodyn::PathColor::PathColor()
    : dataInSlot_("dataIn", "Input of particle pathlines"), dataOutSlot_("dataOut", "Output of particle pathlines") {
    dataInSlot_.SetCompatibleCall<PathLineDataCallDescription>();
    MakeSlotAvailable(&dataInSlot_);

    dataOutSlot_.SetCallback(
        PathLineDataCall::ClassName(), PathLineDataCall::FunctionName(0), &PathColor::getDataCallback);
    dataOutSlot_.SetCallback(
        PathLineDataCall::ClassName(), PathLineDataCall::FunctionName(1), &PathColor::getExtentCallback);
    MakeSlotAvailable(&dataOutSlot_);
}


megamol::thermodyn::PathColor::~PathColor() { this->Release(); }


bool megamol::thermodyn::PathColor::create() { return true; }


void megamol::thermodyn::PathColor::release() {}


bool megamol::thermodyn::PathColor::getDataCallback(core::Call& c) {
    auto outCall = dynamic_cast<PathLineDataCall*>(&c);
    if (outCall == nullptr) return false;

    auto inCall = dataInSlot_.CallAs<PathLineDataCall>();
    if (inCall == nullptr) return false;

    if (!(*inCall)(0)) return false;

    if (inCall->DataHash() != inDataHash_) {
        inDataHash_ = inCall->DataHash();

        entrySizes_ = inCall->GetEntrySize();
        auto const& dirsPresent = inCall->HasDirections();
        colsPresent_ = inCall->HasColors();
        pathStore_ = *inCall->GetPathStore();
        auto const ts = inCall->GetTimeSteps();

        float const start_col[] = {1.0f, 0.0f, 0.0f};
        float const end_col[] = {0.0f, 1.0f, 0.0f};

        for (size_t plidx = 0; plidx < pathStore_.size(); ++plidx) {
            auto const hasDir = dirsPresent[plidx];
            auto const hasCol = colsPresent_[plidx];
            auto const entrySize = entrySizes_[plidx];
            int dirOff = 3;
            if (hasCol) dirOff += 4;
            int newEntrySize = 7;
            if (hasDir) newEntrySize = 10;
            std::vector<float> newPath(newEntrySize * ts);
            for (auto& path : pathStore_[plidx]) {
                auto const& oldPath = path.second;
                auto const pathsize = oldPath.size();
                for (size_t fidx = 0; fidx < ts; ++fidx) {
                    newPath[fidx * newEntrySize + 0] = oldPath[fidx*entrySize + 0];
                    newPath[fidx * newEntrySize + 1] = oldPath[fidx*entrySize + 1];
                    newPath[fidx * newEntrySize + 2] = oldPath[fidx*entrySize + 2];
                    newPath[fidx * newEntrySize + 3] =
                        1.0f - (static_cast<float>(fidx)) / static_cast<float>(ts);
                    newPath[fidx * newEntrySize + 4] =
                        (static_cast<float>(fidx)) / static_cast<float>(ts);
                    newPath[fidx * newEntrySize + 5] = 0.0f;
                    newPath[fidx * newEntrySize + 6] = 1.0f;
                    if (hasDir) {
                        newPath[fidx * newEntrySize + 7] = oldPath[fidx*entrySize + dirOff + 0];
                        newPath[fidx * newEntrySize + 8] = oldPath[fidx*entrySize + dirOff + 1];
                        newPath[fidx * newEntrySize + 9] = oldPath[fidx*entrySize + dirOff + 2];
                    }
                }
                path.second = newPath;
            }
            colsPresent_[plidx] = true;
            entrySizes_[plidx] = newEntrySize;
        }
    }

    outCall->SetColorFlags(colsPresent_);
    outCall->SetDirFlags(inCall->HasDirections());
    outCall->SetEntrySizes(entrySizes_);
    outCall->SetPathStore(&pathStore_);
    outCall->SetTimeSteps(inCall->GetTimeSteps());

    return true;
}


bool megamol::thermodyn::PathColor::getExtentCallback(core::Call& c) {
    auto outCall = dynamic_cast<PathLineDataCall*>(&c);
    if (outCall == nullptr) return false;

    auto inCall = dataInSlot_.CallAs<PathLineDataCall>();
    if (inCall == nullptr) return false;

    if (!(*inCall)(1)) return false;


    outCall->AccessBoundingBoxes().SetObjectSpaceBBox(inCall->AccessBoundingBoxes().ObjectSpaceBBox());
    outCall->AccessBoundingBoxes().SetObjectSpaceClipBox(inCall->AccessBoundingBoxes().ObjectSpaceClipBox());
    outCall->AccessBoundingBoxes().MakeScaledWorld(1.0f);


    outCall->SetFrameCount(1);

    outCall->SetDataHash(inDataHash_);

    return true;
}
