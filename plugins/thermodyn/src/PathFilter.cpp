#include "stdafx.h"
#include "PathFilter.h"

#include "thermodyn/PathLineDataCall.h"


megamol::thermodyn::PathFilter::PathFilter()
    : dataInSlot_("dataIn", "Input of particle pathlines")
    , dataOutSlot_("dataOut", "Output of filtered particle pathlines") {
    dataInSlot_.SetCompatibleCall<PathLineDataCallDescription>();
    MakeSlotAvailable(&dataInSlot_);

    dataOutSlot_.SetCallback(
        PathLineDataCall::ClassName(), PathLineDataCall::FunctionName(0), &PathFilter::getDataCallback);
    dataOutSlot_.SetCallback(
        PathLineDataCall::ClassName(), PathLineDataCall::FunctionName(1), &PathFilter::getExtentCallback);
    MakeSlotAvailable(&dataOutSlot_);
}


megamol::thermodyn::PathFilter::~PathFilter() { this->Release(); }


bool megamol::thermodyn::PathFilter::create() { return true; }


void megamol::thermodyn::PathFilter::release() {}


bool megamol::thermodyn::PathFilter::getDataCallback(core::Call& c) {
    auto outCall = dynamic_cast<PathLineDataCall*>(&c);
    if (outCall == nullptr) return false;

    auto inCall = dataInSlot_.CallAs<PathLineDataCall>();
    if (inCall == nullptr) return false;

    if (!(*inCall)(0)) return false;

    if (inCall->DataHash() != inDataHash_) {
        inDataHash_ = inCall->DataHash();

        pathStore_ = *inCall->GetPathStore(); //< this is a copy
        dirsPresent_ = inCall->HasDirections();
        colsPresent_ = inCall->HasColors();
        entrySizes_ = inCall->GetEntrySize();

        for (size_t plidx = 0; plidx < pathStore_.size(); ++plidx) {
            bool hasDir = dirsPresent_[plidx];
            bool hasCol = colsPresent_[plidx];
            auto entrySize = entrySizes_[plidx];
            int dirOff = 3;
            if (hasCol) dirOff += 4;
            std::vector<size_t> toRemove;
            for (auto const& path : pathStore_[plidx]) {
                auto pathsize = path.second.size();
                auto const& p = path.second;
                bool posDir = false;
                bool negDir = false;
                for (size_t eidx = 0; eidx < pathsize; eidx += entrySize) {
                    if (p[eidx + dirOff + 1] >= 0.5f) posDir = true;
                    if (p[eidx + dirOff + 1] <= -0.5f) negDir = true;
                }
                if (!(posDir && negDir)) {
                    toRemove.push_back(path.first);
                }
            }
            auto& ps = pathStore_[plidx];
            for (auto const& idx : toRemove) {
                ps.erase(idx);
            }
        }
    }

    outCall->SetColorFlags(colsPresent_);
    outCall->SetDirFlags(dirsPresent_);
    outCall->SetEntrySizes(entrySizes_);
    outCall->SetPathStore(&pathStore_);

    return true;
}


bool megamol::thermodyn::PathFilter::getExtentCallback(core::Call& c) {
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
