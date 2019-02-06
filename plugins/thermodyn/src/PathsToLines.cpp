#include "stdafx.h"
#include "PathsToLines.h"

#include "thermodyn/PathLineDataCall.h"


megamol::thermodyn::PathToLines::PathToLines()
    : dataInSlot_("dataIn", "Input of pathlines"), dataOutSlot_("dataOut", "Output of lines") {
    dataInSlot_.SetCompatibleCall<PathLineDataCallDescription>();
    MakeSlotAvailable(&dataInSlot_);

    dataOutSlot_.SetCallback(
        geocalls::LinesDataCall::ClassName(), geocalls::LinesDataCall::FunctionName(0), &PathToLines::getDataCallback);
    dataOutSlot_.SetCallback(geocalls::LinesDataCall::ClassName(), geocalls::LinesDataCall::FunctionName(1),
        &PathToLines::getExtentCallback);
    MakeSlotAvailable(&dataOutSlot_);
}


megamol::thermodyn::PathToLines::~PathToLines() { this->Release(); }


bool megamol::thermodyn::PathToLines::create() { return true; }


void megamol::thermodyn::PathToLines::release() {}


bool megamol::thermodyn::PathToLines::getDataCallback(core::Call& c) {
    auto inCall = dataInSlot_.CallAs<PathLineDataCall>();
    if (inCall == nullptr) return false;

    auto outCall = dynamic_cast<geocalls::LinesDataCall*>(&c);
    if (outCall == nullptr) return false;

    if (inCall->DataHash() != inDataHash_) {
        inDataHash_ = inCall->DataHash();

        linesStore_.clear();

        auto const entrySize = inCall->GetEntrySize();
        auto const pathStore = inCall->GetPathStore();

        for (auto const& pl : *pathStore) {
            linesStore_.reserve(linesStore_.size() + pl.size());
            for (auto const& el : pl) {
                
            }
        }
    }

    return true;
}


bool megamol::thermodyn::PathToLines::getExtentCallback(core::Call& c) { return true; }
