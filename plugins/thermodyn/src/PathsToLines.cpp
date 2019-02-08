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

    if (!(*inCall)(0)) return false;

    if (inCall->DataHash() != inDataHash_) {
        inDataHash_ = inCall->DataHash();

        linesStore_.clear();

        auto const& entrySizes = inCall->GetEntrySize();
        auto const& colsPresent = inCall->HasColors();
        auto const& dirsPresent = inCall->HasDirections();
        auto const& pathStore = *inCall->GetPathStore();

        for (size_t plidx = 0; plidx < pathStore.size(); ++plidx) {
            auto const& pathlines = pathStore[plidx];
            auto const entrysize = entrySizes[plidx];
            auto const colPresent = colsPresent[plidx];
            auto const dirPresent = dirsPresent[plidx];
            linesStore_.reserve(linesStore_.size() + pathlines.size());
            linesData_.reserve(linesData_.size() + pathlines.size());
            for (auto const& el : pathlines) {
                auto const& data = el.second;
                std::vector<float> line;
                line.reserve(data.size() / entrysize * 3);
                for (size_t idx = 0; idx < data.size(); idx += entrysize) {
                    line.push_back(data[idx + 0]);
                    line.push_back(data[idx + 1]);
                    line.push_back(data[idx + 2]);
                }
                linesData_.push_back(line);
                Lines l;
                l.Set(line.size()/3, linesData_.back().data(), {255, 255, 255, 255});
                linesStore_.push_back(l);
            }
        }
    }

    outCall->SetFrameCount(1);
    outCall->SetFrameID(0);
    outCall->SetData(linesStore_.size(), linesStore_.data());

    return true;
}


bool megamol::thermodyn::PathToLines::getExtentCallback(core::Call& c) {
    auto inCall = dataInSlot_.CallAs<PathLineDataCall>();
    if (inCall == nullptr) return false;

    auto outCall = dynamic_cast<geocalls::LinesDataCall*>(&c);
    if (outCall == nullptr) return false;

    if (!(*inCall)(1)) return false;

    outCall->SetFrameCount(1);
    outCall->SetFrameID(0);
    outCall->AccessBoundingBoxes().SetObjectSpaceBBox(inCall->GetBoundingBoxes().ObjectSpaceBBox());
    outCall->AccessBoundingBoxes().SetObjectSpaceClipBox(inCall->GetBoundingBoxes().ObjectSpaceClipBox());
    outCall->AccessBoundingBoxes().MakeScaledWorld(1.0f);

    return true;
}
