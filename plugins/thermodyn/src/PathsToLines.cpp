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

    inCall->SetFrameID(outCall->FrameID(), true);
    if (!(*inCall)(0)) return false;

    if (inCall->DataHash() != inDataHash_ || inCall->FrameID() != frameID_) {
        inDataHash_ = inCall->DataHash();
        frameID_ = inCall->FrameID();

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

            //f (pathlines.empty()) continue;

            linesStore_.reserve(linesStore_.size() + pathlines.size());
            linesData_.reserve(linesData_.size() + pathlines.size());
            if (colPresent) {
                colsData_.reserve(colsData_.size() + pathlines.size());
            }
            for (auto const& el : pathlines) {
                auto const& data = el.second;
                std::vector<float> line;
                std::vector<float> col;
                line.reserve(data.size() / entrysize * 3);
                if (colPresent) col.reserve(data.size() / entrysize * 4);
                for (size_t idx = 0; idx < data.size(); idx += entrysize) {
                    line.push_back(data[idx + 0]);
                    line.push_back(data[idx + 1]);
                    line.push_back(data[idx + 2]);
                    if (colPresent) {
                        col.push_back(data[idx + 3]);
                        col.push_back(data[idx + 4]);
                        col.push_back(data[idx + 5]);
                        col.push_back(data[idx + 6]);
                    }
                }
                linesData_.push_back(line);
                Lines l;
                if (!colPresent) {
                    l.Set(line.size()/3, linesData_.back().data(), {255, 255, 255, 255});
                } else {
                    colsData_.push_back(col);
                    l.Set(line.size()/3, linesData_.back().data(), colsData_.back().data(), true);
                    //l.Set(line.size()/3, linesData_.back().data(), {255, 255, 255, 255});
                }
                linesStore_.push_back(l);
            }
        }
    }

    outCall->SetFrameCount(inCall->FrameCount());
    outCall->SetFrameID(frameID_);
    outCall->SetData(linesStore_.size(), linesStore_.data());

    return true;
}


bool megamol::thermodyn::PathToLines::getExtentCallback(core::Call& c) {
    auto inCall = dataInSlot_.CallAs<PathLineDataCall>();
    if (inCall == nullptr) return false;

    auto outCall = dynamic_cast<geocalls::LinesDataCall*>(&c);
    if (outCall == nullptr) return false;

    if (!(*inCall)(1)) return false;

    outCall->SetFrameCount(inCall->FrameCount());
    //outCall->SetFrameID(inCall->FrameID());
    outCall->AccessBoundingBoxes().SetObjectSpaceBBox(inCall->GetBoundingBoxes().ObjectSpaceBBox());
    outCall->AccessBoundingBoxes().SetObjectSpaceClipBox(inCall->GetBoundingBoxes().ObjectSpaceClipBox());
    outCall->AccessBoundingBoxes().MakeScaledWorld(1.0f);

    return true;
}
