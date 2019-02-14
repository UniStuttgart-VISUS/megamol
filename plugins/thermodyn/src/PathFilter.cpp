#include "stdafx.h"
#include "PathFilter.h"

#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "thermodyn/PathLineDataCall.h"


megamol::thermodyn::PathFilter::PathFilter()
    : dataInSlot_("dataIn", "Input of particle pathlines")
    , dataOutSlot_("dataOut", "Output of filtered particle pathlines")
    , filterTypeSlot_("type", "Type of path filter")
    , filterAxisSlot_("axis", "Axis on which filter is applied")
    , filterThresholdSlot_("threshold", "Threshold of the filter")
    , maxIntSlot_("maxInt", "Max value of interface")
    , minIntSlot_("minInt", "Min value of interface") {
    dataInSlot_.SetCompatibleCall<PathLineDataCallDescription>();
    MakeSlotAvailable(&dataInSlot_);

    dataOutSlot_.SetCallback(
        PathLineDataCall::ClassName(), PathLineDataCall::FunctionName(0), &PathFilter::getDataCallback);
    dataOutSlot_.SetCallback(
        PathLineDataCall::ClassName(), PathLineDataCall::FunctionName(1), &PathFilter::getExtentCallback);
    MakeSlotAvailable(&dataOutSlot_);

    auto ep = new core::param::EnumParam(static_cast<int>(FilterType::MainDirection));
    ep->SetTypePair(static_cast<int>(FilterType::MainDirection), "MainDirection");
    ep->SetTypePair(static_cast<int>(FilterType::Interface),"Interface");
    filterTypeSlot_ << ep;
    MakeSlotAvailable(&filterTypeSlot_);

    ep = new core::param::EnumParam(0);
    ep->SetTypePair(0, "x");
    ep->SetTypePair(1, "y");
    ep->SetTypePair(2, "z");
    filterAxisSlot_ << ep;
    MakeSlotAvailable(&filterAxisSlot_);

    filterThresholdSlot_ << new core::param::FloatParam(
        0.0f, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());
    MakeSlotAvailable(&filterThresholdSlot_);

    maxIntSlot_ << new core::param::FloatParam(
        0.0f, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());
    MakeSlotAvailable(&maxIntSlot_);

    minIntSlot_ << new core::param::FloatParam(
        0.0f, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());
    MakeSlotAvailable(&minIntSlot_);
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

        auto const filterType = filterTypeSlot_.Param<core::param::EnumParam>()->Value();
        auto const filterAxis = filterAxisSlot_.Param<core::param::EnumParam>()->Value();
        auto const filterThreshold = filterThresholdSlot_.Param<core::param::FloatParam>()->Value();
        auto const minInt = minIntSlot_.Param<core::param::FloatParam>()->Value();
        auto const maxInt = maxIntSlot_.Param<core::param::FloatParam>()->Value();

        pathStore_ = *inCall->GetPathStore(); //< this is a copy
        dirsPresent_ = inCall->HasDirections();
        colsPresent_ = inCall->HasColors();
        entrySizes_ = inCall->GetEntrySize();

        switch (filterType) {
        case FilterType::Interface: {
            
            for (size_t plidx = 0; plidx < pathStore_.size(); ++plidx) {
                auto const hasDir = dirsPresent_[plidx];
                auto const hasCol = colsPresent_[plidx];
                auto const entrySize = entrySizes_[plidx];
                int dirOff = 3;
                if (hasCol) dirOff += 4;
                std::vector<size_t> toRemove;
                for (auto const& path : pathStore_[plidx]) {
                    auto const pathsize = path.second.size();
                    auto const& p = path.second;
                    bool leftInt1 = false;
                    bool rightInt1 = false;
                    bool leftInt2 = false;
                    bool rightInt2 = false;
                    for (size_t eidx = 0; eidx < pathsize; eidx += entrySize) {
                        if (p[eidx + filterAxis] < minInt) leftInt1 = true;
                        if (p[eidx + filterAxis] > minInt) rightInt1 = true;
                        if (p[eidx + filterAxis] < maxInt) leftInt2 = true;
                        if (p[eidx + filterAxis] > maxInt) rightInt2 = true;
                    }
                    if (!(leftInt1 && rightInt1) && !(leftInt2 && rightInt2)) {
                        toRemove.push_back(path.first);
                    }
                }
                auto& ps = pathStore_[plidx];
                for (auto const& idx : toRemove) {
                    ps.erase(idx);
                }
            }

        } break;
        case FilterType::MainDirection:
        default: {

            for (size_t plidx = 0; plidx < pathStore_.size(); ++plidx) {
                auto const hasDir = dirsPresent_[plidx];
                auto const hasCol = colsPresent_[plidx];
                auto const entrySize = entrySizes_[plidx];
                int dirOff = 3;
                if (hasCol) dirOff += 4;
                std::vector<size_t> toRemove;
                for (auto const& path : pathStore_[plidx]) {
                    auto const pathsize = path.second.size();
                    auto const& p = path.second;
                    bool posDir = false;
                    bool negDir = false;
                    for (size_t eidx = 0; eidx < pathsize; eidx += entrySize) {
                        if (p[eidx + dirOff + filterAxis] >= filterThreshold) posDir = true;
                        if (p[eidx + dirOff + filterAxis] <= -filterThreshold) negDir = true;
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
        }
    }

    outCall->SetColorFlags(colsPresent_);
    outCall->SetDirFlags(dirsPresent_);
    outCall->SetEntrySizes(entrySizes_);
    outCall->SetPathStore(&pathStore_);
    outCall->SetTimeSteps(inCall->GetTimeSteps());

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
