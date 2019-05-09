#include "stdafx.h"
#include "PathFilter.h"

#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "thermodyn/PathLineDataCall.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/StringParam.h"
#include "vislib/math/ShallowPoint.h"


megamol::thermodyn::PathFilter::PathFilter()
    : dataInSlot_("dataIn", "Input of particle pathlines")
    , dataOutSlot_("dataOut", "Output of filtered particle pathlines")
    , filterTypeSlot_("type", "Type of path filter")
    , filterAxisSlot_("axis", "Axis on which filter is applied")
    , filterThresholdSlot_("threshold", "Threshold of the filter")
    , maxIntSlot_("maxInt", "Max value of interface")
    , minIntSlot_("minInt", "Min value of interface")
    , timeCutSlot_("timeCut", "Crop the time dimension")
    , boxSlot_("box", "Box definition for the Box Filter (minx, miny, minz, maxx, maxy, maxz)")
    , percSlot_("percentage", "Percentage of path to pass") {
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
    ep->SetTypePair(static_cast<int>(FilterType::Plane), "Plane");
    ep->SetTypePair(static_cast<int>(FilterType::BoxFilter), "BoxFilter");
    ep->SetTypePair(static_cast<int>(FilterType::Hotness), "Hotness");
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

    timeCutSlot_ << new core::param::BoolParam(false);
    MakeSlotAvailable(&timeCutSlot_);
    
    boxSlot_ << new core::param::StringParam("0.0, 0.0, 0.0, 1.0, 1.0, 1.0");
    MakeSlotAvailable(&boxSlot_);

    percSlot_ << new core::param::FloatParam(10.0f, std::numeric_limits<float>::min(), 100.0f);
    MakeSlotAvailable(&percSlot_);
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
        auto const timeCut = timeCutSlot_.Param<core::param::BoolParam>()->Value();
        auto const perc = percSlot_.Param<core::param::FloatParam>()->Value();

        pathStore_ = *inCall->GetPathStore(); //< this is a copy
        pathFrameStore_ = *inCall->GetPathFrameStore();
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
                auto& fs = pathFrameStore_[plidx];
                for (auto const& idx : toRemove) {
                    ps.erase(idx);
                    fs.erase(idx);
                }
            }

        } break;
        case FilterType::Plane: {

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
                    std::vector<bool> planeCon(pathsize/entrySize, false);
                    for (size_t eidx = 0; eidx < pathsize; eidx += entrySize) {
                        bool planeCon1 = false;
                        bool planeCon2 = false;
                        if (std::fabs(p[eidx+dirOff+0]) >= filterThreshold) planeCon1=true;
                        if (std::fabs(p[eidx+dirOff+2]) >= filterThreshold) planeCon2=true;
                        planeCon[eidx/entrySize] = planeCon1&&planeCon2;
                    }
                    auto count = std::count(planeCon.begin(), planeCon.end(),true);
                    if (count > 0.9f*planeCon.size()) toRemove.push_back(path.first);
                }
                auto& ps = pathStore_[plidx];
                auto& fs = pathFrameStore_[plidx];
                for (auto const& idx : toRemove) {
                    ps.erase(idx);
                    fs.erase(idx);
                }
            }


        } break;
        case FilterType::BoxFilter: {
            auto const box = getBoxFromString(boxSlot_.Param<core::param::StringParam>()->Value());

            for (size_t plidx = 0; plidx < pathStore_.size(); ++plidx) {
                auto const hasDir = dirsPresent_[plidx];
                auto const hasCol = colsPresent_[plidx];
                auto const entrySize = entrySizes_[plidx];
                int dirOff = 3;
                if (hasCol) dirOff += 4;
                std::vector<size_t> toRemove;
                for (auto& path : pathStore_[plidx]) {
                    auto const pathsize = path.second.size();
                    auto& p = path.second;
                    bool inBox = false;
                    for (size_t eidx = 0; eidx < pathsize; eidx += entrySize) {
                        vislib::math::ShallowPoint<float, 3> const pivot(p.data()+eidx);
                        if (box.Contains(pivot)) {
                            inBox=true;
                            break;
                        }
                    }
                    if (!inBox) {
                        toRemove.push_back(path.first);
                    }
                }
                auto& ps = pathStore_[plidx];
                auto& fs = pathFrameStore_[plidx];
                for (auto const& idx : toRemove) {
                    ps.erase(idx);
                    fs.erase(idx);
                }
            }

        } break;
            case FilterType::Hotness: {
            auto const box = getBoxFromString(boxSlot_.Param<core::param::StringParam>()->Value());

            // assumes temperature exists

            for (size_t plidx = 0; plidx < pathStore_.size(); ++plidx) {
                auto const hasDir = dirsPresent_[plidx];
                auto const hasCol = colsPresent_[plidx];
                auto const entrySize = entrySizes_[plidx];
                int tempOff = 3;
                if (hasCol) tempOff += 4;
                if (hasDir) tempOff += 3;
                std::vector<size_t> toRemove;
                for (auto& path : pathStore_[plidx]) {
                    auto const pathsize = path.second.size();
                    auto& p = path.second;
                    bool inBox = false;
                    for (size_t eidx = 0; eidx < pathsize; eidx += entrySize) {
                        vislib::math::ShallowPoint<float, 3> const pivot(p.data()+eidx);
                        if (box.Contains(pivot)) {
                            inBox=true;
                            break;
                        }
                    }
                    if (!inBox) {
                        toRemove.push_back(path.first);
                    }
                }
                auto& ps = pathStore_[plidx];
                auto& fs = pathFrameStore_[plidx];
                for (auto const& idx : toRemove) {
                    ps.erase(idx);
                    fs.erase(idx);
                }
                // check average temp in interface region for remaining paths
                std::vector<std::pair<size_t, float>> temps;
                for (auto& path : pathStore_[plidx]) {
                    auto const pathsize = path.second.size();
                    auto& p = path.second;

                    std::pair<size_t, float> ret;
                    ret.first = path.first;
                    float val = std::numeric_limits<float>::lowest();

                    for (size_t eidx = 0; eidx < pathsize; eidx += entrySize) {
                        if (p[eidx+filterAxis] > minInt && p[eidx+filterAxis] < maxInt) {
                            val = std::max(val, p[eidx+tempOff]);
                        }
                    }
                    ret.second = val;
                    temps.push_back(ret);
                }
                std::sort(temps.begin(), temps.end(), [](auto const& a, auto const& b){return a.second > b.second;});
                size_t init = std::floorf(temps.size()*(perc/100.0f));
                for (size_t i = init; i < temps.size(); ++i) {
                    ps.erase(temps[i].first);
                    fs.erase(temps[i].first);
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
                auto& fs = pathFrameStore_[plidx];
                for (auto const& idx : toRemove) {
                    ps.erase(idx);
                    fs.erase(idx);
                }
            }
        }
        }
    }

    outCall->SetColorFlags(colsPresent_);
    outCall->SetDirFlags(dirsPresent_);
    outCall->SetEntrySizes(entrySizes_);
    outCall->SetPathStore(&pathStore_);
    outCall->SetPathFrameStore(&pathFrameStore_);
    outCall->SetTimeSteps(inCall->GetTimeSteps());
    outCall->SetDataHash(inDataHash_);

    outCall->AccessBoundingBoxes().SetObjectSpaceBBox(inCall->AccessBoundingBoxes().ObjectSpaceBBox());
    outCall->AccessBoundingBoxes().SetObjectSpaceClipBox(inCall->AccessBoundingBoxes().ObjectSpaceClipBox());
    outCall->AccessBoundingBoxes().MakeScaledWorld(1.0f);

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
