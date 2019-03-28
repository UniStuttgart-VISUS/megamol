#include "stdafx.h"
#include "PathIColClustering.h"

#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "thermodyn/PathLineDataCall.h"

#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmstd_datatools/DBSCAN.h"
#include "vislib/math/ShallowPoint.h"


megamol::thermodyn::PathIColClustering::PathIColClustering()
    : pathsInSlot_("pathIn", "Input of paths")
    , particleInSlot_("particleIn", "Input of particle data")
    , dataOutSlot_("dataOut", "Output of selected paths")
    , minPtsSlot_("minPts", "MinPts param for DBSCAN")
    , sigmaSlot_("sigma", "Sigma param for DBSCAN")
    , thresholdSlot_("threshold", "Temperature threshold in percentage")
    , similaritySlot_("similarity", "Similarity threshold regarding temperature for clustering in percentage") {
    pathsInSlot_.SetCompatibleCall<PathLineDataCallDescription>();
    MakeSlotAvailable(&pathsInSlot_);

    particleInSlot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&particleInSlot_);

    dataOutSlot_.SetCallback(
        PathLineDataCall::ClassName(), PathLineDataCall::FunctionName(0), &PathIColClustering::getDataCallback);
    dataOutSlot_.SetCallback(
        PathLineDataCall::ClassName(), PathLineDataCall::FunctionName(1), &PathIColClustering::getExtentCallback);
    MakeSlotAvailable(&dataOutSlot_);

    minPtsSlot_ << new core::param::IntParam(1, 1, std::numeric_limits<int>::max());
    MakeSlotAvailable(&minPtsSlot_);

    sigmaSlot_ << new core::param::FloatParam(
        0.5f, std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
    MakeSlotAvailable(&sigmaSlot_);

    thresholdSlot_ << new core::param::FloatParam(0.5f, 0.0f, 1.0f);
    MakeSlotAvailable(&thresholdSlot_);

    similaritySlot_ << new core::param::FloatParam(0.1f, 0.0f, 1.0f);
    MakeSlotAvailable(&similaritySlot_);
}


megamol::thermodyn::PathIColClustering::~PathIColClustering() { this->Release(); }


bool megamol::thermodyn::PathIColClustering::create() { return true; }


void megamol::thermodyn::PathIColClustering::release() {}


bool megamol::thermodyn::PathIColClustering::getDataCallback(core::Call& c) {
    auto inPathsCall = pathsInSlot_.CallAs<PathLineDataCall>();
    if (inPathsCall == nullptr) return false;

    auto inParCall = particleInSlot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (inParCall == nullptr) return false;

    auto outCall = dynamic_cast<PathLineDataCall*>(&c);
    if (outCall == nullptr) return false;

    auto const frameID = outCall->FrameID();

    inParCall->SetFrameID(frameID, true);
    inPathsCall->SetFrameID(frameID, true);
    if (!(*inPathsCall)(0)) return false;
    if (!(*inParCall)(0)) return false;

    if (inPathsCall->DataHash() != inPathsHash_ || frameID != frameID_ || isDirty()) {
        inPathsHash_ = inPathsCall->DataHash();
        inParHash_ = inParCall->DataHash();
        frameID_ = inParCall->FrameID();
        ++outDataHash_;

        resetDirty();

        pathStore_ = *inPathsCall->GetPathStore();
        pathFrameStore_ = *inPathsCall->GetPathFrameStore();
        auto const& entrySizes = inPathsCall->GetEntrySize();
        auto const& dirsPresent = inPathsCall->HasDirections();
        auto const& colsPresent = inPathsCall->HasColors();

        auto const sigma = sigmaSlot_.Param<core::param::FloatParam>()->Value();
        auto const minPts = minPtsSlot_.Param<core::param::IntParam>()->Value();

        auto const plc = pathStore_.size();
        if (plc != inParCall->GetParticleListCount()) {
            vislib::sys::Log::DefaultLog.WriteError(
                "PathIColClustering: Particle list count differs between path input and particle input\n");
            return false;
        }

        auto const bbox = inParCall->AccessBoundingBoxes().ObjectSpaceBBox();

        for (unsigned int plidx = 0; plidx < plc; ++plidx) {
            auto const& parts = inParCall->AccessParticles(plidx);
            auto& paths = pathStore_[plidx];
            auto& frames = pathFrameStore_[plidx];

            auto const threshold = thresholdSlot_.Param<core::param::FloatParam>()->Value() *
                                       (parts.GetMaxColourIndexValue() - parts.GetMinColourIndexValue()) +
                                   parts.GetMinColourIndexValue();

            auto const similarity = similaritySlot_.Param<core::param::FloatParam>()->Value() *
                                        (parts.GetMaxColourIndexValue() - parts.GetMinColourIndexValue()) +
                                    parts.GetMinColourIndexValue();

            auto data = preparePoints(parts);
            auto const numPoints = parts.GetCount();

            auto const rad = parts.GetGlobalRadius();

            auto const entrysize = entrySizes[plidx];

            // cluster particle data for requested frame within requested area
            using DB = stdplugin::datatools::HDBSCAN<float, true, 3, false>;

            DB scanner(numPoints, 4, data, bbox, minPts, sigma,
                similarity); //< TODO: Probably need to resolve periodic boundary conditions
            auto clusters = scanner.Scan();

            // select hotspot clusters
            // probably need series of min spheres
            std::vector<std::array<float, 4>> min_spheres;
            min_spheres.reserve(clusters.size());
            for (size_t cidx = 0; cidx < clusters.size(); ++cidx) {
                auto temp = getTemperatureAvg(clusters[cidx]);
                if (temp >= threshold) {
                    auto spr = replaceTempInPoints(clusters[cidx], rad);
                    min_spheres.push_back(getMinSphere(spr));
                }
            }

            // expose path that cross cluster at given timestep
            std::vector<size_t> to_remove;
            to_remove.reserve(paths.size());

            for (auto& el : paths) {
                auto it = frames.find(el.first);
                if (it == frames.end()) {
                    // unexpected behavior
                    to_remove.push_back(el.first);
                    continue;
                }
                auto fpos = std::find(it->second.begin(), it->second.end(), frameID);
                if (fpos == it->second.end()) {
                    to_remove.push_back(el.first);
                    continue;
                }
                auto fidx = std::distance(it->second.begin(), fpos);
                vislib::math::ShallowPoint<float, 3> pt(&el.second[fidx * entrysize]);
                bool check = false;
                for (auto& cl : min_spheres) {
                    vislib::math::ShallowPoint<float, 3> spr(cl.data());
                    if (spr.Distance(pt) <= cl[3]) {
                        check = true;
                        break;
                    }
                }
                if (!check) to_remove.push_back(el.first);
            }
            for (auto const& el : to_remove) {
                paths.erase(el);
                frames.erase(el);
            }
        }
    }

    outCall->SetPathStore(&pathStore_);
    outCall->SetPathFrameStore(&pathFrameStore_);
    outCall->SetEntrySizes(inPathsCall->GetEntrySize());
    outCall->SetDirFlags(inPathsCall->HasDirections());
    outCall->SetColorFlags(inPathsCall->HasColors());
    outCall->SetTimeSteps(inPathsCall->GetTimeSteps());

    outCall->AccessBoundingBoxes().SetObjectSpaceBBox(inPathsCall->AccessBoundingBoxes().ObjectSpaceBBox());
    outCall->AccessBoundingBoxes().SetObjectSpaceClipBox(inPathsCall->AccessBoundingBoxes().ObjectSpaceClipBox());
    outCall->AccessBoundingBoxes().MakeScaledWorld(1.0f);

    outCall->SetFrameCount(inParCall->FrameCount());
    outCall->SetFrameID(frameID_);

    return true;
}


bool megamol::thermodyn::PathIColClustering::getExtentCallback(core::Call& c) {
    auto inPathsCall = pathsInSlot_.CallAs<PathLineDataCall>();
    if (inPathsCall == nullptr) return false;

    auto inParCall = particleInSlot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (inParCall == nullptr) return false;

    auto outCall = dynamic_cast<PathLineDataCall*>(&c);
    if (outCall == nullptr) return false;

    if (!(*inPathsCall)(1)) return false;
    if (!(*inParCall)(1)) return false;


    outCall->AccessBoundingBoxes().SetObjectSpaceBBox(inPathsCall->AccessBoundingBoxes().ObjectSpaceBBox());
    outCall->AccessBoundingBoxes().SetObjectSpaceClipBox(inPathsCall->AccessBoundingBoxes().ObjectSpaceClipBox());
    outCall->AccessBoundingBoxes().MakeScaledWorld(1.0f);

    outCall->SetFrameCount(inParCall->FrameCount());
    // outCall->SetFrameID(frameID_);

    outCall->SetDataHash(outDataHash_);

    return true;
}
