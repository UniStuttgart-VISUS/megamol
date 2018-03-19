#include "stdafx.h"
#include "TrajectoryAnimator.h"

#include <limits>

#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/FloatParam.h"

#include "geometry_calls/LinesDataCall.h"

#include "vislib/sys/Log.h"


megamol::stdplugin::datatools::TrajectoryAnimator::TrajectoryAnimator(void)
    : megamol::core::Module()
    , linesOutSlot("linesOut", "Output of transition trajectories")
    , pointsOutSlot("pointOut", "Output of points which highlight trajectories")
    , linesInSlot("linesIn", "Input of trajectories")
    , pointsInSlot("pointsIn", "Input of points to enable interpolation")
    , animationFactorSlot("animationFactor", "Steers the slow down of the animation wrt the framecount")
    , inFrameCountSlot("frameCount", "Set number of input frames")
    , globalRadiusSlot("globalRadius", "Set the global radius")
    , inFrameCount{0}
    , outFrameCount{0}
    , datahash{std::numeric_limits<size_t>::max()}
    , frameID{std::numeric_limits<unsigned int>::max()}
    , startFrameID{std::numeric_limits<unsigned int>::max()}
    , endFrameID{std::numeric_limits<unsigned int>::max()} {
    this->linesOutSlot.SetCallback(megamol::geocalls::LinesDataCall::ClassName(),
        megamol::geocalls::LinesDataCall::FunctionName(0), &TrajectoryAnimator::getLinesDataCallback);
    this->linesOutSlot.SetCallback(megamol::geocalls::LinesDataCall::ClassName(),
        megamol::geocalls::LinesDataCall::FunctionName(1), &TrajectoryAnimator::getLinesExtentCallback);
    this->MakeSlotAvailable(&this->linesOutSlot);

    this->pointsOutSlot.SetCallback(megamol::core::moldyn::MultiParticleDataCall::ClassName(),
        megamol::core::moldyn::MultiParticleDataCall::FunctionName(0), &TrajectoryAnimator::getPointsDataCallback);
    this->pointsOutSlot.SetCallback(megamol::core::moldyn::MultiParticleDataCall::ClassName(),
        megamol::core::moldyn::MultiParticleDataCall::FunctionName(1), &TrajectoryAnimator::getPointsExtentCallback);
    this->MakeSlotAvailable(&this->pointsOutSlot);

    this->linesInSlot.SetCompatibleCall<megamol::geocalls::LinesDataCallDescription>();
    this->MakeSlotAvailable(&this->linesInSlot);

    this->pointsInSlot.SetCompatibleCall<megamol::core::moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->pointsInSlot);

    this->animationFactorSlot << new megamol::core::param::IntParam(1, 1, 1000);
    this->MakeSlotAvailable(&this->animationFactorSlot);

    this->inFrameCountSlot << new megamol::core::param::IntParam(0, 0, std::numeric_limits<int>::max());
    this->MakeSlotAvailable(&this->inFrameCountSlot);

    this->globalRadiusSlot << new megamol::core::param::FloatParam(0.5f, std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
    this->MakeSlotAvailable(&this->globalRadiusSlot);
}


megamol::stdplugin::datatools::TrajectoryAnimator::~TrajectoryAnimator(void) { this->Release(); }


bool megamol::stdplugin::datatools::TrajectoryAnimator::create(void) { return true; }


void megamol::stdplugin::datatools::TrajectoryAnimator::release(void) {}


bool megamol::stdplugin::datatools::TrajectoryAnimator::getLinesDataCallback(megamol::core::Call& c) {
    if (!this->assertData(c)) return false;

    megamol::geocalls::LinesDataCall* linesInCall = this->linesInSlot.CallAs<megamol::geocalls::LinesDataCall>();
    if (linesInCall == nullptr) return false;

    megamol::geocalls::LinesDataCall* linesOutCall = dynamic_cast<megamol::geocalls::LinesDataCall*>(&c);
    if (linesOutCall == nullptr) return false;

    (*linesOutCall) = (*linesInCall);

    return true;
}


bool megamol::stdplugin::datatools::TrajectoryAnimator::getLinesExtentCallback(megamol::core::Call& c) {
    if (!this->assertData(c)) return false;

    megamol::geocalls::LinesDataCall* linesOutCall = dynamic_cast<megamol::geocalls::LinesDataCall*>(&c);
    if (linesOutCall == nullptr) return false;

    linesOutCall->SetDataHash(this->datahash);
    linesOutCall->SetFrameCount(this->outFrameCount);
    linesOutCall->SetFrameID(this->frameID);

    linesOutCall->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
    linesOutCall->AccessBoundingBoxes().SetObjectSpaceClipBox(this->bbox);
    linesOutCall->AccessBoundingBoxes().MakeScaledWorld(1.0f);

    return true;
}


bool megamol::stdplugin::datatools::TrajectoryAnimator::getPointsDataCallback(megamol::core::Call& c) {
    megamol::core::moldyn::MultiParticleDataCall* pointsOutCall =
        dynamic_cast<megamol::core::moldyn::MultiParticleDataCall*>(&c);
    if (pointsOutCall == nullptr) return false;

    pointsOutCall->SetParticleListCount(1);
    auto& particles = pointsOutCall->AccessParticles(0);
    particles.SetCount(this->pointData.size() / 3);
    particles.SetVertexData(megamol::core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ,
        this->pointData.data());
    particles.SetGlobalRadius(this->globalRadiusSlot.Param<megamol::core::param::FloatParam>()->Value());
    particles.SetGlobalColour(128, 128, 128);

    return true;
}


bool megamol::stdplugin::datatools::TrajectoryAnimator::getPointsExtentCallback(megamol::core::Call& c) {
    megamol::core::moldyn::MultiParticleDataCall* pointsOutCall =
        dynamic_cast<megamol::core::moldyn::MultiParticleDataCall*>(&c);
    if (pointsOutCall == nullptr) return false;

    pointsOutCall->SetDataHash(this->datahash);
    pointsOutCall->SetFrameCount(this->outFrameCount);
    pointsOutCall->SetFrameID(this->frameID);

    pointsOutCall->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
    pointsOutCall->AccessBoundingBoxes().SetObjectSpaceClipBox(this->bbox);
    pointsOutCall->AccessBoundingBoxes().MakeScaledWorld(1.0f);

    return true;
}


bool megamol::stdplugin::datatools::TrajectoryAnimator::assertData(
    megamol::core::Call& linesC) {
    megamol::geocalls::LinesDataCall* linesInCall = this->linesInSlot.CallAs<megamol::geocalls::LinesDataCall>();
    if (linesInCall == nullptr) return false;

    megamol::geocalls::LinesDataCall* linesOutCall = dynamic_cast<megamol::geocalls::LinesDataCall*>(&linesC);
    if (linesOutCall == nullptr) return false;

    /*megamol::core::moldyn::MultiParticleDataCall* pointsOutCall =
        dynamic_cast<megamol::core::moldyn::MultiParticleDataCall*>(&pointsC);
    if (pointsOutCall == nullptr) return false;*/

    if (this->isDirty() || this->datahash != linesInCall->DataHash() || this->frameID != linesOutCall->FrameID()) {

        unsigned int const requestedFrameID = linesOutCall->FrameID();
        this->frameID = requestedFrameID;

        this->inFrameCount = this->inFrameCountSlot.Param<megamol::core::param::IntParam>()->Value();

        auto const animationFactor = this->animationFactorSlot.Param<megamol::core::param::IntParam>()->Value();

        this->outFrameCount = this->inFrameCount*animationFactor;

        if (this->datahash != linesInCall->DataHash()) {
            linesInCall->SetFrameID(0, true);

            if (!(*linesInCall)(1)) return false;
            if (!(*linesInCall)(0)) return false;
        }
        this->datahash = linesInCall->DataHash();

        this->bbox = linesInCall->AccessBoundingBoxes().ObjectSpaceBBox();

        auto const linesCount = linesInCall->Count();
        auto const lines = linesInCall->GetLines();

        this->pointData.clear();
        this->pointData.reserve(linesCount * 3);

        unsigned int const start = requestedFrameID / animationFactor;
        unsigned int const end = start + 1;
        float const diff =
            (static_cast<float>(requestedFrameID) / static_cast<float>(animationFactor)) - static_cast<float>(start);

        if (start < this->inFrameCount && end < this->inFrameCount) {
            for (unsigned int li = 0; li < linesCount; ++li) {
                auto& line = lines[li];

                if (line.Count() != this->inFrameCount) {
                    vislib::sys::Log::DefaultLog.WriteError(
                        "TrajectoryAnimator: Unexpected length of line. Length is %d and should be %d\n", line.Count(),
                        this->inFrameCount);
                    return false;
                }

                point startPoint;
                point endPoint;

                startPoint.x = line[start].vert.GetXf();
                startPoint.y = line[start].vert.GetYf();
                startPoint.z = line[start].vert.GetZf();

                endPoint.x = line[end].vert.GetXf();
                endPoint.y = line[end].vert.GetYf();
                endPoint.z = line[end].vert.GetZf();

                // interpolate
                auto const res = interpolate(startPoint, endPoint, diff);

                this->pointData.push_back(res.x);
                this->pointData.push_back(res.y);
                this->pointData.push_back(res.z);
            }
        }
    }

    return true;
}


bool megamol::stdplugin::datatools::TrajectoryAnimator::isDirty() {
    if (this->animationFactorSlot.IsDirty() ||
        this->inFrameCountSlot.IsDirty()) {
        this->animationFactorSlot.ResetDirty();
        this->inFrameCountSlot.ResetDirty();
        return true;
    }

    return false;
}
