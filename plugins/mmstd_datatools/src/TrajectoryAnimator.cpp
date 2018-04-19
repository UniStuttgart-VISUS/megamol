#include "stdafx.h"
#include "TrajectoryAnimator.h"

#include <algorithm>
#include <limits>

#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/EnumParam.h"

#include "geometry_calls/LinesDataCall.h"

#include "vislib/sys/Log.h"


megamol::stdplugin::datatools::TrajectoryAnimator::TrajectoryAnimator(void)
    : megamol::core::Module()
    , linesOutSlot("linesOut", "Output of transition trajectories")
    , pointsOutSlot("pointOut", "Output of points which highlight trajectories")
    , highlightsOutSlot("highlightsOut", "Output of highlights for trajectories as points")
    , linesInSlot("linesIn", "Input of trajectories")
    , pointsInSlot("pointsIn", "Input of points to enable interpolation")
    , animationFactorSlot("frameFactor", "Steers the slow down of the animation wrt the framecount")
    , inFrameCountSlot("frameCount", "Set number of input frames")
    , globalRadiusSlot("globalRadius", "Set the global radius")
    , animationLengthSlot("animationLength", "Set the length of the animation in frames")
    , minTransSlot("trans::min", "Pos of lowest transition plane")
    , maxTransSlot("trans::max", "Pos of highest transition plane")
    , transDirSlot("trans::dir", "Main direction of transition")
    , transEpsSlot("trans::eps", "Epsilon area to detect transition")
    , highlightRadiusSlot("highlightRadius", "Radius of the highlights")
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

    this->highlightsOutSlot.SetCallback(megamol::core::moldyn::MultiParticleDataCall::ClassName(),
        megamol::core::moldyn::MultiParticleDataCall::FunctionName(0), &TrajectoryAnimator::getHighlightsDataCallback);
    this->highlightsOutSlot.SetCallback(megamol::core::moldyn::MultiParticleDataCall::ClassName(),
        megamol::core::moldyn::MultiParticleDataCall::FunctionName(1), &TrajectoryAnimator::getHighlightsExtentCallback);
    this->MakeSlotAvailable(&this->highlightsOutSlot);

    this->linesInSlot.SetCompatibleCall<megamol::geocalls::LinesDataCallDescription>();
    this->MakeSlotAvailable(&this->linesInSlot);

    this->pointsInSlot.SetCompatibleCall<megamol::core::moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->pointsInSlot);

    this->animationFactorSlot << new megamol::core::param::IntParam(1, 1, 1000);
    this->MakeSlotAvailable(&this->animationFactorSlot);

    this->inFrameCountSlot << new megamol::core::param::IntParam(0, 0, std::numeric_limits<int>::max());
    this->MakeSlotAvailable(&this->inFrameCountSlot);

    this->globalRadiusSlot << new megamol::core::param::FloatParam(
        0.5f, std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
    this->MakeSlotAvailable(&this->globalRadiusSlot);

    this->animationLengthSlot << new megamol::core::param::IntParam(1, 1, std::numeric_limits<int>::max());
    this->MakeSlotAvailable(&this->animationLengthSlot);

    this->minTransSlot << new megamol::core::param::FloatParam(0.0f, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());
    this->MakeSlotAvailable(&this->minTransSlot);

    this->maxTransSlot << new megamol::core::param::FloatParam(0.0f, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());
    this->MakeSlotAvailable(&this->maxTransSlot);

    auto ep = new megamol::core::param::EnumParam(0);
    ep->SetTypePair(0, "x");
    ep->SetTypePair(1, "y");
    ep->SetTypePair(2, "z");
    this->transDirSlot << ep;
    this->MakeSlotAvailable(&this->transDirSlot);

    this->transEpsSlot << new megamol::core::param::FloatParam(1.0f, std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
    this->MakeSlotAvailable(&this->transEpsSlot);

    this->highlightRadiusSlot << new megamol::core::param::FloatParam(0.5f, std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
    this->MakeSlotAvailable(&this->highlightRadiusSlot);
}


megamol::stdplugin::datatools::TrajectoryAnimator::~TrajectoryAnimator(void) { this->Release(); }


bool megamol::stdplugin::datatools::TrajectoryAnimator::create(void) {
    this->dummyLinePos.push_back(0.0f);
    this->dummyLinePos.push_back(0.0f);
    this->dummyLinePos.push_back(0.0f);

    this->dummyLinePos.push_back(0.1f);
    this->dummyLinePos.push_back(0.1f);
    this->dummyLinePos.push_back(0.1f);

    this->dummyLine.Set(2, this->dummyLinePos.data(), vislib::graphics::ColourRGBAu8(255, 255, 255, 0));

    return true;
}


void megamol::stdplugin::datatools::TrajectoryAnimator::release(void) {}


bool megamol::stdplugin::datatools::TrajectoryAnimator::getLinesDataCallback(megamol::core::Call& c) {
    if (!this->assertData(c)) return false;

    megamol::geocalls::LinesDataCall* linesInCall = this->linesInSlot.CallAs<megamol::geocalls::LinesDataCall>();
    if (linesInCall == nullptr) return false;

    megamol::geocalls::LinesDataCall* linesOutCall = dynamic_cast<megamol::geocalls::LinesDataCall*>(&c);
    if (linesOutCall == nullptr) return false;

    //(*linesOutCall) = (*linesInCall);

    if (this->renderQueue.size() != 0) {
        linesOutCall->SetData(this->renderQueue.size(), this->renderQueue.data());
    } else {
        linesOutCall->SetData(1, &this->dummyLine);
    }

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
    particles.SetVertexData(
        megamol::core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ, this->pointData.data());
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


bool megamol::stdplugin::datatools::TrajectoryAnimator::getHighlightsDataCallback(megamol::core::Call & c) {
    megamol::core::moldyn::MultiParticleDataCall* pointsOutCall =
        dynamic_cast<megamol::core::moldyn::MultiParticleDataCall*>(&c);
    if (pointsOutCall == nullptr) return false;

    if (this->highlightPointData.size()== 0) {
        this->highlightPointData.push_back(0.0f);
        this->highlightPointData.push_back(0.0f);
        this->highlightPointData.push_back(0.0f);
    }

    pointsOutCall->SetParticleListCount(1);
    auto& particles = pointsOutCall->AccessParticles(0);
    particles.SetCount(this->highlightPointData.size() / 3);
    particles.SetVertexData(
        megamol::core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ, this->highlightPointData.data());
    particles.SetGlobalRadius(this->highlightRadiusSlot.Param<megamol::core::param::FloatParam>()->Value());
    particles.SetGlobalColour(255, 255, 255, 255);

    return true;
}


bool megamol::stdplugin::datatools::TrajectoryAnimator::getHighlightsExtentCallback(megamol::core::Call & c) {
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


bool megamol::stdplugin::datatools::TrajectoryAnimator::assertData(megamol::core::Call& linesC) {
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

        auto const animationLength = this->animationLengthSlot.Param<megamol::core::param::IntParam>()->Value();

        unsigned int const trans_axis = this->transDirSlot.Param<megamol::core::param::EnumParam>()->Value();

        float tmp_min_trans = this->minTransSlot.Param<megamol::core::param::FloatParam>()->Value();
        float tmp_max_trans = this->maxTransSlot.Param<megamol::core::param::FloatParam>()->Value();
        if (tmp_max_trans < tmp_min_trans) std::swap(tmp_min_trans, tmp_max_trans);

        float const min_bor{tmp_min_trans};
        float const max_bor{tmp_max_trans};

        auto const trans_eps = this->transEpsSlot.Param<megamol::core::param::FloatParam>()->Value();

        this->outFrameCount = this->inFrameCount * animationFactor;

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

                auto start_end = checkPadding(line);
                if (start < start_end.first || end > start_end.second) continue;

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

                // if current point is within the transition area, add the current trajectory to the render queue
                if (checkTransition(startPoint, endPoint, min_bor, max_bor, trans_axis, trans_eps)) {
                    auto const it = this->transitionLines.find(li);
                    if (it == this->transitionLines.end()) {
                        this->transitionLines2Frames[li] = requestedFrameID;
                        this->transitionLines[li] = removePadding(line, start_end);
                    }
                }
            }

            // create render queue
            this->renderQueue.clear();
            std::vector<unsigned int> to_erase;
            for (auto const& el : this->transitionLines2Frames) {
                auto const idx = el.first;
                if (requestedFrameID > el.second+animationLength) {
                    to_erase.push_back(idx);
                    /*this->transitionLines2Frames.erase(this->transitionLines2Frames.find(idx));
                    this->transitionLines.erase(this->transitionLines.find(idx));*/
                }
            }

            for (auto const& el : to_erase) {
                this->transitionLines2Frames.erase(this->transitionLines2Frames.find(el));
                this->transitionLines.erase(this->transitionLines.find(el));
            }

            for (auto const& el : this->transitionLines) {
                megamol::geocalls::LinesDataCall::Lines l;
                l.Set(el.second.first.size() / 3, el.second.first.data(), el.second.second.data(), true);
                this->renderQueue.push_back(l);
            }

            // animate highlight points
            this->highlightPointData.clear();
            this->highlightPointData.reserve(this->transitionLines.size() * 3);
            for (auto const& el: this->transitionLines2Frames) {
                auto const idx = el.first;

                auto const anim_start = el.second;
                auto const anim_end = el.second + animationLength;

                auto const& pos = this->transitionLines.find(idx)->second.first;

                auto const traj_l = pos.size() / 3;

                auto const idx_inter = (requestedFrameID - anim_start) / static_cast<float>(anim_end - anim_start);

                unsigned int const pos_idx_start = std::floorf(traj_l * idx_inter);

                if (pos_idx_start<traj_l) {
                    this->highlightPointData.push_back(pos[pos_idx_start * 3 + 0]);
                    this->highlightPointData.push_back(pos[pos_idx_start * 3 + 1]);
                    this->highlightPointData.push_back(pos[pos_idx_start * 3 + 2]);
                }
            }
        }
    }

    return true;
}


bool megamol::stdplugin::datatools::TrajectoryAnimator::isDirty() {
    if (this->animationFactorSlot.IsDirty() || this->inFrameCountSlot.IsDirty() || this->minTransSlot.IsDirty() ||
        this->maxTransSlot.IsDirty() || this->transDirSlot.IsDirty() || this->transEpsSlot.IsDirty()) {
        this->animationFactorSlot.ResetDirty();
        this->inFrameCountSlot.ResetDirty();
        this->minTransSlot.ResetDirty();
        this->maxTransSlot.ResetDirty();
        this->transDirSlot.ResetDirty();
        this->transEpsSlot.ResetDirty();
        return true;
    }

    return false;
}


bool megamol::stdplugin::datatools::TrajectoryAnimator::checkTransition(point const& pos_a, point const& pos_b,
    float const min_trans, float const max_trans, unsigned int const trans_dir, float const trans_eps) {

    auto a = pos_a[trans_dir];
    auto b = pos_b[trans_dir];

    if (a > b) std::swap(a, b);

    return (a < min_trans && b > min_trans && a < max_trans && b < max_trans) ||
           (a < max_trans && b > max_trans && a > min_trans && b > min_trans);

    /*return (d >= min_trans - trans_eps && d <= min_trans + trans_eps) ||
        (d >= max_trans - trans_dir && d <= max_trans + trans_dir);*/
}


std::pair<std::vector<float>, std::vector<unsigned char>>
megamol::stdplugin::datatools::TrajectoryAnimator::removePadding(megamol::geocalls::LinesDataCall::Lines const& l, std::pair<unsigned int, unsigned int> const& start_end) const {
    std::vector<point> pos;
    pos.reserve(l.Count());
    std::vector<color> color;
    color.reserve(l.Count());

    for (unsigned int li = 0; li < l.Count(); ++li) {
        pos.push_back({l[li].vert.GetXf(), l[li].vert.GetYf(), l[li].vert.GetZf()});

        color.push_back({l[li].col.GetRu8(), l[li].col.GetGu8(), l[li].col.GetBu8(), l[li].col.GetAu8()});
    }

    /*auto const pos_it = std::unique(pos.begin(), pos.end());
    auto const col_it = std::unique(color.begin(), color.end());*/

    /*pos.resize(std::distance(pos.begin(), pos_it));
    color.resize(std::distance(color.begin(), col_it));*/

    std::vector<float> pos_f;
    pos_f.reserve((start_end.second - start_end.first) * 3);
    //pos_f.reserve(std::distance(pos.begin(), pos_it) * 3);
    std::vector<unsigned char> col_u;
    col_u.reserve((start_end.second - start_end.first) * 4);
    //col_u.reserve(std::distance(color.begin(), col_it) * 4);

    for (unsigned int idx = start_end.first; idx <= start_end.second; ++idx) {
        pos_f.push_back(pos[idx].x);
        pos_f.push_back(pos[idx].y);
        pos_f.push_back(pos[idx].z);

        col_u.push_back(color[idx].r);
        col_u.push_back(color[idx].g);
        col_u.push_back(color[idx].b);
        col_u.push_back(color[idx].a);
    }

    /*if (pos.size() > 1) {
        for (unsigned int idx = 0; idx < pos.size() - 1; ++idx) {
            if (pos[idx] != pos[idx + 1]) {
                pos_f.push_back(pos[idx].x);
                pos_f.push_back(pos[idx].y);
                pos_f.push_back(pos[idx].z);

                col_u.push_back(color[idx].r);
                col_u.push_back(color[idx].g);
                col_u.push_back(color[idx].b);
                col_u.push_back(color[idx].a);
            }
        }

        if (pos.size() > 2) {
            if (pos[pos.size() - 2] != pos[pos.size() - 1]) {
                pos_f.push_back(pos[pos.size() - 1].x);
                pos_f.push_back(pos[pos.size() - 1].y);
                pos_f.push_back(pos[pos.size() - 1].z);

                col_u.push_back(color[pos.size() - 1].r);
                col_u.push_back(color[pos.size() - 1].g);
                col_u.push_back(color[pos.size() - 1].b);
                col_u.push_back(color[pos.size() - 1].a);
            }
        }
    }

    pos_f.shrink_to_fit();
    col_u.shrink_to_fit();*/

    /*for (auto pit = pos.begin(); pit != pos_it; ++pit) {
        pos_f.push_back(pit->x);
        pos_f.push_back(pit->y);
        pos_f.push_back(pit->z);
    }

    for (auto cit = color.begin(); cit != col_it; ++cit) {
        col_u.push_back(cit->r);
        col_u.push_back(cit->g);
        col_u.push_back(cit->b);
        col_u.push_back(cit->a);
    }*/

    return std::make_pair(pos_f, col_u);
}


std::pair<unsigned int, unsigned int> megamol::stdplugin::datatools::TrajectoryAnimator::checkPadding(
    megamol::geocalls::LinesDataCall::Lines const& l) const {
    std::pair<unsigned int, unsigned int> start_end;

    std::vector<point> pos;
    pos.reserve(l.Count());

    for (unsigned int li = 0; li < l.Count(); ++li) {
        pos.push_back({l[li].vert.GetXf(), l[li].vert.GetYf(), l[li].vert.GetZf()});
    }

    if (pos.size() > 1) {
        for (size_t i = 0; i < pos.size() - 1; ++i) {
            if (pos[i] != pos[i+1]) {
                start_end.first = i;
                break;
            }
        }

        for (size_t i = pos.size() - 1; i > 0; --i) {
            if (pos[i] != pos[i - 1]) {
                start_end.second = i;
                break;
            }
        }
    }

    return start_end;
}
