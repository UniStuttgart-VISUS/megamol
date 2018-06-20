#include "stdafx.h"
#include "ParticleVelocitiesDirAnalyzer.h"

#include <functional>
#include <limits>

#include "mmcore/moldyn/DirectionalParticleDataCall.h"

#include "mmcore/param/EnumParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/BoolParam.h"


megamol::stdplugin::datatools::ParticleVelocitiesDirAnalyzer::ParticleVelocitiesDirAnalyzer(void)
    : megamol::core::Module()
    , dataOutSlot("dataOut", "Data output")
    , dataInSlot("dataIn", "Data input")
    , mainDirParamSlot("mainDir", "Set dir of analysis")
    , resParamSlot("res", "Resolution of the volume to build along longest bbox edge")
    , inverseDensityWeightingSlot("inverseDensWeighting", "Toggle weighting of cell values with inverse density")
    , frameID{-1}
    , in_datahash{std::numeric_limits<size_t>::max()}
    , my_datahash{0} {
    this->dataOutSlot.SetCallback(megamol::core::moldyn::VolumeDataCall::ClassName(),
        megamol::core::moldyn::VolumeDataCall::FunctionName(0),
        &ParticleVelocitiesDirAnalyzer::getDataCallback);
    this->dataOutSlot.SetCallback(megamol::core::moldyn::VolumeDataCall::ClassName(),
        megamol::core::moldyn::VolumeDataCall::FunctionName(1),
        &ParticleVelocitiesDirAnalyzer::getExtentCallback);
    this->MakeSlotAvailable(&this->dataOutSlot);

    this->dataInSlot.SetCompatibleCall<megamol::core::moldyn::DirectionalParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->dataInSlot);

    auto ep = new megamol::core::param::EnumParam(0);
    ep->SetTypePair(0, "x");
    ep->SetTypePair(1, "y");
    ep->SetTypePair(2, "z");
    this->mainDirParamSlot << ep;
    this->MakeSlotAvailable(&this->mainDirParamSlot);

    this->resParamSlot << new megamol::core::param::IntParam(100, 1, 1024);
    this->MakeSlotAvailable(&this->resParamSlot);

    this->inverseDensityWeightingSlot << new megamol::core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->inverseDensityWeightingSlot);
}


megamol::stdplugin::datatools::ParticleVelocitiesDirAnalyzer::~ParticleVelocitiesDirAnalyzer(void) {
    this->Release();
}


bool megamol::stdplugin::datatools::ParticleVelocitiesDirAnalyzer::create(void) { return true; }


void megamol::stdplugin::datatools::ParticleVelocitiesDirAnalyzer::release() {}


bool megamol::stdplugin::datatools::ParticleVelocitiesDirAnalyzer::getDataCallback(megamol::core::Call& c) {
    auto outCall = dynamic_cast<megamol::core::moldyn::VolumeDataCall*>(&c);
    if (outCall == nullptr) return false;

    if (!this->assertData(outCall)) return false;

    outCall->SetDataHash(this->my_datahash);
    outCall->SetFrameCount(this->frameCount);
    outCall->SetFrameID(this->frameID);


    outCall->SetComponents(1);
    outCall->SetVolumeDimension(this->cell_num[0], this->cell_num[1], this->cell_num[2]);
    outCall->SetVoxelMapPointer(this->volume.data());
    outCall->SetMinimumDensity(this->stats[0]);
    outCall->SetMeanDensity(this->stats[1]);
    outCall->SetMaximumDensity(this->stats[2]);
    outCall->SetBoundingBox(this->bbox);

    return true;
}


bool megamol::stdplugin::datatools::ParticleVelocitiesDirAnalyzer::getExtentCallback(megamol::core::Call& c) {
    auto outCall = dynamic_cast<megamol::core::moldyn::VolumeDataCall*>(&c);
    if (outCall == nullptr) return false;

    if (!this->assertData(outCall)) return false;

    outCall->SetDataHash(this->my_datahash);
    outCall->SetFrameCount(this->frameCount);
    outCall->SetFrameID(this->frameID);

    outCall->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
    outCall->AccessBoundingBoxes().SetObjectSpaceClipBox(this->bbox);
    outCall->AccessBoundingBoxes().MakeScaledWorld(1.0f);

    return true;
}


bool megamol::stdplugin::datatools::ParticleVelocitiesDirAnalyzer::assertData(megamol::core::moldyn::VolumeDataCall* cvd) {
    auto inCall = this->dataInSlot.CallAs<megamol::core::moldyn::DirectionalParticleDataCall>();
    if (inCall == nullptr) return false;

    inCall->SetFrameID(cvd->FrameID(), cvd->IsFrameForced());
    if (!(*inCall)(1)) return false;
    if (!(*inCall)(0)) return false;

    if (inCall->FrameID() != this->frameID || inCall->DataHash() != this->in_datahash || this->isDirty()) {
        this->frameID = inCall->FrameID();
        this->frameCount = inCall->FrameCount();
        if (this->isDirty() || inCall->DataHash() != this->in_datahash) {
            this->my_datahash++;
            this->resetDirty();
        }
        this->in_datahash = inCall->DataHash();

        int const main_dir_idx = this->mainDirParamSlot.Param<megamol::core::param::EnumParam>()->Value();

        bool const is_dens_weight = this->inverseDensityWeightingSlot.Param<megamol::core::param::BoolParam>()->Value();

        std::function<float(megamol::core::moldyn::DirectionalParticles::dir_particle_t const&)> accessor;

        switch (main_dir_idx) {
            case 0:
                accessor = [](megamol::core::moldyn::DirectionalParticles::dir_particle_t const& par)->float {
                    return par.dir.GetDirXf();
                };
                break;
            case 1:
                accessor = [](megamol::core::moldyn::DirectionalParticles::dir_particle_t const& par)->float {
                    return par.dir.GetDirYf();
                };
                break;
            case 2:
                accessor = [](megamol::core::moldyn::DirectionalParticles::dir_particle_t const& par)->float {
                    return par.dir.GetDirZf();
                };
                break;
            default:
                return false;
        }

        int const res = this->resParamSlot.Param<megamol::core::param::IntParam>()->Value();

        bbox = inCall->AccessBoundingBoxes().ObjectSpaceBBox();

        auto const cell_length = bbox.LongestEdge() / static_cast<float>(res);

        this->cell_num[0] = std::floor(bbox.Width() / cell_length);
        this->cell_num[1] = std::floor(bbox.Height() / cell_length);
        this->cell_num[2] = std::floor(bbox.Depth() / cell_length);

        this->cell_size[0] = bbox.Width() / static_cast<float>(this->cell_num[0]);
        this->cell_size[1] = bbox.Height() / static_cast<float>(this->cell_num[1]);
        this->cell_size[2] = bbox.Depth() / static_cast<float>(this->cell_num[2]);

        auto const num_cells = this->cell_num[0] * this->cell_num[1] * this->cell_num[2];

        this->volume.resize(num_cells);
        std::fill(this->volume.begin(), this->volume.end(), 0.0f);

        std::vector<size_t> num_samples(num_cells, 1);

        auto const parListCount = inCall->GetParticleListCount();

        for (unsigned int pli = 0; pli < parListCount; ++pli) {
            auto const& parts = inCall->AccessParticles(pli);
            auto const part_count = parts.GetCount();

            for (size_t i = 0; i < part_count; ++i) {
                float const x = parts[i].vert.GetXf() - bbox.GetLeft();
                float const y = parts[i].vert.GetYf() - bbox.GetBottom();
                float const z = parts[i].vert.GetZf() - bbox.GetBack();

                float const val = std::fabs(accessor(parts[i]));

                int const idx_x = std::floor(x / this->cell_size[0]);
                int const idx_y = std::floor(y / this->cell_size[1]);
                int const idx_z = std::floor(z / this->cell_size[2]);

                auto const idx = idx_x + this->cell_num[0] * (idx_y + this->cell_num[1] * idx_z);

                auto& vol_val = this->volume[idx];

                vol_val = vol_val + ((val - vol_val) / num_samples[idx]);
                ++num_samples[idx];
            }
        }

        if (is_dens_weight) {
            std::transform(this->volume.begin(), this->volume.end(), num_samples.begin(), this->volume.begin(),
                [](float const& v, size_t const& n)->float {return v / static_cast<float>(n); });
        }

        memset(this->stats, 0, sizeof(float) * 3);
        auto minmax_pair = std::minmax_element(this->volume.begin(), this->volume.end());
        if (minmax_pair.first != this->volume.end() && minmax_pair.second != this->volume.end()) {
            this->stats[0] = *minmax_pair.first;
            this->stats[2] = *minmax_pair.second;
        }
        if (num_cells >= 2) {
            this->stats[1] = this->volume[this->volume.size() / 2];
        }
    }

    return true;
}


bool megamol::stdplugin::datatools::ParticleVelocitiesDirAnalyzer::isDirty() const {
    return mainDirParamSlot.IsDirty() || resParamSlot.IsDirty() || inverseDensityWeightingSlot.IsDirty();
}


void megamol::stdplugin::datatools::ParticleVelocitiesDirAnalyzer::resetDirty() {
    mainDirParamSlot.ResetDirty();
    resParamSlot.ResetDirty();
    inverseDensityWeightingSlot.ResetDirty();
}
