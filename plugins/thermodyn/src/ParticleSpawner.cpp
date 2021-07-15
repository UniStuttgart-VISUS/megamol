#include "ParticleSpawner.h"

#include <random>

#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

#include "glm/glm.hpp"

megamol::thermodyn::ParticleSpawner::ParticleSpawner()
        : out_data_slot_("outData", "")
        , in_data_slot_("inData", "")
        , in_stats_slot_("inStat", "")
        , dx_slot_("vel::x", "")
        , dy_slot_("vel::y", "")
        , dz_slot_("vel::z", "")
        , dmag_slot_("vel::mag", "")
        , alpha_slot_("alpha", "")
        , num_part_slot_("num particles", "") {
    out_data_slot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(0), &ParticleSpawner::get_data_cb);
    out_data_slot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(1), &ParticleSpawner::get_extent_cb);
    MakeSlotAvailable(&out_data_slot_);

    in_data_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&in_data_slot_);

    in_stats_slot_.SetCompatibleCall<CallStatsInfoDescription>();
    MakeSlotAvailable(&in_stats_slot_);

    dx_slot_ << new core::param::FlexEnumParam("dx");
    MakeSlotAvailable(&dx_slot_);
    dy_slot_ << new core::param::FlexEnumParam("dy");
    MakeSlotAvailable(&dy_slot_);
    dz_slot_ << new core::param::FlexEnumParam("dz");
    MakeSlotAvailable(&dz_slot_);
    dmag_slot_ << new core::param::FlexEnumParam("dmag");
    MakeSlotAvailable(&dmag_slot_);

    alpha_slot_ << new core::param::FloatParam(1.25f, std::numeric_limits<float>::min());
    MakeSlotAvailable(&alpha_slot_);

    num_part_slot_ << new core::param::IntParam(42000, 0);
    MakeSlotAvailable(&num_part_slot_);
}


megamol::thermodyn::ParticleSpawner::~ParticleSpawner() {
    this->Release();
}


bool megamol::thermodyn::ParticleSpawner::create() {
    return true;
}


void megamol::thermodyn::ParticleSpawner::release() {}


bool megamol::thermodyn::ParticleSpawner::assert_data(
    core::moldyn::MultiParticleDataCall& part_call, CallStatsInfo& stats_call) {
    auto const alpha = alpha_slot_.Param<core::param::FloatParam>()->Value();

    auto const num_part = num_part_slot_.Param<core::param::IntParam>()->Value();

    auto const dx_name = dx_slot_.Param<core::param::FlexEnumParam>()->Value();
    auto const dy_name = dy_slot_.Param<core::param::FlexEnumParam>()->Value();
    auto const dz_name = dz_slot_.Param<core::param::FlexEnumParam>()->Value();
    auto const dmag_name = dmag_slot_.Param<core::param::FlexEnumParam>()->Value();

    auto const distros = stats_call.getData();

    auto fit_dx =
        std::find_if(distros.begin(), distros.end(), [&dx_name](auto const& val) { return val.name == dx_name; });
    auto fit_dy =
        std::find_if(distros.begin(), distros.end(), [&dy_name](auto const& val) { return val.name == dy_name; });
    auto fit_dz =
        std::find_if(distros.begin(), distros.end(), [&dz_name](auto const& val) { return val.name == dz_name; });
    auto fit_dmag =
        std::find_if(distros.begin(), distros.end(), [&dmag_name](auto const& val) { return val.name == dmag_name; });

    if (fit_dx == distros.end() || fit_dy == distros.end() || fit_dz == distros.end() || fit_dmag == distros.end()) {
        core::utility::log::Log::DefaultLog.WriteError("[ParticleSpawner] Could not find requested distribution");
        return false;
    }

    auto dx_stat = std::make_pair(fit_dx->mean, fit_dx->stddev);
    auto dy_stat = std::make_pair(fit_dy->mean, fit_dy->stddev);
    auto dz_stat = std::make_pair(fit_dz->mean, fit_dz->stddev);
    auto dmag_stat = std::make_pair(fit_dmag->mean, fit_dmag->stddev);

    auto const pl_count = part_call.GetParticleListCount();

    auto const box = part_call.AccessBoundingBoxes().ObjectSpaceBBox();

    data_.resize(pl_count);

    for (std::remove_const_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
        auto& data = data_[pl_idx];
        data.clear();
        data.reserve(num_part * 7);

        auto const& parts = part_call.AccessParticles(pl_idx);

        auto const p_count = parts.GetCount();

        auto const xAcc = parts.GetParticleStore().GetXAcc();
        auto const yAcc = parts.GetParticleStore().GetYAcc();
        auto const zAcc = parts.GetParticleStore().GetZAcc();

        std::vector<std::pair<Point_3, std::size_t>> points;
        points.reserve(p_count);

        for (std::remove_const_t<decltype(p_count)> pidx = 0; pidx < p_count; ++pidx) {
            points.emplace_back(std::make_pair(Point_3(xAcc->Get_f(pidx), yAcc->Get_f(pidx), zAcc->Get_f(pidx)), pidx));
        }

        Triangulation_3 tri = Triangulation_3(points.begin(), points.end());

        auto as = Alpha_shape_3(tri, alpha);

        auto px_dist = std::uniform_real_distribution<float>(box.GetLeft(), box.GetRight());
        auto py_dist = std::uniform_real_distribution<float>(box.GetBottom(), box.GetTop());
        auto pz_dist = std::uniform_real_distribution<float>(box.GetBack(), box.GetFront());

        auto dx_dist = std::normal_distribution<float>(dx_stat.first, dx_stat.second);
        auto dy_dist = std::normal_distribution<float>(dy_stat.first, dy_stat.second);
        auto dz_dist = std::normal_distribution<float>(dz_stat.first, dz_stat.second);
        auto dmag_dist = std::normal_distribution<float>(dmag_stat.first, dmag_stat.second);

        auto rng = std::mt19937_64(42);

        auto const start = std::chrono::high_resolution_clock::now();
        auto now = std::chrono::high_resolution_clock::now();
        using namespace std::chrono_literals;
        int num_spawned = 0;
        while (num_spawned <= num_part && (std::chrono::duration_cast<std::chrono::seconds>(now - start) < 600s)) {
            auto const pos = Point_3(px_dist(rng), py_dist(rng), pz_dist(rng));

            auto const cell = as.locate(pos);
            if (as.is_valid_finite(cell) && (as.INTERIOR == cell->get_classification_type())) {
                // we have a valid candidate
                auto const dmag = dmag_dist(rng);
                auto const dir = glm::normalize(glm::vec3(dx_dist(rng), dy_dist(rng), dz_dist(rng))) * dmag;

                data.push_back(pos.x());
                data.push_back(pos.y());
                data.push_back(pos.z());
                data.push_back(dmag);
                data.push_back(dir.x);
                data.push_back(dir.y);
                data.push_back(dir.z);

                ++num_spawned;
            }
            now = std::chrono::high_resolution_clock::now();
        }

        core::utility::log::Log::DefaultLog.WriteInfo("[ParticleSpawner] Created particles %d", num_spawned);
    }

    return true;
}


bool megamol::thermodyn::ParticleSpawner::get_data_cb(core::Call& c) {
    auto out_part = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (out_part == nullptr)
        return false;
    auto in_data = in_data_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_data == nullptr)
        return false;
    auto stats_data = in_stats_slot_.CallAs<CallStatsInfo>();
    if (stats_data == nullptr)
        return false;

    in_data->SetFrameID(out_part->FrameID());
    if (!(*in_data)(1))
        return false;
    if (!(*in_data)(0))
        return false;

    auto meta = stats_data->getMetaData();
    meta.m_frame_ID = out_part->FrameID();
    stats_data->setMetaData(meta);
    if (!(*stats_data)(1))
        return false;
    if (!(*stats_data)(0))
        return false;


    if (in_data->DataHash() != in_data_hash_ || in_data->FrameID() != frame_id_ || is_dirty()) {
        auto const res = assert_data(*in_data, *stats_data);


        frame_id_ = in_data->FrameID();
        in_data_hash_ = in_data->DataHash();
        reset_dirty();

        ++out_data_hash_;
    }

    out_part->SetParticleListCount(in_data->GetParticleListCount());

    for (std::remove_const_t<decltype(in_data->GetParticleListCount())> pl_idx = 0;
         pl_idx < in_data->GetParticleListCount(); ++pl_idx) {

        auto const& part_data = data_[pl_idx];
        auto& parts = out_part->AccessParticles(pl_idx);

        parts.SetCount(part_data.size() / 7);
        parts.SetVertexData(
            core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ, part_data.data(), 7 * sizeof(float));
        parts.SetColourData(
            core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I, part_data.data() + 3, 7 * sizeof(float));
        parts.SetDirData(
            core::moldyn::SimpleSphericalParticles::DIRDATA_FLOAT_XYZ, part_data.data() + 4, 7 * sizeof(float));
        parts.SetGlobalRadius(in_data->AccessParticles(pl_idx).GetGlobalRadius());
        parts.SetColourMapIndexValues(in_data->AccessParticles(pl_idx).GetMinColourIndexValue(),
            in_data->AccessParticles(pl_idx).GetMaxColourIndexValue());
    }


    auto const bbox = in_data->AccessBoundingBoxes().ObjectSpaceBBox();
    auto const cbox = in_data->AccessBoundingBoxes().ObjectSpaceClipBox();

    out_part->AccessBoundingBoxes().SetObjectSpaceBBox(bbox);
    out_part->AccessBoundingBoxes().SetObjectSpaceClipBox(cbox);

    out_part->SetFrameCount(in_data->FrameCount());
    out_part->SetFrameID(in_data->FrameID());

    out_part->SetDataHash(out_data_hash_);

    return true;
}


bool megamol::thermodyn::ParticleSpawner::get_extent_cb(core::Call& c) {
    auto out_part = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (out_part == nullptr)
        return false;
    auto in_data = in_data_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_data == nullptr)
        return false;

    in_data->SetFrameID(out_part->FrameID());
    if (!(*in_data)(1))
        return false;

    auto const bbox = in_data->AccessBoundingBoxes().ObjectSpaceBBox();
    auto const cbox = in_data->AccessBoundingBoxes().ObjectSpaceClipBox();

    out_part->AccessBoundingBoxes().SetObjectSpaceBBox(bbox);
    out_part->AccessBoundingBoxes().SetObjectSpaceClipBox(cbox);

    out_part->SetFrameCount(in_data->FrameCount());
    out_part->SetFrameID(in_data->FrameID());

    return true;
}
