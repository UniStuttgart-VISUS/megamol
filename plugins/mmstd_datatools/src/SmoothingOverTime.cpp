#include "SmoothingOverTime.h"

#include "mmcore/param/IntParam.h"


megamol::stdplugin::datatools::SmoothingOverTime::SmoothingOverTime()
        : out_data_slot_("outData", "")
        , in_data_slot_("inData", "")
        , frame_count_slot_("frame count", "")
        , frame_skip_slot_("frame skip", "") {
    out_data_slot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(0), &SmoothingOverTime::get_data_cb);
    out_data_slot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(1), &SmoothingOverTime::get_extent_cb);
    MakeSlotAvailable(&out_data_slot_);

    in_data_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&in_data_slot_);

    frame_count_slot_ << new core::param::IntParam(1, 1);
    MakeSlotAvailable(&frame_count_slot_);

    frame_skip_slot_ << new core::param::IntParam(1, 1);
    MakeSlotAvailable(&frame_skip_slot_);
}


megamol::stdplugin::datatools::SmoothingOverTime::~SmoothingOverTime() {
    this->Release();
}


bool megamol::stdplugin::datatools::SmoothingOverTime::create() {
    return true;
}


void megamol::stdplugin::datatools::SmoothingOverTime::release() {}


bool megamol::stdplugin::datatools::SmoothingOverTime::get_data_cb(core::Call& c) {
    auto out_data = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (out_data == nullptr)
        return false;
    auto in_data = in_data_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_data == nullptr)
        return false;

    in_data->SetFrameID(out_data->FrameID());
    if (!(*in_data)(1))
        return false;
    if (!(*in_data)(0))
        return false;

    if (in_data->FrameID() != frame_id_ /*|| in_data->DataHash() != in_data_hash_*/ || is_dirty()) {
        auto const req_frame_count = frame_count_slot_.Param<core::param::IntParam>()->Value();
        auto const frame_skip = frame_skip_slot_.Param<core::param::IntParam>()->Value();

        if (in_data->FrameID() + req_frame_count >= in_data->FrameCount())
            return false;

        auto const pl_count = in_data->GetParticleListCount();

        weigths_.resize(pl_count);
        identity_.resize(pl_count);
        smoothed_icol_.resize(pl_count);
        minmax_.resize(pl_count);

        for (std::remove_cv_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
            auto const& parts = in_data->AccessParticles(pl_idx);

            auto const& idAcc = parts.GetParticleStore().GetIDAcc();
            auto const& icAcc = parts.GetParticleStore().GetCRAcc();

            auto& weigths = weigths_[pl_idx];
            auto& ident = identity_[pl_idx];
            auto& icols = smoothed_icol_[pl_idx];

            auto const p_count = parts.GetCount();
            weigths.resize(p_count);
            ident.resize(p_count);
            icols.resize(p_count);
            for (std::remove_cv_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx) {
                weigths[p_idx] = 1.f;
                ident[p_idx] = idAcc->Get_u64(p_idx);
                icols[p_idx] = icAcc->Get_f(p_idx);
            }
        }

        auto base_fid = in_data->FrameID();

        for (int fid = base_fid + frame_skip; fid < base_fid + req_frame_count; fid += frame_skip) {
            bool got_it = false;
            do {
                in_data->SetFrameID(fid, true);
                got_it = (*in_data)(1);
                got_it = got_it && (*in_data)(0);
            } while (in_data->FrameID() != fid && !got_it);

            auto const pl_count = in_data->GetParticleListCount();
            for (std::remove_cv_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
                auto const& parts = in_data->AccessParticles(pl_idx);

                auto const& idAcc = parts.GetParticleStore().GetIDAcc();
                auto const& icAcc = parts.GetParticleStore().GetCRAcc();

                auto& weigths = weigths_[pl_idx];
                auto& ident = identity_[pl_idx];
                auto& icols = smoothed_icol_[pl_idx];

                auto const p_count = parts.GetCount();
                for (std::remove_cv_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx) {
                    auto const id = idAcc->Get_u64(p_idx);
                    auto idx = p_idx;
                    /*if (ident[p_idx] != id)*/ {
                        auto const fit = std::find(ident.begin(), ident.end(), id);
                        if (fit == ident.end())
                            continue;
                        idx = std::distance(ident.begin(), fit);
                    }
                    weigths[idx] += 1.f;
                    icols[idx] += icAcc->Get_f(p_idx);
                }
            }
        }

        for (std::remove_cv_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
            auto& weigths = weigths_[pl_idx];
            auto& ident = identity_[pl_idx];
            auto& icols = smoothed_icol_[pl_idx];
            std::transform(
                weigths.begin(), weigths.end(), icols.begin(), icols.begin(), [](float w, float i) { return i / w; });

            auto minmax = std::minmax_element(icols.begin(), icols.end());
            minmax_[pl_idx] = {*minmax.first, *minmax.second};
        }

        frame_id_ = base_fid;
        in_data_hash_ = in_data->DataHash();
        reset_dirty();
        ++out_data_hash_;
    }

    out_data->SetParticleListCount(in_data->GetParticleListCount());
    for (unsigned int pl_idx = 0; pl_idx < in_data->GetParticleListCount(); ++pl_idx) {
        auto& out_parts = out_data->AccessParticles(pl_idx);
        auto const& in_parts = in_data->AccessParticles(pl_idx);
        out_parts = in_parts;
        out_parts.SetColourData(core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I, smoothed_icol_[pl_idx].data());
        out_parts.SetColourMapIndexValues(minmax_[pl_idx][0], minmax_[pl_idx][1]);
    }

    out_data->SetFrameCount(in_data->FrameCount());
    out_data->SetFrameID(frame_id_);
    out_data->SetDataHash(out_data_hash_);

    return true;
}


bool megamol::stdplugin::datatools::SmoothingOverTime::get_extent_cb(core::Call& c) {
    auto out_data = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (out_data == nullptr)
        return false;
    auto in_data = in_data_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_data == nullptr)
        return false;
    if (!(*in_data)(1))
        return false;

    out_data->AccessBoundingBoxes().SetObjectSpaceBBox(in_data->AccessBoundingBoxes().ObjectSpaceBBox());
    out_data->AccessBoundingBoxes().SetObjectSpaceClipBox(in_data->AccessBoundingBoxes().ObjectSpaceClipBox());

    out_data->SetFrameCount(in_data->FrameCount());
    /*out_data->SetFrameID(frame_id_);
    out_data->SetDataHash(out_data_hash_);*/

    return true;
}
