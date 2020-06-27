#include "stdafx.h"
#include "PhaseAnimator.h"

#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/FloatParam.h"

#include "thermodyn/BoxDataCall.h"


megamol::thermodyn::PhaseAnimator::PhaseAnimator()
    : out_data_slot_("dataOut", "")
    , part_in_data_slot_("partDataIn", "")
    , box_in_data_slot_("boxDataIn", "")
    , fluid_alpha_slot_("fluid_alpha", "")
    , interface_alpha_slot_("interface_alpha", "")
    , gas_alpha_slot_("gas_alpha", "")
    , data_hash_(std::numeric_limits<size_t>::max())
    , out_data_hash_(0)
    , frame_id_(0) {
    out_data_slot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(0), &PhaseAnimator::getDataCallback);
    out_data_slot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(1), &PhaseAnimator::getExtentCallback);
    MakeSlotAvailable(&out_data_slot_);

    part_in_data_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&part_in_data_slot_);

    box_in_data_slot_.SetCompatibleCall<thermodyn::BoxDataCallDescription>();
    MakeSlotAvailable(&box_in_data_slot_);

    fluid_alpha_slot_ << new core::param::FloatParam(1.0f, 0.0f, 1.0f);
    MakeSlotAvailable(&fluid_alpha_slot_);

    interface_alpha_slot_ << new core::param::FloatParam(1.0f, 0.0f, 1.0f);
    MakeSlotAvailable(&interface_alpha_slot_);

    gas_alpha_slot_ << new core::param::FloatParam(1.0f, 0.0f, 1.0f);
    MakeSlotAvailable(&gas_alpha_slot_);
}


megamol::thermodyn::PhaseAnimator::~PhaseAnimator() { this->Release(); }


bool megamol::thermodyn::PhaseAnimator::create() { return true; }


void megamol::thermodyn::PhaseAnimator::release() {}


bool megamol::thermodyn::PhaseAnimator::getDataCallback(core::Call& c) {
    auto out_call = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (out_call == nullptr) return false;

    auto part_in_call = part_in_data_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (part_in_call == nullptr) return false;

    auto box_in_call = box_in_data_slot_.CallAs<BoxDataCall>();
    if (box_in_call == nullptr) return false;

    part_in_call->SetFrameID(out_call->FrameID(), true);
    if (!(*part_in_call)(1)) return false;
    if (!(*box_in_call)(1)) return false;

    if (part_in_call->DataHash() != data_hash_ || part_in_call->FrameID() != frame_id_ || isDirty()) {
        if (!(*part_in_call)(0)) return false;
        if (!(*box_in_call)(0)) return false;

        auto fluid_alpha = fluid_alpha_slot_.Param<core::param::FloatParam>()->Value();
        auto interface_alpha = interface_alpha_slot_.Param<core::param::FloatParam>()->Value();
        auto gas_alpha = gas_alpha_slot_.Param<core::param::FloatParam>()->Value();

        auto boxes = box_in_call->GetBoxes();

        auto const fluid_box =
            *std::find_if(boxes->cbegin(), boxes->cend(), [](auto const& el) { return el.name_ == "fluid"; });
        auto const interface_box =
            *std::find_if(boxes->cbegin(), boxes->cend(), [](auto const& el) { return el.name_ == "interface"; });
        auto const gas_box =
            *std::find_if(boxes->cbegin(), boxes->cend(), [](auto const& el) { return el.name_ == "gas"; });

        (*out_call) = (*part_in_call);

        auto const plc = out_call->GetParticleListCount();

        data_.resize(plc);

        for (unsigned int plidx = 0; plidx < plc; ++plidx) {
            auto& parts = out_call->AccessParticles(plidx);

            auto const pc = parts.GetCount();
            std::vector<float> alphas(pc);
            data_[plidx].resize(7 * pc);
            auto& ps = data_[plidx];

            auto const& store = parts.GetParticleStore();

            auto const& xAcc = store.GetXAcc();
            auto const& yAcc = store.GetYAcc();
            auto const& zAcc = store.GetZAcc();

            auto const& crAcc = store.GetCRAcc();
            auto const& cgAcc = store.GetCGAcc();
            auto const& cbAcc = store.GetCBAcc();
            auto const& caAcc = store.GetCAAcc();

            for (size_t pidx = 0; pidx < pc; ++pidx) {
                ps[7 * pidx + 0] = xAcc->Get_f(pidx);
                ps[7 * pidx + 1] = yAcc->Get_f(pidx);
                ps[7 * pidx + 2] = zAcc->Get_f(pidx);

                ps[7 * pidx + 3] = crAcc->Get_f(pidx) / 256.f;
                ps[7 * pidx + 4] = cgAcc->Get_f(pidx) / 256.f;
                ps[7 * pidx + 5] = cbAcc->Get_f(pidx) / 256.f;
                ps[7 * pidx + 6] = caAcc->Get_f(pidx) / 256.f;
            }

            // fluid
            auto const& fbbox = fluid_box.box_;
            for (size_t pidx = 0; pidx < pc; ++pidx) {
                if (fbbox.Contains(vislib::math::Point<float, 3>(ps[7 * pidx + 0], ps[7 * pidx + 1], ps[7 * pidx + 2]),
                        vislib::math::Cuboid<float>::FACE_ALL)) {
                    ps[7 * pidx + 6] = fluid_alpha;
                }
            }

            // interface
            auto const& ibbox = interface_box.box_;
            for (size_t pidx = 0; pidx < pc; ++pidx) {
                if (ibbox.Contains(vislib::math::Point<float, 3>(ps[7 * pidx + 0], ps[7 * pidx + 1], ps[7 * pidx + 2]),
                        vislib::math::Cuboid<float>::FACE_ALL)) {
                    ps[7 * pidx + 6] = interface_alpha;
                }
            }

            // gas
            auto const& gbbox = gas_box.box_;
            for (size_t pidx = 0; pidx < pc; ++pidx) {
                if (gbbox.Contains(vislib::math::Point<float, 3>(ps[7 * pidx + 0], ps[7 * pidx + 1], ps[7 * pidx + 2]),
                        vislib::math::Cuboid<float>::FACE_ALL)) {
                    ps[7 * pidx + 6] = gas_alpha;
                }
            }

            parts.SetVertexData(
                core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ, ps.data(), 7 * sizeof(float));
            parts.SetColourData(
                core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_RGBA, ps.data() + 3, 7 * sizeof(float));
        }

        data_hash_ = part_in_call->DataHash();
        ++out_data_hash_;
        frame_id_ = part_in_call->FrameID();
        resetDirty();
    }

    return true;
}


bool megamol::thermodyn::PhaseAnimator::getExtentCallback(core::Call& c) {
    auto inCall = part_in_data_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (inCall == nullptr) return false;

    auto outCall = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (outCall == nullptr) return false;

    if (!(*inCall)(1)) return false;

    outCall->AccessBoundingBoxes().SetObjectSpaceBBox(inCall->AccessBoundingBoxes().ObjectSpaceBBox());
    outCall->AccessBoundingBoxes().SetObjectSpaceClipBox(inCall->AccessBoundingBoxes().ObjectSpaceClipBox());
    outCall->AccessBoundingBoxes().MakeScaledWorld(1.0f);

    outCall->SetDataHash(out_data_hash_);
    outCall->SetFrameCount(inCall->FrameCount());
    outCall->SetFrameID(frame_id_);

    return true;
}
