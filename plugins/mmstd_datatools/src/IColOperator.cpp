#include "IColOperator.h"

#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"

#include "mmstd_datatools/MultiParticleDataAdaptor.h"


megamol::stdplugin::datatools::IColOperator::IColOperator()
        : stdplugin::datatools::AbstractParticleManipulator("outData", "inDataA")
        , in_dataB_slot_("inDataB", "Fetches the second ICol value stream")
        , primary_operator_slot_("primary operator", "")
        , a_operator_slot_("a operator", "")
        , b_operator_slot_("b operator", "")
        , a_value_slot_("a value", "")
        , b_value_slot_("b value", "")
        , inAHash_(0)
        , inBHash_(0)
        , outHash_(0)
        , frameID_(0)
        , colors_()
        , minCol_(0.0f)
        , maxCol_(1.0f) {
    in_dataB_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&in_dataB_slot_);

    using operator_type_ut = std::underlying_type_t<operator_type>;

    auto ep = new core::param::EnumParam(static_cast<int>(operator_type::ot_plus));
    ep->SetTypePair(static_cast<int>(operator_type::ot_plus), "plus");
    ep->SetTypePair(static_cast<int>(operator_type::ot_and), "and");
    ep->SetTypePair(static_cast<int>(operator_type::ot_less), "less");
    ep->SetTypePair(static_cast<int>(operator_type::ot_greater), "greater");
    primary_operator_slot_ << ep;
    MakeSlotAvailable(&primary_operator_slot_);

    auto ep2 = new core::param::EnumParam(static_cast<int>(operator_type::ot_plus));
    ep2->SetTypePair(static_cast<int>(operator_type::ot_plus), "plus");
    ep2->SetTypePair(static_cast<int>(operator_type::ot_and), "and");
    ep2->SetTypePair(static_cast<int>(operator_type::ot_less), "less");
    ep2->SetTypePair(static_cast<int>(operator_type::ot_greater), "greater");
    a_operator_slot_ << ep2;
    MakeSlotAvailable(&a_operator_slot_);

    auto ep3 = new core::param::EnumParam(static_cast<int>(operator_type::ot_plus));
    ep3->SetTypePair(static_cast<int>(operator_type::ot_plus), "plus");
    ep3->SetTypePair(static_cast<int>(operator_type::ot_and), "and");
    ep3->SetTypePair(static_cast<int>(operator_type::ot_less), "less");
    ep3->SetTypePair(static_cast<int>(operator_type::ot_greater), "greater");
    b_operator_slot_ << ep3;
    MakeSlotAvailable(&b_operator_slot_);

    a_value_slot_ << new core::param::FloatParam(0.f);
    MakeSlotAvailable(&a_value_slot_);

    b_value_slot_ << new core::param::FloatParam(0.f);
    MakeSlotAvailable(&b_value_slot_);
}


megamol::stdplugin::datatools::IColOperator::~IColOperator() {
    this->Release();
}


megamol::stdplugin::datatools::IColOperator::icol_operator
megamol::stdplugin::datatools::IColOperator::parse_icol_operator_type(
    megamol::stdplugin::datatools::IColOperator::operator_type type) {
    switch (type) {
    case megamol::stdplugin::datatools::IColOperator::operator_type::ot_and:
        return [](float a, float b) -> float { return a && b; };
    case megamol::stdplugin::datatools::IColOperator::operator_type::ot_less:
        return [](float a, float b) -> float { return a < b; };
    case megamol::stdplugin::datatools::IColOperator::operator_type::ot_greater:
        return [](float a, float b) -> float { return a > b; };
    case megamol::stdplugin::datatools::IColOperator::operator_type::ot_plus:
    default:
        return [](float a, float b) -> float { return a + b; };
    }
}


bool megamol::stdplugin::datatools::IColOperator::manipulateData(
    megamol::core::moldyn::MultiParticleDataCall& outData, megamol::core::moldyn::MultiParticleDataCall& inDataA) {
    core::moldyn::MultiParticleDataCall* inDataBptr = in_dataB_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (inDataBptr == nullptr)
        return false;
    core::moldyn::MultiParticleDataCall& inDataB = *inDataBptr;

    inDataB.SetFrameID(inDataA.FrameID(), true);
    if (!inDataB(0))
        return false;

    if ((inAHash_ != inDataA.DataHash()) || (inDataA.DataHash() == 0) || (inBHash_ != inDataB.DataHash()) ||
        (inDataB.DataHash() == 0) || (frameID_ != inDataA.FrameID()) || is_dirty()) {
        // Update data
        inAHash_ = inDataA.DataHash();
        inBHash_ = inDataB.DataHash();
        ++outHash_;
        frameID_ = inDataA.FrameID();
        reset_dirty();

        auto const primary_operator = primary_operator_slot_.Param<core::param::EnumParam>()->Value();
        auto const a_operator = a_operator_slot_.Param<core::param::EnumParam>()->Value();
        auto const b_operator = b_operator_slot_.Param<core::param::EnumParam>()->Value();

        auto const a_value = a_value_slot_.Param<core::param::FloatParam>()->Value();
        auto const b_value = b_value_slot_.Param<core::param::FloatParam>()->Value();

        stdplugin::datatools::MultiParticleDataAdaptor a(inDataA);
        stdplugin::datatools::MultiParticleDataAdaptor b(inDataB);

        if (a.get_count() != b.get_count()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("Data streams of A and B are not of same size");
            inDataB.Unlock();
            return false;
        }


        colors_.resize(a.get_count());
        if (colors_.size() > 0) {
            for (size_t i = 0; i < a.get_count(); ++i) {
                colors_[i] = (*a.get_color(i)) > a_value && (*b.get_color(i)) < b_value;
            }

            minCol_ = maxCol_ = colors_[0];
            for (size_t i = 1; i < a.get_count(); ++i) {
                if (minCol_ > colors_[i])
                    minCol_ = colors_[i];
                if (maxCol_ < colors_[i])
                    maxCol_ = colors_[i];
            }

        } else {
            minCol_ = 0.0f;
            maxCol_ = 1.0f;
        }
    }

    inDataB.Unlock();

    outData = inDataA;
    outData.SetDataHash(outHash_);
    outData.SetFrameID(frameID_);
    inDataA.SetUnlocker(nullptr, false);

    const float* data = colors_.data();
    for (unsigned int list = 0; list < outData.GetParticleListCount(); ++list) {
        auto& plist = outData.AccessParticles(list);
        plist.SetColourData(core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I, data, 0);
        plist.SetColourMapIndexValues(minCol_, maxCol_);
        data += plist.GetCount();
    }

    return true;
}
