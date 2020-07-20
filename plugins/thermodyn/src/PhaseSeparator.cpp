#include "stdafx.h"
#include "PhaseSeparator.h"

#include <numeric>
#include <cmath>

#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"

#include "thermodyn/BoxDataCall.h"


megamol::thermodyn::PhaseSeparator::PhaseSeparator()
    : dataInSlot_("dataIn", "Input of particle data")
    , dataOutSlot_("dataOut", "Ouput of boxes representing the phases")
    , criticalTempSlot_("Tc", "Critical temperature of simulated material")
    , ensembleTempSlot_("T", "Temperature set by the ensemble")
    , fluidColorSlot_("fluidColor", "Color of the box representing the fluid")
    , interfaceColorSlot_("interfaceColor", "Color of the box representing the interface")
    , gasColorSlot_("gasColor", "Color of the box representing the gas")
    , axisSlot_("axis", "Main axis for density analysis")
    , numSlicesSlot_("numSlices", "Number of slices for density analysis") {
    dataInSlot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&dataInSlot_);

    dataOutSlot_.SetCallback(BoxDataCall::ClassName(), BoxDataCall::FunctionName(0), &PhaseSeparator::getDataCallback);
    dataOutSlot_.SetCallback(
        BoxDataCall::ClassName(), BoxDataCall::FunctionName(1), &PhaseSeparator::getExtentCallback);
    MakeSlotAvailable(&dataOutSlot_);

    criticalTempSlot_ << new core::param::FloatParam(
        1.0f, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());
    MakeSlotAvailable(&criticalTempSlot_);

    ensembleTempSlot_ << new core::param::FloatParam(
        1.0f, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());
    MakeSlotAvailable(&ensembleTempSlot_);

    fluidColorSlot_ << new core::param::StringParam("1.0, 1.0, 1.0, 1.0");
    MakeSlotAvailable(&fluidColorSlot_);

    interfaceColorSlot_ << new core::param::StringParam("1.0, 1.0, 1.0, 1.0");
    MakeSlotAvailable(&interfaceColorSlot_);

    gasColorSlot_ << new core::param::StringParam("1.0, 1.0, 1.0, 1.0");
    MakeSlotAvailable(&gasColorSlot_);

    auto ep = new core::param::EnumParam(0);
    ep->SetTypePair(0, "x");
    ep->SetTypePair(1, "y");
    ep->SetTypePair(2, "z");
    axisSlot_ << ep;
    MakeSlotAvailable(&axisSlot_);

    numSlicesSlot_ << new core::param::IntParam(100, 1, std::numeric_limits<int>::max());
    MakeSlotAvailable(&numSlicesSlot_);
}


megamol::thermodyn::PhaseSeparator::~PhaseSeparator() { this->Release(); }


bool megamol::thermodyn::PhaseSeparator::create() { return true; }


void megamol::thermodyn::PhaseSeparator::release() {}


bool megamol::thermodyn::PhaseSeparator::getDataCallback(core::Call& c) {
    // assuming we get the local density as ICol in the particle stream

    auto const Tc = criticalTempSlot_.Param<core::param::FloatParam>()->Value();
    auto const T = ensembleTempSlot_.Param<core::param::FloatParam>()->Value();

    // Interface width
    auto const D = 1.720f * std::pow((Tc - T) / Tc, 1.89f) + 1.103 * std::pow((Tc - T) / Tc, -0.62f);

    auto inCall = dataInSlot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (inCall == nullptr) return false;

    auto outCall = dynamic_cast<BoxDataCall*>(&c);
    if (outCall == nullptr) return false;

    inCall->SetFrameID(outCall->FrameID(), true);
    if (!(*inCall)(1)) return false;
    if (!(*inCall)(0)) return false;

    if (inCall->DataHash() != inDataHash_ || inCall->FrameID() != frameID_) {
        auto const plc = inCall->GetParticleListCount();
        if (plc > 1) {
            vislib::sys::Log::DefaultLog.WriteWarn("PhaseSeparator: You have to select a specific list entry\n");
            return false;
        }

        auto const& bbox = inCall->AccessBoundingBoxes().ObjectSpaceBBox();
        auto const& parts = inCall->AccessParticles(0);

        if (parts.GetColourDataType() != core::moldyn::SimpleSphericalParticles::ColourDataType::COLDATA_FLOAT_I &&
            parts.GetColourDataType() != core::moldyn::SimpleSphericalParticles::ColourDataType::COLDATA_DOUBLE_I) {
            vislib::sys::Log::DefaultLog.WriteWarn("PhaseSeparator: Require density as ICol stream\n");
            return false;
        }

        auto const pc = parts.GetCount();

        auto const& store = parts.GetParticleStore();
        auto const& xAcc = store.GetXAcc();
        auto const& yAcc = store.GetYAcc();
        auto const& zAcc = store.GetZAcc();
        auto const& iAcc = store.GetCRAcc();
        std::shared_ptr<core::moldyn::Accessor> pAcc;

        auto const axis = axisSlot_.Param<core::param::EnumParam>()->Value();
        auto const numSlices = numSlicesSlot_.Param<core::param::IntParam>()->Value();

        std::vector<float> trend(numSlices, 0.0f);
        std::vector<size_t> cnt(numSlices, 0);
        float offset = 0.0f;
        auto diff = 1.0f;
        if (axis == 0) {
            diff = bbox.Width() / (numSlices);
            pAcc = xAcc;
            offset = bbox.GetLeft();
        } else if (axis == 1) {
            diff = bbox.Height() / (numSlices);
            pAcc = yAcc;
            offset = bbox.GetBottom();
        } else if (axis == 2) {
            diff = bbox.Depth() / (numSlices);
            pAcc = zAcc;
            offset = bbox.GetBack();
        }

        for (size_t pidx = 0; pidx < pc; ++pidx) {
            auto const pos = pAcc->Get_f(pidx) - offset;
            auto const val = iAcc->Get_f(pidx);
            auto idx = static_cast<size_t>(std::floor(pos / diff));
            idx = vislib::math::Clamp<size_t>(idx, 0, numSlices - 1);
            ++cnt[idx];
            trend[idx] += val / cnt[idx];
        }

        // determine interface
        auto tmp = trend;
        tmp.erase(std::remove(tmp.begin(), tmp.end(), 0.0f), tmp.end());
        auto const minmax = std::minmax_element(tmp.begin(), tmp.end());
        auto max = *minmax.second;
        auto min = *minmax.first;
        auto mid = (0.5f * (max - min)) + min;
        auto perc = 0.1f * (max - min);

        auto smooth = tmp;
        smooth.erase(
            std::remove_if(smooth.begin(), smooth.end(), [&max, &perc](auto const& a) { return !(a > max - perc); }),
            smooth.end());
        // std::nth_element(smooth.begin(), smooth.begin()+(smooth.size()/2), smooth.end());
        auto val = std::accumulate(smooth.begin(), smooth.end(), 0.0f);
        // max = smooth[smooth.size()/2];
        max = val / static_cast<float>(smooth.size());

        smooth = tmp;
        smooth.erase(
            std::remove_if(smooth.begin(), smooth.end(), [&min, &perc](auto const& a) { return !(a < min + perc); }),
            smooth.end());
        // std::nth_element(smooth.begin(), smooth.begin()+(smooth.size()/2), smooth.end());
        val = std::accumulate(smooth.begin(), smooth.end(), 0.0f);
        // min = smooth[smooth.size()/2];
        min = val / static_cast<float>(smooth.size());

        /*mid = (0.5f * (max - min)) + min;
        perc = 0.1f * (max-min);*/


        /*auto rit = trend.crbegin();
        for (; rit != trend.crend(); ++rit) {
            if (*rit > 0.0f && *rit >= min+perc) break;
        }
        auto last = rit;
        for (; rit != trend.crend(); ++rit) {
            if (*rit >= max-perc) break;
        }
        auto first = rit;
        auto l_idx = numSlices - std::distance(trend.crbegin(), last);
        auto f_idx = numSlices - std::distance(trend.crbegin(), first);*/

        auto rit = trend.crbegin();
        auto last = rit;
        auto first = rit;
        bool totalBreak = false;
        for (; rit != trend.crend(); ++rit) {
            if (totalBreak) break;
            if (*rit > max - perc) {
                first = rit;
                for (; rit > trend.crbegin(); --rit) {
                    if (*rit < min + perc) {
                        last = rit;
                        totalBreak = true;
                        break;
                    }
                }
            }
        }

        /*for (; rit != trend.crend(); ++rit) {
            if (*rit > max-perc) break;
        }*/

        auto l_idx = numSlices - 1 - std::distance(trend.crbegin(), last);
        auto f_idx = numSlices - 1 - std::distance(trend.crbegin(), first);

        /*auto rit = trend.cbegin();
        for (; rit != trend.cend(); ++rit) {
            if (*rit > 0.0f && *rit < max-perc) break;
        }
        auto first = rit;
        for (; rit != trend.cend(); ++rit) {
            if (*rit < min+perc) break;
        }
        auto last = rit;
        auto l_idx = std::distance(trend.cbegin(), last);
        auto f_idx = std::distance(trend.cbegin(), first);*/

        auto const idx = (l_idx - f_idx) / 2 + f_idx;

        auto const int_pos = (idx + 0.5f) * diff;
        auto lower_b = int_pos - (0.5f * D);
        auto higher_b = int_pos + (0.5f * D);

        // determine boxes
        auto fluidbox = bbox;
        auto interfacebox = bbox;
        auto gasbox = bbox;
        if (axis == 0) {
            fluidbox.SetRight(lower_b);
            interfacebox.SetLeft(lower_b);
            interfacebox.SetRight(higher_b);
            gasbox.SetLeft(higher_b);
        } else if (axis == 1) {
            fluidbox.SetTop(lower_b);
            interfacebox.SetBottom(lower_b);
            interfacebox.SetTop(higher_b);
            gasbox.SetBottom(higher_b);
        } else if (axis == 2) {
            fluidbox.SetFront(lower_b);
            interfacebox.SetBack(lower_b);
            interfacebox.SetFront(higher_b);
            gasbox.SetBack(higher_b);
        }

        BoxDataCall::box_entry_t fluid_be;
        fluid_be.box_ = fluidbox;
        fluid_be.name_ = std::string("fluid");
        auto const fluidColor = fluidColorSlot_.Param<core::param::StringParam>()->Value();
        auto fluid_c = getColor(fluidColor);
        memcpy(fluid_be.color_, fluid_c.data(), 4 * sizeof(float));

        BoxDataCall::box_entry_t interface_be;
        interface_be.box_ = interfacebox;
        interface_be.name_ = std::string("interface");
        auto const interfaceColor = interfaceColorSlot_.Param<core::param::StringParam>()->Value();
        auto interface_c = getColor(interfaceColor);
        memcpy(interface_be.color_, interface_c.data(), 4 * sizeof(float));

        BoxDataCall::box_entry_t gas_be;
        gas_be.box_ = gasbox;
        gas_be.name_ = std::string("gas");
        auto const gasColor = gasColorSlot_.Param<core::param::StringParam>()->Value();
        auto gas_c = getColor(gasColor);
        memcpy(gas_be.color_, gas_c.data(), 4 * sizeof(float));

        boxes_.clear();
        boxes_.push_back(fluid_be);
        boxes_.push_back(interface_be);
        boxes_.push_back(gas_be);

        inDataHash_ = inCall->DataHash();
        frameID_ = inCall->FrameID();
    }

    outCall->SetBoxes(&boxes_);

    outCall->SetDataHash(inDataHash_);
    outCall->SetFrameCount(inCall->FrameCount());
    outCall->SetFrameID(frameID_);

    return true;
}


bool megamol::thermodyn::PhaseSeparator::getExtentCallback(core::Call& c) {
    auto inCall = dataInSlot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (inCall == nullptr) return false;

    auto outCall = dynamic_cast<BoxDataCall*>(&c);
    if (outCall == nullptr) return false;

    if (!(*inCall)(1)) return false;

    outCall->AccessBoundingBoxes().SetObjectSpaceBBox(inCall->AccessBoundingBoxes().ObjectSpaceBBox());
    outCall->AccessBoundingBoxes().SetObjectSpaceClipBox(inCall->AccessBoundingBoxes().ObjectSpaceClipBox());
    outCall->AccessBoundingBoxes().MakeScaledWorld(1.0f);

    outCall->SetDataHash(inDataHash_);
    outCall->SetFrameCount(inCall->FrameCount());
    outCall->SetFrameID(frameID_);

    return true;
}
