#include "stdafx.h"
#include "DumpIColTrend.h"

#include "mmcore/moldyn/DirectionalParticleDataCall.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/IntParam.h"

#include "vislib/math/mathfunctions.h"


megamol::stdplugin::datatools::DumpIColTrend::DumpIColTrend()
    : dataInSlot_("dataIn", "Data input")
    , dumpSlot_("dump", "Dumps the trend onto disk")
    , axisSlot_("axis", "Axis selector")
    , numBucketsSlot_("numBuckets", "Number of buckets to sample the data")
    , frameSlot_("frameNum", "Number of frame to calculate the trend for") {
    dataInSlot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    dataInSlot_.SetCompatibleCall<core::moldyn::DirectionalParticleDataCallDescription>();
    MakeSlotAvailable(&dataInSlot_);

    dumpSlot_ << new core::param::ButtonParam();
    dumpSlot_.SetUpdateCallback(&DumpIColTrend::dump);
    MakeSlotAvailable(&dumpSlot_);

    auto ep = new core::param::EnumParam(0);
    ep->SetTypePair(0, "x");
    ep->SetTypePair(1, "y");
    ep->SetTypePair(2, "z");
    axisSlot_ << ep;
    MakeSlotAvailable(&axisSlot_);

    numBucketsSlot_ << new core::param::IntParam(100, 1, std::numeric_limits<int>::max());
    MakeSlotAvailable(&numBucketsSlot_);

    frameSlot_ << new core::param::IntParam(0, 0, std::numeric_limits<int>::max());
    MakeSlotAvailable(&frameSlot_);
}


megamol::stdplugin::datatools::DumpIColTrend::~DumpIColTrend() { this->Release(); }


bool megamol::stdplugin::datatools::DumpIColTrend::create() { return true; }


void megamol::stdplugin::datatools::DumpIColTrend::release() {}


bool megamol::stdplugin::datatools::DumpIColTrend::dump(core::param::ParamSlot& p) {
    auto inCall = dataInSlot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (inCall == nullptr) return false;

    if (!(*inCall)(1)) return false;

    auto const frameCount = inCall->FrameCount();
    auto reqFrame = frameSlot_.Param<core::param::IntParam>()->Value();
    reqFrame = vislib::math::Clamp<int>(reqFrame, 0, frameCount - 1);

    inCall->SetFrameID(reqFrame, true);
    if (!(*inCall)(0)) return false;

    auto const numBuckets = numBucketsSlot_.Param<core::param::IntParam>()->Value();
    auto const axis = axisSlot_.Param<core::param::EnumParam>()->Value();

    auto const bbox = inCall->AccessBoundingBoxes().ObjectSpaceBBox();
    auto const bbox_ptr = bbox.PeekBounds();
    auto const length = bbox_ptr[axis + 3] - bbox_ptr[axis];
    auto const min_pos = bbox_ptr[axis];
    auto const diff = length / numBuckets;

    auto const plc = inCall->GetParticleListCount();

    std::vector<std::vector<float>> storage(plc);
    std::vector<float> mids(numBuckets);
    auto const hdiff = diff*0.5f;
    for (size_t idx = 0; idx < numBuckets; ++idx) {
        mids[idx] = idx*diff+hdiff;
    }

    for (unsigned int plidx = 0; plidx < plc; ++plidx) {
        auto& trend = storage[plidx];
        trend.resize(numBuckets);

        std::vector<size_t> numSamples(numBuckets);

        auto const& parts = inCall->AccessParticles(plidx);
        auto const& store = parts.GetParticleStore();
        std::shared_ptr<core::moldyn::Accessor> acc;
        auto const& i_acc = store.GetRAcc();
        switch (axis) {
        case 0:
            acc = store.GetXAcc();
            break;
        case 1:
            acc = store.GetYAcc();
            break;
        default:
        case 2:
            acc = store.GetZAcc();
        }
        auto const pc = parts.GetCount();
        for (size_t pidx = 0; pidx < pc; ++pidx) {
            auto const pos = acc->Get_f(pidx) - min_pos;
            auto const val = i_acc->Get_f(pidx);
            auto idx = static_cast<size_t>(std::floorf(pos / diff));
            idx = vislib::math::Clamp<size_t>(idx, 0, numBuckets - 1);
            numSamples[idx]++;
            trend[idx] += val / numSamples[idx];
        }

        std::string filename = std::string("dump_trend_")+std::to_string(plidx)+std::string(".dat");
        dumpTrend(filename, trend, mids);
    }

    return true;
}
