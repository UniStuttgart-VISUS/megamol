#include "stdafx.h"
#include "DensityProfile.h"

#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/moldyn/DirectionalParticleDataCall.h"

#include "mmcore/param/FloatParam.h"

#include "vislib/sys/ConsoleProgressBar.h"


megamol::stdplugin::datatools::DensityProfile::DensityProfile()
    : megamol::core::Module()
    , outDataSlot("outData", "Output of diagram")
    , inDataSlot("inData", "Input of particles")
    , sliceSizeFactorSlot("sliceSizeFactor", "Size of slice as factor of particle diameter")
    , datahash{std::numeric_limits<size_t>::max()} {
    outDataSlot.SetCallback(megamol::stdplugin::datatools::floattable::CallFloatTableData::ClassName(),
        megamol::stdplugin::datatools::floattable::CallFloatTableData::FunctionName(0),
        &DensityProfile::getDataCallback);
    outDataSlot.SetCallback(megamol::stdplugin::datatools::floattable::CallFloatTableData::ClassName(),
        megamol::stdplugin::datatools::floattable::CallFloatTableData::FunctionName(1),
        &DensityProfile::getHashCallback);
    this->MakeSlotAvailable(&this->outDataSlot);

    inDataSlot.SetCompatibleCall<megamol::core::moldyn::MultiParticleDataCallDescription>();
    inDataSlot.SetCompatibleCall<megamol::core::moldyn::DirectionalParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);

    sliceSizeFactorSlot << new megamol::core::param::FloatParam(1.0f,
        std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
    this->MakeSlotAvailable(&this->sliceSizeFactorSlot);
}


megamol::stdplugin::datatools::DensityProfile::~DensityProfile() {
    this->Release();
}


bool megamol::stdplugin::datatools::DensityProfile::create() {
    data_ci_.SetName("density");
    data_ci_.SetType(megamol::stdplugin::datatools::floattable::CallFloatTableData::ColumnType::QUANTITATIVE);

    return true;
}


void megamol::stdplugin::datatools::DensityProfile::release() {

}


bool megamol::stdplugin::datatools::DensityProfile::getDataCallback(megamol::core::Call& c) {
    megamol::stdplugin::datatools::floattable::CallFloatTableData* outCall
        = dynamic_cast<megamol::stdplugin::datatools::floattable::CallFloatTableData*>(&c);

    if (outCall == nullptr) return false;

    if (!assertData()) return false;

    outCall->SetDataHash(this->datahash);
    outCall->SetFrameCount(1);
    outCall->SetFrameID(0);
    outCall->Set(1, this->data_.size(), &this->data_ci_, this->data_.data());

    return true;
}


bool megamol::stdplugin::datatools::DensityProfile::getHashCallback(megamol::core::Call& c) {
    megamol::stdplugin::datatools::floattable::CallFloatTableData* outCall
        = dynamic_cast<megamol::stdplugin::datatools::floattable::CallFloatTableData*>(&c);

    if (outCall == nullptr) return false;

    if (!assertData()) return false;

    outCall->SetDataHash(this->datahash);
    outCall->SetFrameCount(1);
    outCall->SetFrameID(0);

    return true;
}


bool megamol::stdplugin::datatools::DensityProfile::assertData(void) {
    megamol::core::moldyn::DirectionalParticleDataCall* dirInCall
        = this->inDataSlot.CallAs<megamol::core::moldyn::DirectionalParticleDataCall>();

    megamol::core::moldyn::MultiParticleDataCall* simInCall
        = this->inDataSlot.CallAs<megamol::core::moldyn::MultiParticleDataCall>();

    if (dirInCall == nullptr && simInCall == nullptr) return false;

    megamol::core::AbstractGetData3DCall* inCall = dirInCall ? dynamic_cast<megamol::core::AbstractGetData3DCall*>(dirInCall)
        : dynamic_cast<megamol::core::AbstractGetData3DCall*>(simInCall);

    (*inCall)(1);
    (*inCall)(0);

    if (this->datahash != inCall->DataHash()) {
        this->datahash = inCall->DataHash();

        auto const frameCount = inCall->FrameCount();

        if (frameCount == 0) return false;

        auto const parListCount = dirInCall ? dirInCall->GetParticleListCount() : simInCall->GetParticleListCount();

        if (parListCount == 0) return false;

        auto const bbox = inCall->AccessBoundingBoxes().ObjectSpaceBBox();

        unsigned int lng_edge_idx{0};
        float lng_edge{std::numeric_limits<float>::lowest()};
        for (unsigned int d = 0; d < 3; ++d) {
            if (std::abs(bbox.PeekBounds()[d + 3] - bbox.PeekBounds()[d]) > lng_edge) {
                lng_edge = std::abs(bbox.PeekBounds()[d + 3] - bbox.PeekBounds()[d]);
                lng_edge_idx = d;
            }
        }

        float const sliceSizeFactor = this->sliceSizeFactorSlot.Param<megamol::core::param::FloatParam>()->Value();

        // get minimal par radius
        float min_rad{std::numeric_limits<float>::max()};
        for (unsigned int pli = 0; pli < parListCount; ++pli) {
            auto const& parts = dirInCall ? dirInCall->AccessParticles(pli) : simInCall->AccessParticles(pli);

            if (parts.GetGlobalRadius() < min_rad) {
                min_rad = parts.GetGlobalRadius();
            }
        }

        float tmp_slice_size = min_rad * sliceSizeFactor;
        float const slice_size = lng_edge / std::ceilf(lng_edge / tmp_slice_size);
        unsigned int const bucket_count = static_cast<unsigned int>(std::ceilf(lng_edge / slice_size));

        printf("HURZ: %f", slice_size);

        std::vector<uint64_t> slices(bucket_count, 0);

        vislib::sys::ConsoleProgressBar cpb;
        cpb.Start("DensityProfile", frameCount);

        for (unsigned int fi = 0; fi < frameCount; ++fi) {
            do {
                inCall->SetFrameID(fi, true);
                (*inCall)(0);
            } while (inCall->FrameID() != fi);

            for (unsigned int pli = 0; pli < parListCount; ++pli) {
                auto const& parts = dirInCall ? dirInCall->AccessParticles(pli) : simInCall->AccessParticles(pli);
                auto const part_count = parts.GetCount();
                for (uint64_t par_i = 0; par_i < part_count; ++par_i) {
                    megamol::core::moldyn::SimpleSphericalParticles::particle_t par = parts[par_i];
                    float par_pos[3] = {par.vert.GetXf(), par.vert.GetYf(), par.vert.GetZf()};
                    unsigned int bucket_idx = static_cast<unsigned int>(std::floorf(par_pos[lng_edge_idx] / slice_size)) >= bucket_count
                        ? bucket_count - 1 : static_cast<unsigned int>(std::floorf(par_pos[lng_edge_idx] / slice_size));
                    ++(slices[bucket_idx]);
                }
            }

            cpb.Set(fi);
        }

        cpb.Stop();

        float slice_volume{slice_size};
        for (unsigned int d = 0; d < 3; ++d) {
            if (d != lng_edge_idx) {
                slice_volume *= std::abs(bbox.PeekBounds()[d + 3] - bbox.PeekBounds()[d]);
            }
        }

        data_.resize(slices.size());
        std::transform(slices.begin(), slices.end(), data_.begin(), [&](auto const& val)->float {return static_cast<float>(val / frameCount) / slice_volume; });

        auto tmp_minmax = std::minmax_element(data_.begin(), data_.end());
        data_minmax_.first = *(tmp_minmax.first);
        data_minmax_.second = *(tmp_minmax.second);

        data_ci_.SetMinimumValue(data_minmax_.first);
        data_ci_.SetMaximumValue(data_minmax_.second);
    }

    return true;
}
