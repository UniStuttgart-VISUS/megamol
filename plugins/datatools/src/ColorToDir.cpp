#include "ColorToDir.h"


megamol::datatools::ColorToDir::ColorToDir() : AbstractParticleManipulator("outData", "inData") {}


megamol::datatools::ColorToDir::~ColorToDir() {
    this->Release();
}


bool megamol::datatools::ColorToDir::manipulateData(
    geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) {
    /*if (!(inData)(0))
        return false;*/
    outData = inData;
    if (frame_id_ != inData.FrameID() || in_data_hash_ != inData.DataHash()) {
        data_.clear();
        auto const pl_count = outData.GetParticleListCount();
        data_.resize(pl_count);

        for (std::remove_cv_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
            auto& parts = outData.AccessParticles(pl_idx);
            if (parts.GetColourDataType() == geocalls::SimpleSphericalParticles::COLDATA_FLOAT_RGB ||
                parts.GetColourDataType() == geocalls::SimpleSphericalParticles::COLDATA_FLOAT_RGBA) {
                parts.SetDirData(geocalls::SimpleSphericalParticles::DIRDATA_FLOAT_XYZ, parts.GetColourData(),
                    parts.GetColourDataStride() == 0
                        ? geocalls::SimpleSphericalParticles::ColorDataSize[parts.GetColourDataType()]
                        : parts.GetColourDataStride());
            } else if (parts.GetColourDataType() != geocalls::SimpleSphericalParticles::COLDATA_NONE) {
                auto& data = data_[pl_idx];
                auto const p_count = parts.GetCount();

                auto const dx_acc = parts.GetParticleStore().GetCRAcc();
                auto const dy_acc = parts.GetParticleStore().GetCGAcc();
                auto const dz_acc = parts.GetParticleStore().GetCBAcc();

                data.resize(3 * p_count);

                for (std::remove_cv_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx) {
                    data[p_idx * 3 + 0] = dx_acc->Get_f(p_idx);
                    data[p_idx * 3 + 1] = dy_acc->Get_f(p_idx);
                    data[p_idx * 3 + 2] = dz_acc->Get_f(p_idx);
                }

                parts.SetDirData(geocalls::SimpleSphericalParticles::DIRDATA_FLOAT_XYZ, data.data());
            }
        }

        frame_id_ = inData.FrameID();
        in_data_hash_ = inData.DataHash();
        ++out_data_hash_;
    }

    outData.SetDataHash(out_data_hash_);
    inData.SetUnlocker(nullptr, false);

    return true;
}
