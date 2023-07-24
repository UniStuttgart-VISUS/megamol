#include "AddParticleColors.h"

#include "mmstd/renderer/CallGetTransferFunction.h"


megamol::datatools::AddParticleColors::AddParticleColors()
        : AbstractParticleManipulator("outData", "indata")
        , _tf_slot("inTF", "") {
    _tf_slot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    MakeSlotAvailable(&_tf_slot);
}


megamol::datatools::AddParticleColors::~AddParticleColors() {
    this->Release();
}


float megamol::datatools::AddParticleColors::lerp(float a, float b, float inter) {
    return a * (1.0f - inter) + b * inter;
}


glm::vec4 megamol::datatools::AddParticleColors::sample_tf(
    float const* tf, unsigned int tf_size, int base, float rest) {
    if (base < 0 || tf_size == 0)
        return glm::vec4(0);
    auto const last_el = tf_size - 1;
    if (base >= last_el)
        return glm::vec4(tf[last_el * 4], tf[last_el * 4 + 1], tf[last_el * 4 + 2], tf[last_el * 4 + 3]);

    auto const a = base;
    auto const b = base + 1;

    return glm::vec4(lerp(tf[a * 4], tf[b * 4], rest), lerp(tf[a * 4 + 1], tf[b * 4 + 1], rest),
        lerp(tf[a * 4 + 2], tf[b * 4 + 2], rest), lerp(tf[a * 4 + 3], tf[b * 4 + 3], rest));
}


bool megamol::datatools::AddParticleColors::manipulateData(
    geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) {

    core::view::CallGetTransferFunction* cgtf = _tf_slot.CallAs<core::view::CallGetTransferFunction>();
    if (cgtf == nullptr)
        return false;
    if (!(*cgtf)())
        return false;

    outData = inData;

    if (_frame_id != inData.FrameID() || _in_data_hash != inData.DataHash() || cgtf->IsDirty()) {
        auto const tf = cgtf->GetTextureData();
        auto const tf_size = cgtf->TextureSize();

        auto const pl_count = outData.GetParticleListCount();
        _colors.clear();
        _colors.resize(pl_count);

        float min_i = std::numeric_limits<float>::max();
        float max_i = std::numeric_limits<float>::lowest();
        for (unsigned int plidx = 0; plidx < pl_count; ++plidx) {
            auto& parts = outData.AccessParticles(plidx);
            min_i = std::min(min_i, parts.GetMinColourIndexValue());
            max_i = std::max(max_i,parts.GetMaxColourIndexValue());
        }
        // Update data set (if new data set was loaded, or if frame changed)
        if (_frame_id != inData.FrameID() || _in_data_hash != inData.DataHash()) {
            cgtf->SetRange(std::array<float, 2>{min_i, max_i});
            (*cgtf)();
        }
        auto actual_range = cgtf->Range();
        min_i = actual_range[0];
        max_i = actual_range[1];
        auto const fac_i = 1.0f / (max_i - min_i + 1e-8f);

        for (unsigned int plidx = 0; plidx < pl_count; ++plidx) {
            auto& parts = outData.AccessParticles(plidx);
            if (parts.GetColourDataType() != geocalls::SimpleSphericalParticles::COLDATA_FLOAT_I &&
                parts.GetColourDataType() != geocalls::SimpleSphericalParticles::COLDATA_DOUBLE_I)
                continue;

            auto const p_count = parts.GetCount();
            auto& col_vec = _colors[plidx];
            col_vec.resize(p_count);


            auto const iAcc = parts.GetParticleStore().GetCRAcc();

            for (std::size_t pidx = 0; pidx < p_count; ++pidx) {
                auto const col = iAcc->Get_f(pidx);
                auto const val = (col - min_i) * fac_i * static_cast<float>(tf_size);
                std::remove_const_t<decltype(val)> main = 0;
                auto rest = std::modf(val, &main);
                col_vec[pidx].rgba = sample_tf(tf, tf_size, static_cast<int>(main), rest);
            }
        }


        _frame_id = inData.FrameID();
        _in_data_hash = inData.DataHash();
        ++_out_data_hash;
        cgtf->ResetDirty();
    }

    auto const pl_count = outData.GetParticleListCount();
    for (unsigned int plidx = 0; plidx < pl_count; ++plidx) {
        auto& parts = outData.AccessParticles(plidx);
        if (parts.GetColourDataType() != geocalls::SimpleSphericalParticles::COLDATA_FLOAT_I &&
            parts.GetColourDataType() != geocalls::SimpleSphericalParticles::COLDATA_DOUBLE_I)
            continue;
        parts.SetColourData(geocalls::SimpleSphericalParticles::COLDATA_FLOAT_RGBA, _colors[plidx].data());
    }

    outData.SetDataHash(_out_data_hash);
    inData.SetUnlocker(nullptr, false);

    return true;
}
