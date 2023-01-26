#include "IColToIdentity.h"


megamol::datatools::IColToIdentity::IColToIdentity() : AbstractParticleManipulator("outData", "indata") {}


megamol::datatools::IColToIdentity::~IColToIdentity() {
    this->Release();
};


bool megamol::datatools::IColToIdentity::manipulateData(
    geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) {
    using geocalls::MultiParticleDataCall;

    outData = inData; // also transfers the unlocker to 'outData'

    inData.SetUnlocker(nullptr, false); // keep original data locked
                                        // original data will be unlocked through outData

    auto const plc = outData.GetParticleListCount();
    for (unsigned int i = 0; i < plc; ++i) {
        auto& p = outData.AccessParticles(i);

        if (p.GetColourDataType() != geocalls::SimpleSphericalParticles::COLDATA_FLOAT_I &&
            p.GetColourDataType() != geocalls::SimpleSphericalParticles::COLDATA_DOUBLE_I) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "IColToIdentity: Particlelist %d has no intensity\n", i);
            continue;
        }

        auto const cnt = p.GetCount();

        geocalls::SimpleSphericalParticles::IDDataType idt;

        if (p.GetColourDataType() == geocalls::SimpleSphericalParticles::COLDATA_FLOAT_I) {
            this->ids.resize(cnt * sizeof(unsigned int));
            auto const basePtrOut = reinterpret_cast<unsigned int*>(this->ids.data());
            auto& iCol = p.GetParticleStore().GetRAcc();

            for (size_t pidx = 0; pidx < cnt; ++pidx) {
                basePtrOut[pidx] = static_cast<unsigned int>(iCol->Get_u32(pidx));
            }
            idt = geocalls::SimpleSphericalParticles::IDDATA_UINT32;
        } else {
            this->ids.resize(cnt * sizeof(uint64_t));
            auto const basePtrOut = reinterpret_cast<uint64_t*>(this->ids.data());
            auto& iCol = p.GetParticleStore().GetRAcc();

            for (size_t pidx = 0; pidx < cnt; ++pidx) {
                basePtrOut[pidx] = static_cast<unsigned int>(iCol->Get_u64(pidx));
            }
            idt = geocalls::SimpleSphericalParticles::IDDATA_UINT64;
        }

        p.SetIDData(idt, this->ids.data());
    }

    return true;
}
