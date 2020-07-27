#include "stdafx.h"
#include "IColToIdentity.h"


megamol::stdplugin::datatools::IColToIdentity::IColToIdentity(void)
    : AbstractParticleManipulator("outData", "indata") {}


megamol::stdplugin::datatools::IColToIdentity::~IColToIdentity(void) { this->Release(); };


bool megamol::stdplugin::datatools::IColToIdentity::manipulateData(
    megamol::core::moldyn::MultiParticleDataCall& outData, megamol::core::moldyn::MultiParticleDataCall& inData) {
    using megamol::core::moldyn::MultiParticleDataCall;

    outData = inData; // also transfers the unlocker to 'outData'

    inData.SetUnlocker(nullptr, false); // keep original data locked
                                        // original data will be unlocked through outData

    auto const plc = outData.GetParticleListCount();
    for (unsigned int i = 0; i < plc; ++i) {
        auto& p = outData.AccessParticles(i);

        if (p.GetColourDataType() != core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I &&
            p.GetColourDataType() != core::moldyn::SimpleSphericalParticles::COLDATA_DOUBLE_I) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn("IColToIdentity: Particlelist %d has no intensity\n", i);
            continue;
        }

        auto const cnt = p.GetCount();

        core::moldyn::SimpleSphericalParticles::IDDataType idt;

        if (p.GetColourDataType() == core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I) {
            this->ids.resize(cnt * sizeof(unsigned int));
            auto const basePtrOut = reinterpret_cast<unsigned int*>(this->ids.data());
            auto& iCol = p.GetParticleStore().GetRAcc();

            for (size_t pidx = 0; pidx < cnt; ++pidx) {
                basePtrOut[pidx] = static_cast<unsigned int>(iCol->Get_u32(pidx));
            }
            idt = core::moldyn::SimpleSphericalParticles::IDDATA_UINT32;
        } else {
            this->ids.resize(cnt * sizeof(uint64_t));
            auto const basePtrOut = reinterpret_cast<uint64_t*>(this->ids.data());
            auto& iCol = p.GetParticleStore().GetRAcc();

            for (size_t pidx = 0; pidx < cnt; ++pidx) {
                basePtrOut[pidx] = static_cast<unsigned int>(iCol->Get_u64(pidx));
            }
            idt = core::moldyn::SimpleSphericalParticles::IDDATA_UINT64;
        }

        p.SetIDData(idt, this->ids.data());
    }

    return true;
}
