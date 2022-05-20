#include "ParticleIdentitySort.h"
#include <numeric>


megamol::datatools::ParticleIdentitySort::ParticleIdentitySort(void)
        : AbstractParticleManipulator("outData", "indata") {}


megamol::datatools::ParticleIdentitySort::~ParticleIdentitySort(void) {
    this->Release();
};


bool megamol::datatools::ParticleIdentitySort::manipulateData(
    geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) {
    using geocalls::MultiParticleDataCall;

    outData = inData; // also transfers the unlocker to 'outData'

    inData.SetUnlocker(nullptr, false); // keep original data locked
                                        // original data will be unlocked through outData

    auto const plc = outData.GetParticleListCount();
    this->data_.clear();
    for (unsigned int i = 0; i < plc; ++i) {
        auto& p = outData.AccessParticles(i);

        if (p.GetIDDataType() == geocalls::SimpleSphericalParticles::IDDATA_NONE) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "ParticleIdentitySort: Particlelist %d has no indentity array\n", i);
            continue;
        }

        this->data_.emplace_back();
        auto& dlist = this->data_.back();

        auto vs = p.GetVertexDataStride();
        auto cs = p.GetColourDataStride();
        auto is = p.GetIDDataStride();

        auto avs = vs;
        auto acs = cs;
        auto ais = is;

        auto const vp = reinterpret_cast<char const* const>(p.GetVertexData());
        auto const cp = reinterpret_cast<char const* const>(p.GetColourData());
        auto const ip = reinterpret_cast<char const* const>(p.GetIDData());

        auto ts = vs;

        bool const sep = (vs == 0) || (cs == 0) || (is == 0);

        if (sep) {
            vs = geocalls::SimpleSphericalParticles::VertexDataSize[p.GetVertexDataType()];
            cs = geocalls::SimpleSphericalParticles::ColorDataSize[p.GetColourDataType()];
            is = geocalls::SimpleSphericalParticles::IDDataSize[p.GetIDDataType()];

            ts = vs + cs + is;

            // sep = true;
        }

        auto const cnt = p.GetCount();

        std::vector<size_t> keys(cnt);
        std::iota(keys.begin(), keys.end(), 0);

        auto const& iAcc = p.GetParticleStore().GetIDAcc();

        std::sort(keys.begin(), keys.end(),
            [&iAcc](auto const& a, auto const& b) -> bool { return iAcc->Get_u64(a) < iAcc->Get_u64(b); });

        dlist.resize(cnt * ts);

        auto const basePtr = dlist.data();

        if (sep) {
            for (size_t pidx = 0; pidx < cnt; ++pidx) {
                auto const sidx = keys[pidx];
                memcpy(basePtr + ts * pidx, vp + sidx * avs, vs);
                memcpy(basePtr + ts * pidx + vs, cp + sidx * acs, cs);
                memcpy(basePtr + ts * pidx + vs + cs, ip + sidx * ais, is);

                /*auto const didx = iAcc->Get_u64(pidx);
                memcpy(basePtr + ts * didx, vp, vs);
                memcpy(basePtr + ts * didx + vs, cp, cs);
                memcpy(basePtr + ts * didx + vs + cs, ip, is);*/
            }
        } else {
            for (size_t pidx = 0; pidx < cnt; ++pidx) {
                auto const sidx = keys[pidx];
                memcpy(basePtr + ts * pidx, vp + sidx * ts, ts);

                /*auto const didx = iAcc->Get_u64(pidx);
                memcpy(basePtr + ts * didx, vp, ts);*/
            }
        }

        p.SetVertexData(p.GetVertexDataType(), basePtr, ts);
        p.SetColourData(p.GetColourDataType(), basePtr + vs, ts);
        p.SetIDData(p.GetIDDataType(), basePtr + vs + cs, ts);
    }
    return true;
}
