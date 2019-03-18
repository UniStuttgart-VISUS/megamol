#pragma once

#include <array>
#include <type_traits>

#include "mmstd_datatools/AbstractManipulator.h"

#include "mmcore/moldyn/DirectionalParticleDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/param/StringParam.h"
#include "vislib/StringTokeniser.h"


namespace megamol {
namespace stdplugin {
namespace datatools {

template <class T> class AbstractPDCInterleaver : public AbstractManipulator<T> {
public:
    /** Return module class name */
    static constexpr const char* ClassName(void) {
        return std::is_same<T, core::moldyn::DirectionalParticleDataCall>::value ? "DPDCInterleaver"
                                                                                 : "MPDCInterleaver";
    }

    /** Return module class description */
    static const char* Description(void) { return "Interleaves data streams ([id], [vert], [col], [dir])"; }

    /** Module is always available */
    static bool IsAvailable(void) { return true; }

    /** Ctor */
    AbstractPDCInterleaver(void);

    /** Dtor */
    virtual ~AbstractPDCInterleaver(void);

protected:
    static bool const isDirCall = std::is_same<T, core::moldyn::DirectionalParticleDataCall>::value;

    /**
     * Manipulates the particle data
     *
     * @remarks the default implementation does not changed the data
     *
     * @param outData The call receiving the manipulated data
     * @param inData The call holding the original data
     *
     * @return True on success
     */
    bool manipulateData(T& outData, T& inData) override;

private:
    static unsigned int getDirDataType(typename T::Particles const& part, std::true_type) {
        return part.GetDirDataType();
    }

    static unsigned int getDirDataType(typename T::Particles const& part, std::false_type) { return 0; }

    static void const* getDirData(typename T::Particles& part, std::true_type) { return part.GetDirData(); }

    static void const* getDirData(typename T::Particles& part, std::false_type) { return nullptr; }

    static unsigned int getDirDataStride(typename T::Particles& part, std::true_type) { return part.GetDirDataStride(); }

    static unsigned int getDirDataStride(typename T::Particles& part, std::false_type) { return 0; }

    static void setDirData(
        typename T::Particles& part, unsigned int const type, char const* ptr, size_t const stride, std::true_type) {
        part.SetDirData(static_cast<typename T::Particles::DirDataType>(type), ptr, stride);
    }

    static void setDirData(
        typename T::Particles& part, unsigned int const type, char const* ptr, size_t const stride, std::false_type) {}

    std::vector<std::vector<char>> data_;

}; // end class AbstractParticleBoxFilter


template <class T>
AbstractPDCInterleaver<T>::AbstractPDCInterleaver()
    : AbstractManipulator<T>("dataIn", "dataOut") {
}


template <class T> AbstractPDCInterleaver<T>::~AbstractPDCInterleaver() { this->Release(); }


template <class T> bool AbstractPDCInterleaver<T>::manipulateData(T& outData, T& inData) {
    outData = inData;

    inData.SetUnlocker(nullptr, false); // keep original data locked
                                        // original data will be unlocked through outData

    data_.clear();

    auto const plc = outData.GetParticleListCount();

    data_.resize(plc);

    for (unsigned int plidx = 0; plidx < plc; ++plidx) {
        auto& part = outData.AccessParticles(plidx);

        auto id_ptr = reinterpret_cast<char const*>(part.GetIDData());
        auto vert_ptr = reinterpret_cast<char const*>(part.GetVertexData());
        auto col_ptr = reinterpret_cast<char const*>(part.GetColourData());
        auto dir_ptr = reinterpret_cast<char const*>(getDirData(part, std::integral_constant<bool, isDirCall>()));

        auto const idt = part.GetIDDataType();
        auto const vdt = part.GetVertexDataType();
        auto const cdt = part.GetColourDataType();
        auto const ddt = getDirDataType(part, std::integral_constant<bool, isDirCall>());

        auto const is = core::moldyn::SimpleSphericalParticles::IDDataSize[idt];
        auto const vs = core::moldyn::SimpleSphericalParticles::VertexDataSize[vdt];
        auto const cs = core::moldyn::SimpleSphericalParticles::ColorDataSize[cdt];
        auto const ds = core::moldyn::DirectionalParticles::DirDataSize[ddt];

        auto const istride = part.GetIDDataStride() == 0 ? is : part.GetIDDataStride();
        auto const vstride = part.GetVertexDataStride() == 0 ? vs : part.GetVertexDataStride();
        auto const cstride = part.GetColourDataStride() == 0 ? cs : part.GetColourDataStride();
        auto const dstride = getDirDataStride(part, std::integral_constant<bool, isDirCall>());

        auto const stride = is + vs + cs + ds;

        auto const pcount = part.GetCount();
        data_[plidx].resize(pcount * stride);
        auto cur_data_ptr = data_[plidx].data();

        for (size_t pidx = 0; pidx < pcount; ++pidx) {
            if (id_ptr != nullptr) {
                auto const base_ptr = id_ptr + pidx * istride;
                std::copy(base_ptr, base_ptr + is, cur_data_ptr + pidx * stride);
            }
            if (vert_ptr != nullptr) {
                auto const base_ptr = vert_ptr + pidx * vstride;
                std::copy(base_ptr, base_ptr + vs, cur_data_ptr + pidx * stride + is);
            }
            if (col_ptr != nullptr) {
                auto const base_ptr = col_ptr + pidx * cstride;
                std::copy(base_ptr, base_ptr + cs, cur_data_ptr + pidx * stride + is + vs);
            }
            if (dir_ptr != nullptr) {
                auto const base_ptr = dir_ptr + pidx * dstride;
                std::copy(base_ptr, base_ptr + ds, cur_data_ptr + pidx * stride + is + vs + cs);
            }
        }

        part.SetCount(pcount);
        if (idt != core::moldyn::SimpleSphericalParticles::IDDATA_NONE) {
            part.SetIDData(idt, cur_data_ptr, stride);
        }
        if (vdt != core::moldyn::SimpleSphericalParticles::VERTDATA_NONE) {
            part.SetVertexData(vdt, cur_data_ptr + is, stride);
        }
        if (cdt != core::moldyn::SimpleSphericalParticles::COLDATA_NONE) {
            part.SetColourData(cdt, cur_data_ptr + is + vs, stride);
        }
        if (ddt != core::moldyn::DirectionalParticles::DIRDATA_NONE) {
            setDirData(part, ddt, cur_data_ptr + is + vs + cs, stride, std::integral_constant<bool, isDirCall>());
        }
    }

    return true;
}

} // end namespace datatools
} // end namespace stdplugin
} // end namespace megamol
