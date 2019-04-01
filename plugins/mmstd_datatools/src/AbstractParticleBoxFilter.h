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

template <class T> class AbstractParticleBoxFilter : public AbstractManipulator<T> {
public:
    /** Return module class name */
    static constexpr const char* ClassName(void) {
        return std::is_same<T, core::moldyn::DirectionalParticleDataCall>::value ? "DirParticleBoxFilter"
                                                                                 : "ParticleBoxFilter";
    }

    /** Return module class description */
    static const char* Description(void) { return "Applies a box filter on a set of particles"; }

    /** Module is always available */
    static bool IsAvailable(void) { return true; }

    /** Ctor */
    AbstractParticleBoxFilter(void);

    /** Dtor */
    virtual ~AbstractParticleBoxFilter(void);

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
    static vislib::math::Cuboid<float> getBoxFromString(vislib::TString const& str) {
        if (!str.Contains(',')) {
            return vislib::math::Cuboid<float>(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        }

        auto tokens = vislib::TStringTokeniser::Split(str, ',', true);

        if (tokens.Count() < 6) {
            return vislib::math::Cuboid<float>(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        }

        std::array<float, 6> vals;
        for (int i = 0; i < 6; ++i) {
            vals[i] = vislib::TCharTraits::ParseDouble(tokens[i]);
        }

        return vislib::math::Cuboid<float>(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]);
    }

    static unsigned int getDirDataType(typename T::Particles const& part, std::true_type) {
        return part.GetDirDataType();
    }

    static unsigned int getDirDataType(typename T::Particles const& part, std::false_type) { return 0; }

    static void setDirData(
        typename T::Particles& part, unsigned int const type, char const* ptr, size_t const stride, std::true_type) {
        part.SetDirData(static_cast<typename T::Particles::DirDataType>(type), ptr, stride);
    }

    static void setDirData(
        typename T::Particles& part, unsigned int const type, char const* ptr, size_t const stride, std::false_type) {}

    core::param::ParamSlot boxSlot_;

    std::vector<std::vector<char>> data_;

    std::vector<size_t> numPts_;

    std::vector<size_t> stride_;
}; // end class AbstractParticleBoxFilter


template <class T>
AbstractParticleBoxFilter<T>::AbstractParticleBoxFilter()
    : AbstractManipulator<T>("dataIn", "dataOut")
    , boxSlot_("box", "Box definition for the Box Filter (minx, miny, minz, maxx, maxy, maxz)") {
    boxSlot_ << new core::param::StringParam("0.0, 0.0, 0.0, 1.0, 1.0, 1.0");
    this->MakeSlotAvailable(&boxSlot_);
}


template <class T> AbstractParticleBoxFilter<T>::~AbstractParticleBoxFilter() { this->Release(); }


template <class T> bool AbstractParticleBoxFilter<T>::manipulateData(T& outData, T& inData) {
    outData = inData;

    inData.SetUnlocker(nullptr, false); // keep original data locked
                                        // original data will be unlocked through outData

    auto const box = getBoxFromString(boxSlot_.Param<core::param::StringParam>()->Value());

    data_.clear();
    numPts_.clear();
    stride_.clear();

    // TODO This assumes the data to be interleaved

    auto const plc = outData.GetParticleListCount();

    data_.resize(plc);
    numPts_.resize(plc, 0);
    stride_.resize(plc, 0);

    for (unsigned int plidx = 0; plidx < plc; ++plidx) {
        auto& part = outData.AccessParticles(plidx);

        auto id_ptr = reinterpret_cast<char const*>(part.GetIDData());
        auto vert_ptr = reinterpret_cast<char const*>(part.GetVertexData());
        auto col_ptr = reinterpret_cast<char const*>(part.GetColourData());

        auto const idt = part.GetIDDataType();
        auto const vdt = part.GetVertexDataType();
        auto const cdt = part.GetColourDataType();
        auto const ddt = getDirDataType(part, std::integral_constant<bool, isDirCall>());

        if (vdt == core::moldyn::SimpleSphericalParticles::VERTDATA_NONE) {
            vislib::sys::Log::DefaultLog.WriteError(
                "ParticleBoxFilter: Box filter cannot be applied on list entry %d containing no vertex information.\n",
                plidx);
            continue;
        }

        if (idt == core::moldyn::SimpleSphericalParticles::IDDATA_NONE) id_ptr = vert_ptr;
        if (cdt == core::moldyn::SimpleSphericalParticles::COLDATA_NONE) col_ptr = vert_ptr;

        auto const base_ptr = std::min(id_ptr, std::min(vert_ptr, col_ptr));
        auto const max_ptr = std::max(id_ptr, std::max(vert_ptr, col_ptr));

        auto const is = core::moldyn::SimpleSphericalParticles::IDDataSize[idt];
        auto const vs = core::moldyn::SimpleSphericalParticles::VertexDataSize[vdt];
        auto const cs = core::moldyn::SimpleSphericalParticles::ColorDataSize[cdt];
        auto const ds = core::moldyn::DirectionalParticles::DirDataSize[ddt];

        auto const& stride = 0 == part.GetVertexDataStride() ? vs : part.GetVertexDataStride();

        if ((max_ptr - base_ptr) > stride) {
            vislib::sys::Log::DefaultLog.WriteWarn(
                "ParticleBoxFilter: Data of list entry %d is not interleaved. Skipping entry.\n", plidx);
            continue;
        }

        auto const pcount = part.GetCount();
        data_[plidx].resize(pcount * stride);
        auto cur_data_ptr = data_[plidx].data();
        auto& cur_numPts = numPts_[plidx];
        auto& cur_stride = stride_[plidx];

        auto const& store = part.GetParticleStore();
        auto const& xacc = store.GetXAcc();
        auto const& yacc = store.GetYAcc();
        auto const& zacc = store.GetZAcc();

        for (size_t pidx = 0; pidx < pcount; ++pidx) {
            // check for each particle whether it is contained within the box
            vislib::math::Point<float, 3> pt(xacc->Get_f(pidx), yacc->Get_f(pidx), zacc->Get_f(pidx));
            if (box.Contains(pt, true)) {
                std::copy(base_ptr + pidx * stride, base_ptr + (pidx + 1) * stride, cur_data_ptr + cur_numPts * stride);
                ++cur_numPts;
            }
        }

        data_[plidx].resize(cur_numPts * stride);
        cur_stride = stride;

        part.SetCount(cur_numPts);
        if (id_ptr < vert_ptr) {
            if (idt != core::moldyn::SimpleSphericalParticles::IDDATA_NONE) {
                part.SetIDData(idt, cur_data_ptr, stride);
            }
            part.SetVertexData(vdt, cur_data_ptr + is, stride);
            if (cdt != core::moldyn::SimpleSphericalParticles::COLDATA_NONE) {
                part.SetColourData(cdt, cur_data_ptr + is + vs, stride);
            }
            if (ddt != core::moldyn::DirectionalParticles::DIRDATA_NONE) {
                setDirData(part, ddt, cur_data_ptr + is + vs + cs, stride, std::integral_constant<bool, isDirCall>());
            }
        } else {
            part.SetVertexData(vdt, cur_data_ptr, stride);
            if (cdt != core::moldyn::SimpleSphericalParticles::COLDATA_NONE) {
                part.SetColourData(cdt, cur_data_ptr + vs, stride);
            }
            if (ddt != core::moldyn::DirectionalParticles::DIRDATA_NONE) {
                setDirData(part, ddt, cur_data_ptr + vs + cs, stride, std::integral_constant<bool, isDirCall>());
            }
            if (idt != core::moldyn::SimpleSphericalParticles::IDDATA_NONE) {
                part.SetIDData(idt, cur_data_ptr + vs + cs + ds, stride);
            }
        }
    }

    return true;
}

} // end namespace datatools
} // end namespace stdplugin
} // end namespace megamol
