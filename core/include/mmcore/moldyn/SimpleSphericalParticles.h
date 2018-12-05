#pragma once

#include <memory>
#include <type_traits>

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/moldyn/Accessor.h"

#include "vislib/Array.h"
#include "vislib/Map.h"
#include "vislib/assert.h"
#include "vislib/math/Cuboid.h"


namespace megamol {
namespace core {
namespace moldyn {

/**
 * Class holding all data of a single particle type
 *
 * TODO: This class currently can only hold data for spheres and should
 *       be extended to be able to handle data for arbitrary glyphs.
 *       This also applies to interpolation of data.
 */
class MEGAMOLCORE_API SimpleSphericalParticles {
public:
    /** possible values for the vertex data */
    enum VertexDataType {
        VERTDATA_NONE = 0,      //< indicates that this object is void
        VERTDATA_FLOAT_XYZ = 1, //< use global radius
        VERTDATA_FLOAT_XYZR = 2,
        VERTDATA_SHORT_XYZ = 3, //< quantized positions and global radius
        VERTDATA_DOUBLE_XYZ = 4
    };

    /** possible values for the colour data */
    enum ColourDataType {
        COLDATA_NONE = 0, //< use global colour
        COLDATA_UINT8_RGB = 1,
        COLDATA_UINT8_RGBA = 2,
        COLDATA_FLOAT_RGB = 3,
        COLDATA_FLOAT_RGBA = 4,
        COLDATA_FLOAT_I = 5, //< single float value to be mapped by a transfer function
        COLDATA_USHORT_RGBA = 6,
        COLDATA_DOUBLE_I = 7
    };

    /** possible values for the id data */
    enum IDDataType { IDDATA_NONE = 0, IDDATA_UINT32 = 1, IDDATA_UINT64 = 2 };

    /**
     * This class holds the accessors to the current data.
     */
    class ParticleStore {
    public:
        explicit ParticleStore() = default;

        void SetVertexData(SimpleSphericalParticles::VertexDataType const t, const void* p, unsigned int const s = 0) {
            switch (t) {
            case SimpleSphericalParticles::VERTDATA_DOUBLE_XYZ: {
                this->x_acc_.reset(new Accessor_Impl<double>(reinterpret_cast<const char*>(p), s));
                this->y_acc_.reset(new Accessor_Impl<double>(reinterpret_cast<const char*>(p) + sizeof(double), s));
                this->z_acc_.reset(new Accessor_Impl<double>(reinterpret_cast<const char*>(p) + 2 * sizeof(double), s));
                this->r_acc_.reset(new Accessor_0());
            } break;
            case SimpleSphericalParticles::VERTDATA_FLOAT_XYZ: {
                this->x_acc_.reset(new Accessor_Impl<float>(reinterpret_cast<const char*>(p), s));
                this->y_acc_.reset(new Accessor_Impl<float>(reinterpret_cast<const char*>(p) + sizeof(float), s));
                this->z_acc_.reset(new Accessor_Impl<float>(reinterpret_cast<const char*>(p) + 2 * sizeof(float), s));
                this->r_acc_.reset(new Accessor_0());
            } break;
            case SimpleSphericalParticles::VERTDATA_FLOAT_XYZR: {
                this->x_acc_.reset(new Accessor_Impl<float>(reinterpret_cast<const char*>(p), s));
                this->y_acc_.reset(new Accessor_Impl<float>(reinterpret_cast<const char*>(p) + sizeof(float), s));
                this->z_acc_.reset(new Accessor_Impl<float>(reinterpret_cast<const char*>(p) + 2 * sizeof(float), s));
                this->r_acc_.reset(new Accessor_Impl<float>(reinterpret_cast<const char*>(p) + 3 * sizeof(float), s));
            } break;
            case SimpleSphericalParticles::VERTDATA_SHORT_XYZ: {
                this->x_acc_.reset(new Accessor_Impl<unsigned short>(reinterpret_cast<const char*>(p), s));
                this->y_acc_.reset(new Accessor_Impl<unsigned short>(reinterpret_cast<const char*>(p) + sizeof(unsigned short), s));
                this->z_acc_.reset(new Accessor_Impl<unsigned short>(reinterpret_cast<const char*>(p) + 2 * sizeof(unsigned short), s));
                this->r_acc_.reset(new Accessor_0());
            } break;
            case SimpleSphericalParticles::VERTDATA_NONE:
            default: {
                this->x_acc_.reset(new Accessor_0());
                this->y_acc_.reset(new Accessor_0());
                this->z_acc_.reset(new Accessor_0());
                this->r_acc_.reset(new Accessor_0());
            }
            }
        }

        void SetColorData(SimpleSphericalParticles::ColourDataType const t, void const* p, unsigned int const s = 0) {
            switch (t) {
            case SimpleSphericalParticles::COLDATA_DOUBLE_I: {
                this->cr_acc_.reset(new Accessor_Impl<double>(reinterpret_cast<char const*>(p), s));
                this->cg_acc_.reset(new Accessor_0());
                this->cb_acc_.reset(new Accessor_0());
                this->ca_acc_.reset(new Accessor_0());
            } break;
            case SimpleSphericalParticles::COLDATA_FLOAT_I: {
                this->cr_acc_.reset(new Accessor_Impl<float>(reinterpret_cast<char const*>(p), s));
                this->cg_acc_.reset(new Accessor_0());
                this->cb_acc_.reset(new Accessor_0());
                this->ca_acc_.reset(new Accessor_0());
            } break;
            case SimpleSphericalParticles::COLDATA_FLOAT_RGB: {
                this->cr_acc_.reset(new Accessor_Impl<float>(reinterpret_cast<char const*>(p), s));
                this->cg_acc_.reset(new Accessor_Impl<float>(reinterpret_cast<char const*>(p) + sizeof(float), s));
                this->cb_acc_.reset(new Accessor_Impl<float>(reinterpret_cast<char const*>(p) + 2 * sizeof(float), s));
                this->ca_acc_.reset(new Accessor_0());
            } break;
            case SimpleSphericalParticles::COLDATA_FLOAT_RGBA: {
                this->cr_acc_.reset(new Accessor_Impl<float>(reinterpret_cast<char const*>(p), s));
                this->cg_acc_.reset(new Accessor_Impl<float>(reinterpret_cast<char const*>(p) + sizeof(float), s));
                this->cb_acc_.reset(new Accessor_Impl<float>(reinterpret_cast<char const*>(p) + 2 * sizeof(float), s));
                this->ca_acc_.reset(new Accessor_Impl<float>(reinterpret_cast<char const*>(p) + 3 * sizeof(float), s));
            } break;
            case SimpleSphericalParticles::COLDATA_UINT8_RGB: {
                this->cr_acc_.reset(new Accessor_Impl<unsigned char>(reinterpret_cast<char const*>(p), s));
                this->cg_acc_.reset(new Accessor_Impl<unsigned char>(reinterpret_cast<char const*>(p) + sizeof(unsigned char), s));
                this->cb_acc_.reset(new Accessor_Impl<unsigned char>(reinterpret_cast<char const*>(p) + 2 * sizeof(unsigned char), s));
                this->ca_acc_.reset(new Accessor_0());
            } break;
            case SimpleSphericalParticles::COLDATA_UINT8_RGBA: {
                this->cr_acc_.reset(new Accessor_Impl<unsigned char>(reinterpret_cast<char const*>(p), s));
                this->cg_acc_.reset(new Accessor_Impl<unsigned char>(reinterpret_cast<char const*>(p) + sizeof(unsigned char), s));
                this->cb_acc_.reset(new Accessor_Impl<unsigned char>(reinterpret_cast<char const*>(p) + 2 * sizeof(unsigned char), s));
                this->ca_acc_.reset(new Accessor_Impl<unsigned char>(reinterpret_cast<char const*>(p) + 3 * sizeof(unsigned char), s));
            } break;
            case SimpleSphericalParticles::COLDATA_USHORT_RGBA: {
                this->cr_acc_.reset(new Accessor_Impl<unsigned short>(reinterpret_cast<char const*>(p), s));
                this->cg_acc_.reset(new Accessor_Impl<unsigned short>(reinterpret_cast<char const*>(p) + sizeof(unsigned short), s));
                this->cb_acc_.reset(new Accessor_Impl<unsigned short>(reinterpret_cast<char const*>(p) + 2 * sizeof(unsigned short), s));
                this->ca_acc_.reset(new Accessor_Impl<unsigned short>(reinterpret_cast<char const*>(p) + 3 * sizeof(unsigned short), s));
            } break;
            case SimpleSphericalParticles::COLDATA_NONE:
            default: {
                this->cr_acc_.reset(new Accessor_0());
                this->cg_acc_.reset(new Accessor_0());
                this->cb_acc_.reset(new Accessor_0());
                this->ca_acc_.reset(new Accessor_0());
            }
            }
        }

        void SetIDData(SimpleSphericalParticles::IDDataType const t, void const* p, unsigned int const s = 0) {
            switch (t) {
            case SimpleSphericalParticles::IDDATA_UINT32: {
                this->id_acc_.reset(new Accessor_Impl<unsigned int>(reinterpret_cast<char const*>(p), s));
            } break;
            case SimpleSphericalParticles::IDDATA_UINT64: {
                this->id_acc_.reset(new Accessor_Impl<uint64_t>(reinterpret_cast<char const*>(p), s));
            } break;
            case SimpleSphericalParticles::IDDATA_NONE:
            default: { this->id_acc_.reset(new Accessor_0()); }
            }
        }

        std::shared_ptr<Accessor> const& GetXAcc() const { return this->x_acc_; }

        std::shared_ptr<Accessor> const& GetYAcc() const { return this->y_acc_; }

        std::shared_ptr<Accessor> const& GetZAcc() const { return this->z_acc_; }

        std::shared_ptr<Accessor> const& GetRAcc() const { return this->r_acc_; }

        std::shared_ptr<Accessor> const& GetCRAcc() const { return this->cr_acc_; }

        std::shared_ptr<Accessor> const& GetCGAcc() const { return this->cg_acc_; }

        std::shared_ptr<Accessor> const& GetCBAcc() const { return this->cb_acc_; }

        std::shared_ptr<Accessor> const& GetCAAcc() const { return this->ca_acc_; }

        std::shared_ptr<Accessor> const& GetIDAcc() const { return this->id_acc_; }

    private:
        std::shared_ptr<Accessor> x_acc_;
        std::shared_ptr<Accessor> y_acc_;
        std::shared_ptr<Accessor> z_acc_;
        std::shared_ptr<Accessor> r_acc_;
        std::shared_ptr<Accessor> cr_acc_;
        std::shared_ptr<Accessor> cg_acc_;
        std::shared_ptr<Accessor> cb_acc_;
        std::shared_ptr<Accessor> ca_acc_;
        std::shared_ptr<Accessor> id_acc_;
    };


    class VertexData_Detail {
    public:
        virtual double GetXd() const = 0;
        virtual double GetYd() const = 0;
        virtual double GetZd() const = 0;
        virtual float const GetXf() const = 0;
        virtual float const GetYf() const = 0;
        virtual float const GetZf() const = 0;
        virtual float const GetRf() const = 0;
        virtual short const GetXs() const = 0;
        virtual short const GetYs() const = 0;
        virtual short const GetZs() const = 0;
        virtual void SetBasePtr(void const* ptr) = 0;
        virtual std::unique_ptr<VertexData_Detail> Clone() const = 0;
        virtual ~VertexData_Detail() = default;
    };

    class VertexData_None : public VertexData_Detail {
    public:
        VertexData_None() = default;

        VertexData_None(VertexData_None const& rhs) = default;

        double GetXd() const override { return 0.0; }

        double GetYd() const override { return 0.0; }

        double GetZd() const override { return 0.0; }

        virtual float const GetXf() const override { return 0.0f; }

        virtual float const GetYf() const override { return 0.0f; }

        virtual float const GetZf() const override { return 0.0f; }

        virtual float const GetRf() const override { return 0.0f; }

        virtual short const GetXs() const override { return 0; }

        virtual short const GetYs() const override { return 0; }

        virtual short const GetZs() const override { return 0; }

        virtual void SetBasePtr(void const* ptr) override{};

        virtual std::unique_ptr<VertexData_Detail> Clone() const override {
            return std::unique_ptr<VertexData_Detail>{new VertexData_None{*this}};
        }
    };

    template <class T, bool hasRad> class VertexData_Impl : public VertexData_Detail {
    public:
        VertexData_Impl() = default;

        VertexData_Impl(VertexData_Impl const& rhs) : basePtr{rhs.basePtr} {}

        double GetXd() const override { return GetX<double>(); }

        virtual float const GetXf() const override { return GetX<float>(); }

        virtual short const GetXs() const override { return GetX<short>(); }

        template <class R> std::enable_if_t<std::is_same<T, R>::value, R> const GetX() const {
            return this->basePtr[0];
        }

        template <class R> std::enable_if_t<!std::is_same<T, R>::value, R> const GetX() const {
            return static_cast<R>(this->basePtr[0]);
        }

        double GetYd() const override { return GetY<double>(); }

        virtual float const GetYf() const override { return GetY<float>(); }

        virtual short const GetYs() const override { return GetY<short>(); }

        template <class R> std::enable_if_t<std::is_same<T, R>::value, R> const GetY() const {
            return this->basePtr[1];
        }

        template <class R> std::enable_if_t<!std::is_same<T, R>::value, R> const GetY() const {
            return static_cast<R>(this->basePtr[1]);
        }

        double GetZd() const override { return GetZ<double>(); }

        virtual float const GetZf() const override { return GetZ<float>(); }

        virtual short const GetZs() const override { return GetZ<short>(); }

        template <class R> std::enable_if_t<std::is_same<T, R>::value, R> const GetZ() const {
            return this->basePtr[2];
        }

        template <class R> std::enable_if_t<!std::is_same<T, R>::value, R> const GetZ() const {
            return static_cast<R>(this->basePtr[2]);
        }

        virtual float const GetRf() const override { return GetR<hasRad>(); }

        template <bool hasRad_v> std::enable_if_t<hasRad_v, float> const GetR() const { return this->basePtr[3]; }

        template <bool hasRad_v> std::enable_if_t<!hasRad_v, float> const GetR() const { return 0.0f; }

        virtual void SetBasePtr(void const* ptr) override { this->basePtr = reinterpret_cast<T const*>(ptr); }

        virtual std::unique_ptr<VertexData_Detail> Clone() const override {
            return std::unique_ptr<VertexData_Detail>{new VertexData_Impl{*this}};
        }

    private:
        T const* basePtr;
    };

    class VertexData_Base {
    public:
        VertexData_Base(std::unique_ptr<VertexData_Detail>&& impl, void const* basePtr)
            : pimpl{std::forward<std::unique_ptr<VertexData_Detail>>(impl)} {
            pimpl->SetBasePtr(basePtr);
        }

        VertexData_Base(VertexData_Base const& rhs) = delete;

        VertexData_Base(VertexData_Base&& rhs) : pimpl{std::forward<std::unique_ptr<VertexData_Detail>>(rhs.pimpl)} {}

        VertexData_Base& operator=(VertexData_Base const& rhs) = delete;

        VertexData_Base& operator=(VertexData_Base&& rhs) {
            pimpl = std::move(rhs.pimpl);
            return *this;
        }

        double GetXd() const { return pimpl->GetXd(); }

        double GetYd() const { return pimpl->GetYd(); }

        double GetZd() const { return pimpl->GetZd(); }

        float const GetXf() const { return pimpl->GetXf(); }

        float const GetYf() const { return pimpl->GetYf(); }

        float const GetZf() const { return pimpl->GetZf(); }

        float const GetRf() const { return pimpl->GetRf(); }

        short const GetXs() const { return pimpl->GetXs(); }

        short const GetYs() const { return pimpl->GetYs(); }

        short const GetZs() const { return pimpl->GetZs(); }

    private:
        std::unique_ptr<VertexData_Detail> pimpl;
    };

    class ColorData_Detail {
    public:
        virtual uint16_t GetRu16() const = 0;
        virtual uint16_t GetGu16() const = 0;
        virtual uint16_t GetBu16() const = 0;
        virtual uint16_t GetAu16() const = 0;
        virtual uint8_t const GetRu8() const = 0;
        virtual uint8_t const GetGu8() const = 0;
        virtual uint8_t const GetBu8() const = 0;
        virtual uint8_t const GetAu8() const = 0;
        virtual float const GetRf() const = 0;
        virtual float const GetGf() const = 0;
        virtual float const GetBf() const = 0;
        virtual float const GetAf() const = 0;
        virtual float const GetIf() const = 0;
        virtual double GetId() const = 0;
        virtual void SetBasePtr(void const* ptr) = 0;
        virtual std::unique_ptr<ColorData_Detail> Clone() const = 0;
        virtual ~ColorData_Detail() = default;
    };

    class ColorData_None : public ColorData_Detail {
    public:
        ColorData_None() = default;

        ColorData_None(ColorData_None const& rhs) = default;

        uint16_t GetRu16() const override { return 0; }

        uint16_t GetGu16() const override { return 0; }

        uint16_t GetBu16() const override { return 0; }

        uint16_t GetAu16() const override { return 0; }

        virtual uint8_t const GetRu8() const override { return 0; }

        virtual uint8_t const GetGu8() const override { return 0; }

        virtual uint8_t const GetBu8() const override { return 0; }

        virtual uint8_t const GetAu8() const override { return 0; }

        virtual float const GetRf() const override { return 0.0f; }

        virtual float const GetGf() const override { return 0.0f; }

        virtual float const GetBf() const override { return 0.0f; }

        virtual float const GetAf() const override { return 0.0f; }

        virtual float const GetIf() const override { return 0.0f; }

        double GetId() const override { return 0.0; }

        virtual void SetBasePtr(void const* ptr) override {}

        virtual std::unique_ptr<ColorData_Detail> Clone() const override {
            return std::unique_ptr<ColorData_Detail>{new ColorData_None{*this}};
        }
    };

    template <class T, bool hasAlpha, bool isI> class ColorData_Impl : public ColorData_Detail {
    public:
        ColorData_Impl() = default;

        ColorData_Impl(ColorData_Impl const& rhs) : basePtr{rhs.basePtr} {}

        uint16_t GetRu16() const override { return GetR<uint16_t, isI>(); }

        virtual uint8_t const GetRu8() const override { return GetR<uint8_t, isI>(); }

        virtual float const GetRf() const override { return GetR<float, isI>(); }

        template <class R, bool isI_v> std::enable_if_t<std::is_same<T, R>::value && !isI_v, R> const GetR() const {
            return this->basePtr[0];
        }

        template <class R, bool isI_v> std::enable_if_t<!std::is_same<T, R>::value && !isI_v, R> const GetR() const {
            return static_cast<R>(this->basePtr[0]);
        }

        template <class R, bool isI_v> std::enable_if_t<isI_v, R> const GetR() const { return static_cast<R>(0.0); }

        uint16_t GetGu16() const override { return GetG<uint16_t, isI>(); }

        virtual uint8_t const GetGu8() const override { return GetG<uint8_t, isI>(); }

        virtual float const GetGf() const override { return GetG<float, isI>(); }

        template <class R, bool isI_v> std::enable_if_t<std::is_same<T, R>::value && !isI_v, R> const GetG() const {
            return this->basePtr[1];
        }

        template <class R, bool isI_v> std::enable_if_t<!std::is_same<T, R>::value && !isI_v, R> const GetG() const {
            return static_cast<R>(this->basePtr[1]);
        }

        template <class R, bool isI_v> std::enable_if_t<isI_v, R> const GetG() const { return static_cast<R>(0.0); }

        uint16_t GetBu16() const override { return GetB<uint16_t, isI>(); }

        virtual uint8_t const GetBu8() const override { return GetB<uint8_t, isI>(); }

        virtual float const GetBf() const override { return GetB<float, isI>(); }

        template <class R, bool isI_v> std::enable_if_t<std::is_same<T, R>::value && !isI_v, R> const GetB() const {
            return this->basePtr[2];
        }

        template <class R, bool isI_v> std::enable_if_t<!std::is_same<T, R>::value && !isI_v, R> const GetB() const {
            return static_cast<R>(this->basePtr[2]);
        }

        template <class R, bool isI_v> std::enable_if_t<isI_v, R> const GetB() const { return static_cast<R>(0.0); }

        uint16_t GetAu16() const override { return GetA<uint16_t, hasAlpha, isI>(); }

        virtual uint8_t const GetAu8() const override { return GetA<uint8_t, hasAlpha, isI>(); }

        virtual float const GetAf() const override { return GetA<float, hasAlpha, isI>(); }

        template <class R, bool hasAlpha_v, bool isI_v>
        std::enable_if_t<std::is_same<T, R>::value && hasAlpha_v && !isI_v, R> const GetA() const {
            return this->basePtr[3];
        }

        template <class R, bool hasAlpha_v, bool isI_v>
        std::enable_if_t<!std::is_same<T, R>::value && hasAlpha_v && !isI_v, R> const GetA() const {
            return static_cast<R>(this->basePtr[3]);
        }

        template <class R, bool hasAlpha_v, bool isI_v> std::enable_if_t<!hasAlpha_v && !isI_v, R> const GetA() const {
            return static_cast<R>(0.0);
        }

        template <class R, bool hasAlpha_v, bool isI_v> std::enable_if_t<isI_v, R> const GetA() const {
            return static_cast<R>(0.0);
        }

        virtual float const GetIf() const override { return GetI<float, isI>(); }

        double GetId() const override { return GetI<double, isI>(); }

        template <class R, bool isI_v> std::enable_if_t<std::is_same<T, R>::value && isI_v, R> const GetI() const {
            return this->basePtr[0];
        }

        template <class R, bool isI_v> std::enable_if_t<!std::is_same<T, R>::value && isI_v, R> const GetI() const {
            return static_cast<R>(this->basePtr[0]);
        }

        template <class R, bool isI_v> std::enable_if_t<!isI_v, R> const GetI() const { return static_cast<R>(0.0); }

        virtual void SetBasePtr(void const* ptr) override { this->basePtr = reinterpret_cast<T const*>(ptr); }

        virtual std::unique_ptr<ColorData_Detail> Clone() const override {
            return std::unique_ptr<ColorData_Detail>{new ColorData_Impl{*this}};
        }

    private:
        T const* basePtr;
    };

    class ColorData_Base {
    public:
        ColorData_Base(std::unique_ptr<ColorData_Detail>&& impl, void const* basePtr)
            : pimpl{std::forward<std::unique_ptr<ColorData_Detail>>(impl)} {
            pimpl->SetBasePtr(basePtr);
        }

        ColorData_Base(ColorData_Base const& rhs) = delete;

        ColorData_Base(ColorData_Base&& rhs) : pimpl{std::forward<std::unique_ptr<ColorData_Detail>>(rhs.pimpl)} {}

        ColorData_Base& operator=(ColorData_Base const& rhs) = delete;

        ColorData_Base& operator=(ColorData_Base&& rhs) {
            pimpl = std::move(rhs.pimpl);
            return *this;
        }

        uint16_t GetRu16() const { return pimpl->GetRu16(); }
        uint16_t GetGu16() const { return pimpl->GetGu16(); }
        uint16_t GetBu16() const { return pimpl->GetBu16(); }
        uint16_t GetAu16() const { return pimpl->GetAu16(); }
        uint8_t const GetRu8() const { return pimpl->GetRu8(); }
        uint8_t const GetGu8() const { return pimpl->GetGu8(); }
        uint8_t const GetBu8() const { return pimpl->GetBu8(); }
        uint8_t const GetAu8() const { return pimpl->GetAu8(); }
        float const GetRf() const { return pimpl->GetRf(); }
        float const GetGf() const { return pimpl->GetGf(); }
        float const GetBf() const { return pimpl->GetBf(); }
        float const GetAf() const { return pimpl->GetAf(); }
        float const GetIf() const { return pimpl->GetIf(); }
        double GetId() const { return pimpl->GetId(); }

    private:
        std::unique_ptr<ColorData_Detail> pimpl;
    };

    class IDData_Detail {
    public:
        virtual uint32_t const GetIDu32() const = 0;
        virtual uint64_t const GetIDu64() const = 0;
        virtual void SetBasePtr(void const* ptr) = 0;
        virtual std::unique_ptr<IDData_Detail> Clone() const = 0;
        virtual ~IDData_Detail() = default;
    };

    class IDData_None : public IDData_Detail {
    public:
        IDData_None() = default;

        IDData_None(IDData_None const& rhs) = default;

        virtual uint32_t const GetIDu32() const override { return 0; }

        virtual uint64_t const GetIDu64() const override { return 0; }

        virtual void SetBasePtr(void const* ptr) override {}

        virtual std::unique_ptr<IDData_Detail> Clone() const override {
            return std::unique_ptr<IDData_Detail>{new IDData_None{*this}};
        }
    };

    template <class T> class IDData_Impl : public IDData_Detail {
    public:
        IDData_Impl() = default;

        IDData_Impl(IDData_Impl const& rhs) : basePtr{rhs.basePtr} {}

        virtual uint32_t const GetIDu32() const override { return GetID<uint32_t>(); }

        virtual uint64_t const GetIDu64() const override { return GetID<uint64_t>(); }

        template <class R> std::enable_if_t<std::is_same<T, R>::value, R> const GetID() const {
            return this->basePtr[0];
        }

        template <class R> std::enable_if_t<!std::is_same<T, R>::value, R> const GetID() const {
            return static_cast<R>(this->basePtr[0]);
        }

        virtual void SetBasePtr(void const* ptr) override { this->basePtr = reinterpret_cast<T const*>(ptr); }

        virtual std::unique_ptr<IDData_Detail> Clone() const override {
            return std::unique_ptr<IDData_Detail>{new IDData_Impl{*this}};
        }

    private:
        T const* basePtr;
    };

    class IDData_Base {
    public:
        IDData_Base(std::unique_ptr<IDData_Detail>&& impl, void const* basePtr)
            : pimpl{std::forward<std::unique_ptr<IDData_Detail>>(impl)} {
            pimpl->SetBasePtr(basePtr);
        }

        IDData_Base(IDData_Base const& rhs) = delete;

        IDData_Base(IDData_Base&& rhs) : pimpl{std::forward<std::unique_ptr<IDData_Detail>>(rhs.pimpl)} {}

        IDData_Base& operator=(IDData_Base const& rhs) = delete;

        IDData_Base& operator=(IDData_Base&& rhs) {
            pimpl = std::move(rhs.pimpl);
            return *this;
        }

        uint32_t const GetIDu32() const { return pimpl->GetIDu32(); }
        uint64_t const GetIDu64() const { return pimpl->GetIDu64(); }

    private:
        std::unique_ptr<IDData_Detail> pimpl;
    };

    /** Struct holding pointers into data streams for a specific particle */
    struct particle_t {
        particle_t(VertexData_Base&& v, ColorData_Base&& c, IDData_Base&& i)
            : vert{std::forward<VertexData_Base>(v)}
            , col{std::forward<ColorData_Base>(c)}
            , id{std::forward<IDData_Base>(i)} {}

        particle_t(particle_t const& rhs) = delete;

        particle_t(particle_t&& rhs) : vert{std::move(rhs.vert)}, col{std::move(rhs.col)}, id{std::move(rhs.id)} {}

        particle_t& operator=(particle_t const& rhs) = delete;

        particle_t& operator=(particle_t&& rhs) {
            vert = std::move(rhs.vert);
            col = std::move(rhs.col);
            id = std::move(rhs.id);
            return *this;
        }

        VertexData_Base vert;
        ColorData_Base col;
        IDData_Base id;
        /*void const* vertPtr;
        void const* colPtr;
        void const* idPtr;*/
    };

    /** possible values of accumulated data sizes over all vertex coordinates */
    static unsigned int VertexDataSize[5];

    /** possible values of accumulated data sizes over all color elements */
    static unsigned int ColorDataSize[8];

    /** possible values of data sizes of the id */
    static unsigned int IDDataSize[3];

    /**
     * Ctor
     */
    SimpleSphericalParticles(void);

    /**
     * Copy ctor
     *
     * @param src The object to clone from
     */
    SimpleSphericalParticles(const SimpleSphericalParticles& src);

    /**
     * Dtor
     */
    ~SimpleSphericalParticles(void);

    /**
     * Answer the colour data type
     *
     * @return The colour data type
     */
    inline ColourDataType GetColourDataType(void) const { return this->colDataType; }

    /**
     * Answer the colour data pointer
     *
     * @return The colour data pointer
     */
    inline const void* GetColourData(void) const { return this->colPtr; }

    /**
     * Answer the colour data stride.
     * It represents the distance to the succeeding colour.
     *
     * @return The colour data stride in byte.
     */
    inline unsigned int GetColourDataStride(void) const {
        return this->colStride == ColorDataSize[this->colDataType] ? 0 : this->colStride;
    }

    /**
     * Answer the number of stored objects
     *
     * @return The number of stored objects
     */
    inline UINT64 GetCount(void) const { return this->count; }

    /**
     * Answer the global colour
     *
     * @return The global colour as a pointer to four unsigned bytes
     *         storing the RGBA colour components
     */
    inline const unsigned char* GetGlobalColour(void) const { return this->col; }

    /**
     * Answer the global radius
     *
     * @return The global radius
     */
    inline float GetGlobalRadius(void) const { return this->radius; }

    /**
     * Answer the global particle type
     *
     * @return the global type
     */
    inline unsigned int GetGlobalType(void) const { return this->particleType; }

    /**
     * Answer the maximum colour index value to be mapped
     *
     * @return The maximum colour index value to be mapped
     */
    inline float GetMaxColourIndexValue(void) const { return this->maxColI; }

    /**
     * Answer the minimum colour index value to be mapped
     *
     * @return The minimum colour index value to be mapped
     */
    inline float GetMinColourIndexValue(void) const { return this->minColI; }

    /**
     * Answer the vertex data type
     *
     * @return The vertex data type
     */
    inline VertexDataType GetVertexDataType(void) const { return this->vertDataType; }

    /**
     * Answer the vertex data pointer
     *
     * @return The vertex data pointer
     */
    inline const void* GetVertexData(void) const { return this->vertPtr; }

    /**
     * Answer the vertex data stride.
     * It represents the distance to the succeeding vertex.
     *
     * @return The vertex data stride in byte.
     */
    inline unsigned int GetVertexDataStride(void) const {
        return this->vertStride == VertexDataSize[this->vertDataType] ? 0 : this->vertStride;
    }

    /**
     * Answer the id data type
     *
     * @return The id data type
     */
    inline IDDataType GetIDDataType(void) const { return this->idDataType; }

    /**
     * Answer the id data pointer
     *
     * @return The id data pointer
     */
    inline const void* GetIDData(void) const { return this->idPtr; }

    /**
     * Answer the id data stride.
     * It represents the distance to the succeeding id.
     *
     * @return The id data stride in byte.
     */
    inline unsigned int GetIDDataStride(void) const {
        return this->idStride == IDDataSize[this->idDataType] ? 0 : this->idStride;
    }

    /**
     * Sets the colour data
     *
     * @param t The type of the colour data
     * @param p The pointer to the colour data (must not be NULL if t
     *          is not 'COLDATA_NONE'
     * @param s The stride of the colour data
     */
    void SetColourData(ColourDataType t, const void* p, unsigned int s = 0) {
        //    ASSERT((p != NULL) || (t == COLDATA_NONE));
        this->colDataType = t;
        this->colPtr = p;
        this->colStride = s == 0 ? ColorDataSize[t] : s;

        switch (t) {
        case COLDATA_UINT8_RGB:
            this->colorAccessor.reset(new ColorData_Impl<uint8_t, false, false>{});
            break;
        case COLDATA_UINT8_RGBA:
            this->colorAccessor.reset(new ColorData_Impl<uint8_t, true, false>{});
            break;
        case COLDATA_FLOAT_RGB:
            this->colorAccessor.reset(new ColorData_Impl<float, false, false>{});
            break;
        case COLDATA_FLOAT_RGBA:
            this->colorAccessor.reset(new ColorData_Impl<float, true, false>{});
            break;
        case COLDATA_FLOAT_I:
            this->colorAccessor.reset(new ColorData_Impl<float, false, true>{});
            break;
        case COLDATA_USHORT_RGBA:
            this->colorAccessor.reset(new ColorData_Impl<uint16_t, true, false>{});
            break;
        case COLDATA_DOUBLE_I:
            this->colorAccessor.reset(new ColorData_Impl<double, false, true>{});
            break;
        case COLDATA_NONE:
        default:
            this->colorAccessor.reset(new ColorData_None{});
        }

        this->par_store_.SetColorData(t, p, this->colStride);
    }

    /**
     * Sets the colour map index values
     *
     * @param minVal The minimum colour index value to be mapped
     * @param maxVal The maximum colour index value to be mapped
     */
    void SetColourMapIndexValues(float minVal, float maxVal) {
        this->maxColI = maxVal;
        this->minColI = minVal;
    }

    /**
     * Sets the number of objects stored and resets all data pointers!
     *
     * @param cnt The number of stored objects
     */
    void SetCount(UINT64 cnt) {
        this->colDataType = COLDATA_NONE;
        this->colPtr = nullptr; // DO NOT DELETE
        this->vertDataType = VERTDATA_NONE;
        this->vertPtr = nullptr; // DO NOT DELETE
        this->idDataType = IDDATA_NONE;
        this->idPtr = nullptr; // DO NOT DELETE

        this->count = cnt;
    }

    /**
     * Sets the global colour data
     *
     * @param r The red colour component
     * @param g The green colour component
     * @param b The blue colour component
     * @param a The opacity alpha
     */
    void SetGlobalColour(unsigned int r, unsigned int g, unsigned int b, unsigned int a = 255) {
        this->col[0] = r;
        this->col[1] = g;
        this->col[2] = b;
        this->col[3] = a;
    }

    /**
     * Sets the global radius
     *
     * @param r The global radius
     */
    void SetGlobalRadius(float r) { this->radius = r; }

    /**
     * Sets the global particle type
     *
     * @param t The global type
     */
    void SetGlobalType(unsigned int t) { this->particleType = t; }

    /**
     * Sets the vertex data
     *
     * @param t The type of the vertex data
     * @param p The pointer to the vertex data (must not be NULL if t
     *          is not 'VERTDATA_NONE'
     * @param s The stride of the vertex data
     */
    void SetVertexData(VertexDataType t, const void* p, unsigned int s = 0) {
        ASSERT(this->disabledNullChecks || (p != NULL) || (t == VERTDATA_NONE));
        this->vertDataType = t;
        this->vertPtr = p;
        this->vertStride = s == 0 ? VertexDataSize[t] : s;

        switch (t) {
        case VERTDATA_FLOAT_XYZ:
            this->vertexAccessor.reset(new VertexData_Impl<float, false>{});
            break;
        case VERTDATA_FLOAT_XYZR:
            this->vertexAccessor.reset(new VertexData_Impl<float, true>{});
            break;
        case VERTDATA_SHORT_XYZ:
            this->vertexAccessor.reset(new VertexData_Impl<short, false>{});
            break;
        case VERTDATA_DOUBLE_XYZ:
            this->vertexAccessor.reset(new VertexData_Impl<double, false>{});
            break;
        case VERTDATA_NONE:
        default:
            this->vertexAccessor.reset(new VertexData_None{});
        }

        this->par_store_.SetVertexData(t, p, this->vertStride);
    }

    /**
     * Sets the ID data
     *
     * @param t The type of the ID data
     * @param p The pointer to the ID data (must not be NULL if t
     *          is not 'IDDATA_NONE'
     * @param s The stride of the ID data
     */
    void SetIDData(IDDataType t, const void* p, unsigned int s = 0) {
        ASSERT(this->disabledNullChecks || (p != NULL) || (t == IDDATA_NONE));
        this->idDataType = t;
        this->idPtr = p;
        this->idStride = s == 0 ? IDDataSize[t] : s;

        switch (t) {
        case IDDATA_UINT32:
            this->idAccessor.reset(new IDData_Impl<uint32_t>{});
            break;
        case IDDATA_UINT64:
            this->idAccessor.reset(new IDData_Impl<uint64_t>{});
            break;
        case IDDATA_NONE:
        default:
            this->idAccessor.reset(new IDData_None{});
        }

        this->par_store_.SetIDData(t, p, this->idStride);
    }

    /**
     * Reports existance of IDs.
     *
     * @return true, if the particles have IDs.
     */
    inline bool HasID() const { return this->idDataType != IDDATA_NONE; }

    /**
     * Assignment operator
     *
     * @param rhs The right hand side operand
     *
     * @return A reference to 'this'
     */
    SimpleSphericalParticles& operator=(const SimpleSphericalParticles& rhs);

    /**
     * Test for equality
     *
     * @param rhs The right hand side operand
     *
     * @return 'true' if 'this' and 'rhs' are equal.
     */
    bool operator==(const SimpleSphericalParticles& rhs) const;

    /**
     * Access particle at index without range check.
     *
     * @param idx Index of particle in the streams.
     *
     * @return Struct of pointers to positions of the particle in the streams.
     */
    inline particle_t operator[](size_t idx) const noexcept {
        return particle_t{
            VertexData_Base{this->vertexAccessor->Clone(),
                this->vertPtr != nullptr ? static_cast<char const*>(this->vertPtr) + idx * this->vertStride : nullptr},
            ColorData_Base{this->colorAccessor->Clone(),
                this->colPtr != nullptr ? static_cast<char const*>(this->colPtr) + idx * this->colStride : nullptr},
            IDData_Base{this->idAccessor->Clone(),
                this->idPtr != nullptr ? static_cast<char const*>(this->idPtr) + idx * this->idStride : nullptr}};
    }

    /**
     * Access particle at index with range check.
     *
     * @param idx Index of particle in the streams.
     *
     * @return Struct of pointers to positions of the particle in the streams.
     *
     * @throws std::out_of_range if idx is larger than particle count.
     */
    inline particle_t const& At(size_t idx) const {
        if (idx < this->count) {
            return this->operator[](idx);
        } else {
            throw std::out_of_range("Idx larger than particle count.");
        }
    }

    /**
     * Get instance of particle store call the accessors.
     * 
     * @return Instance of particle store.
     */
    ParticleStore const& GetParticleStore() const { return this->par_store_; }

    /**
     * Disable NULL-checks in case we have an OpenGL-VAO
     * @param disable flag to disable/enable the checks
     */
    void disableNullChecksForVAOs(bool disable = true) { disabledNullChecks = disable; }

    /**
     * Defines wether we transport VAOs instead of real data
     * @param vao flag to disable/enable the checks
     */
    void SetIsVAO(bool vao) { this->isVAO = vao; }

    /**
     * Disable NULL-checks in case we have an OpenGL-VAO
     * @param disable flag to disable/enable the checks
     */
    bool IsVAO() { return this->isVAO; }

    /**
     * If we handle clusters this could be useful
     */
    struct ClusterInfos {
        /** a map with clusterid to particleids relation*/
        vislib::Map<int, vislib::Array<int>> data;
        /** the map in plain data for upload to gpu */
        unsigned int* plainData;
        /** size of the plain data*/
        size_t sizeofPlainData;
        /** number of clusters*/
        unsigned int numClusters;
        ClusterInfos() : data(), plainData(0), sizeofPlainData(0), numClusters(0){};
    };

    /**
     * Sets the local ClusterInfos-struct
     */
    void SetClusterInfos(ClusterInfos* infos) { this->clusterInfos = infos; }

    /**
     * gets the local ClusterInfos-struct
     */
    ClusterInfos* GetClusterInfos() { return this->clusterInfos; }

    /**
     * Sets the VertexArrayObject, VertexBuffer and ColorBuffer used
     */
    void SetVAOs(unsigned int vao, unsigned int vb, unsigned int cb) {
        this->glVAO = vao;
        this->glVB = vb;
        this->glCB = cb;
    }

    /**
     * Gets the VertexArrayObject, VertexBuffer and ColorBuffer used
     */
    void GetVAOs(unsigned int& vao, unsigned int& vb, unsigned int& cb) {
        vao = this->glVAO;
        vb = this->glVB;
        cb = this->glCB;
    }

    /** Gets the world-space minmax bounding box of the list data */
    vislib::math::Cuboid<float> GetBBox() const { return this->wsBBox; }

    void SetBBox(vislib::math::Cuboid<float> const& bbox) { this->wsBBox = bbox; }

private:
    /** The global colour */
    unsigned char col[4];

    /** The colour data type */
    ColourDataType colDataType;

    /** The colour data pointer */
    const void* colPtr;

    /** The colour data stride */
    unsigned int colStride;

    /** The number of objects stored */
    UINT64 count;

    /** The maximum colour index value to be mapped */
    float maxColI;

    /** The minimum colour index value to be mapped */
    float minColI;

    /** The global radius */
    float radius;

    /** The global type of particles in the list */
    unsigned int particleType;

    /** the world-space minmax bounding box of the list data */
    vislib::math::Cuboid<float> wsBBox;

    /** The vertex data type */
    VertexDataType vertDataType;

    /** The vertex data pointer */
    const void* vertPtr;

    /** The vertex data stride */
    unsigned int vertStride;

    /** disable NULL-checks if used with OpenGL-VAO */
    bool disabledNullChecks;

    /** do we use a VertexArrayObject? */
    bool isVAO;

    /** Vertex Array Object to transport */
    unsigned int glVAO;
    /** Vertex Buffer to transport */
    unsigned int glVB;
    /** Color Buffer to transport */
    unsigned int glCB;

    /** local Cluster Infos*/
    ClusterInfos* clusterInfos;

    /** The particle ID type */
    IDDataType idDataType;

    /** The particle ID pointer */
    void const* idPtr;

    /** The particle ID stride */
    unsigned int idStride;

    /** Polymorphic vertex access object */
    std::unique_ptr<VertexData_Detail> vertexAccessor;

    /** Polymorphic color access object */
    std::unique_ptr<ColorData_Detail> colorAccessor;

    /** Polymorphic id access object */
    std::unique_ptr<IDData_Detail> idAccessor;

    /** Instance of the particle store */
    ParticleStore par_store_;
};

} // namespace moldyn
} // namespace core
} // namespace megamol
