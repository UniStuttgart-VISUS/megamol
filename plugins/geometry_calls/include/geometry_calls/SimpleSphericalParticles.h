#pragma once

#include <memory>
#include <type_traits>

#include "Accessor.h"

#include "vislib/Array.h"
#include "vislib/Map.h"
#include "vislib/assert.h"
#include "vislib/math/Cuboid.h"
#include <array>


namespace megamol::geocalls {

/**
 * Class holding all data of a single particle type
 *
 * TODO: This class currently can only hold data for spheres and should
 *       be extended to be able to handle data for arbitrary glyphs.
 *       This also applies to interpolation of data.
 */
class SimpleSphericalParticles {
public:
    /** possible values for the vertex data */
    enum VertexDataType {
        VERTDATA_NONE = 0,      //< indicates that this object is void
        VERTDATA_FLOAT_XYZ = 1, //< use global radius
        VERTDATA_FLOAT_XYZR = 2,
        VERTDATA_SHORT_XYZ = 3, //< quantized positions and global radius
        VERTDATA_DOUBLE_XYZ = 4
        // TODO: what about DOUBLE_XYZR?
    };

    const inline static std::array<std::string, 5> VertexDataTypeNames = {
        "VERTDATA_NONE", "VERTDATA_FLOAT_XYZ", "VERTDATA_FLOAT_XYZR", "VERTDATA_SHORT_XYZ", "VERTDATA_DOUBLE_XYZ"};
    const inline static std::array<uint32_t, 5> VertexDataTypeComponents = {0, 3, 4, 3, 3};

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

    const inline static std::array<std::string, 8> ColourDataTypeNames = {"COLDATA_NONE", "COLDATA_UINT8_RGB",
        "COLDATA_UINT8_RGBA", "COLDATA_FLOAT_RGB", "COLDATA_FLOAT_RGBA", "COLDATA_FLOAT_I", "COLDATA_USHORT_RGBA",
        "COLDATA_DOUBLE_I"};
    const inline static std::array<uint32_t, 8> ColorDataTypeComponents = {0, 3, 4, 3, 4, 1, 4, 1};

    /** possible values for the direction data */
    enum DirDataType { DIRDATA_NONE = 0, DIRDATA_FLOAT_XYZ = 1 };

    const inline static std::array<std::string, 2> DirDataTypeNames = {"DIRDATA_NONE", "DIRDATA_FLOAT_XYZ"};
    const inline static std::array<uint32_t, 2> DirDataTypeComponents = {0, 3};

    /** possible values for the id data */
    enum IDDataType { IDDATA_NONE = 0, IDDATA_UINT32 = 1, IDDATA_UINT64 = 2 };

    const inline static std::array<std::string, 3> IDDataTypeNames = {"IDDATA_NONE", "IDDATA_UINT32", "IDDATA_UINT32"};
    const inline static std::array<uint32_t, 3> IDDataTypeComponents = {0, 1, 1};

    /**
     * This class holds the accessors to the current data.
     */
    class ParticleStore {
    public:
        explicit ParticleStore() = default;

        /*ParticleStore(ParticleStore const& rhs) = delete;

        ParticleStore(ParticleStore&& rhs) = delete;

        ParticleStore& operator=(ParticleStore const& rhs) = delete;

        ParticleStore& operator=(ParticleStore&& rhs) = delete;*/

        virtual ~ParticleStore() = default;

        void SetVertexData(SimpleSphericalParticles::VertexDataType const t, char const* p, unsigned int const s = 0,
            float const globRad = 0.5f) {
            switch (t) {
            case SimpleSphericalParticles::VERTDATA_DOUBLE_XYZ: {
                this->x_acc_ = std::make_shared<Accessor_Impl<double>>(p, s);
                this->y_acc_ = std::make_shared<Accessor_Impl<double>>(p + sizeof(double), s);
                this->z_acc_ = std::make_shared<Accessor_Impl<double>>(p + 2 * sizeof(double), s);
                this->r_acc_ = std::make_shared<Accessor_Val<float, false>>(globRad);
            } break;
            case SimpleSphericalParticles::VERTDATA_FLOAT_XYZ: {
                this->x_acc_ = std::make_shared<Accessor_Impl<float>>(p, s);
                this->y_acc_ = std::make_shared<Accessor_Impl<float>>(p + sizeof(float), s);
                this->z_acc_ = std::make_shared<Accessor_Impl<float>>(p + 2 * sizeof(float), s);
                this->r_acc_ = std::make_shared<Accessor_Val<float, false>>(globRad);
            } break;
            case SimpleSphericalParticles::VERTDATA_FLOAT_XYZR: {
                this->x_acc_ = std::make_shared<Accessor_Impl<float>>(p, s);
                this->y_acc_ = std::make_shared<Accessor_Impl<float>>(p + sizeof(float), s);
                this->z_acc_ = std::make_shared<Accessor_Impl<float>>(p + 2 * sizeof(float), s);
                this->r_acc_ = std::make_shared<Accessor_Impl<float>>(p + 3 * sizeof(float), s);
            } break;
            case SimpleSphericalParticles::VERTDATA_SHORT_XYZ: {
                this->x_acc_ = std::make_shared<Accessor_Impl<unsigned short>>(p, s);
                this->y_acc_ = std::make_shared<Accessor_Impl<unsigned short>>(p + sizeof(unsigned short), s);
                this->z_acc_ = std::make_shared<Accessor_Impl<unsigned short>>(p + 2 * sizeof(unsigned short), s);
                this->r_acc_ = std::make_shared<Accessor_Val<float, false>>(globRad);
            } break;
            case SimpleSphericalParticles::VERTDATA_NONE:
            default: {
                this->x_acc_ = std::make_shared<Accessor_0>();
                this->y_acc_ = std::make_shared<Accessor_0>();
                this->z_acc_ = std::make_shared<Accessor_0>();
                this->r_acc_ = std::make_shared<Accessor_Val<float, false>>(globRad);
            }
            }
        }

        void SetColorData(SimpleSphericalParticles::ColourDataType const t, char const* p, unsigned int const s = 0,
            unsigned char const r = 255, unsigned char const g = 255, unsigned char const b = 255,
            unsigned char const a = 255) {
            switch (t) {
            case SimpleSphericalParticles::COLDATA_DOUBLE_I: {
                this->cr_acc_ = std::make_shared<Accessor_Impl<double>>(p, s);
                this->cg_acc_ = std::make_shared<Accessor_0>();
                this->cb_acc_ = std::make_shared<Accessor_0>();
                this->ca_acc_ = std::make_shared<Accessor_0>();
            } break;
            case SimpleSphericalParticles::COLDATA_FLOAT_I: {
                this->cr_acc_ = std::make_shared<Accessor_Impl<float>>(p, s);
                this->cg_acc_ = std::make_shared<Accessor_0>();
                this->cb_acc_ = std::make_shared<Accessor_0>();
                this->ca_acc_ = std::make_shared<Accessor_0>();
            } break;
            case SimpleSphericalParticles::COLDATA_FLOAT_RGB: {
                this->cr_acc_ = std::make_shared<Accessor_Impl<float>>(p, s);
                this->cg_acc_ = std::make_shared<Accessor_Impl<float>>(p + sizeof(float), s);
                this->cb_acc_ = std::make_shared<Accessor_Impl<float>>(p + 2 * sizeof(float), s);
                this->ca_acc_ = std::make_shared<Accessor_Val<float, false>>(1.0f);
            } break;
            case SimpleSphericalParticles::COLDATA_FLOAT_RGBA: {
                this->cr_acc_ = std::make_shared<Accessor_Impl<float>>(p, s);
                this->cg_acc_ = std::make_shared<Accessor_Impl<float>>(p + sizeof(float), s);
                this->cb_acc_ = std::make_shared<Accessor_Impl<float>>(p + 2 * sizeof(float), s);
                this->ca_acc_ = std::make_shared<Accessor_Impl<float>>(p + 3 * sizeof(float), s);
            } break;
            case SimpleSphericalParticles::COLDATA_UINT8_RGB: {
                this->cr_acc_ = std::make_shared<Accessor_Impl<unsigned char>>(p, s);
                this->cg_acc_ = std::make_shared<Accessor_Impl<unsigned char>>(p + sizeof(unsigned char), s);
                this->cb_acc_ = std::make_shared<Accessor_Impl<unsigned char>>(p + 2 * sizeof(unsigned char), s);
                this->ca_acc_ = std::make_shared<Accessor_Val<unsigned char, false>>(255);
            } break;
            case SimpleSphericalParticles::COLDATA_UINT8_RGBA: {
                this->cr_acc_ = std::make_shared<Accessor_Impl<unsigned char>>(p, s);
                this->cg_acc_ = std::make_shared<Accessor_Impl<unsigned char>>(p + sizeof(unsigned char), s);
                this->cb_acc_ = std::make_shared<Accessor_Impl<unsigned char>>(p + 2 * sizeof(unsigned char), s);
                this->ca_acc_ = std::make_shared<Accessor_Impl<unsigned char>>(p + 3 * sizeof(unsigned char), s);
            } break;
            case SimpleSphericalParticles::COLDATA_USHORT_RGBA: {
                this->cr_acc_ = std::make_shared<Accessor_Impl<unsigned short>>(p, s);
                this->cg_acc_ = std::make_shared<Accessor_Impl<unsigned short>>(p + sizeof(unsigned short), s);
                this->cb_acc_ = std::make_shared<Accessor_Impl<unsigned short>>(p + 2 * sizeof(unsigned short), s);
                this->ca_acc_ = std::make_shared<Accessor_Impl<unsigned short>>(p + 3 * sizeof(unsigned short), s);
            } break;
            case SimpleSphericalParticles::COLDATA_NONE:
            default: {
                this->cr_acc_ = std::make_shared<Accessor_Val<unsigned char, true>>(r);
                this->cg_acc_ = std::make_shared<Accessor_Val<unsigned char, true>>(g);
                this->cb_acc_ = std::make_shared<Accessor_Val<unsigned char, true>>(b);
                this->ca_acc_ = std::make_shared<Accessor_Val<unsigned char, true>>(a);
            }
            }
        }

        void SetDirData(SimpleSphericalParticles::DirDataType const t, char const* p, unsigned int const s = 0) {
            switch (t) {
            case DIRDATA_FLOAT_XYZ: {
                this->dx_acc_ = std::make_shared<Accessor_Impl<float>>(p, s);
                this->dy_acc_ = std::make_shared<Accessor_Impl<float>>(p + sizeof(float), s);
                this->dz_acc_ = std::make_shared<Accessor_Impl<float>>(p + 2 * sizeof(float), s);
            } break;
            default: {
                this->dx_acc_ = std::make_shared<Accessor_0>();
                this->dy_acc_ = std::make_shared<Accessor_0>();
                this->dz_acc_ = std::make_shared<Accessor_0>();
            }
            }
        }

        void SetIDData(SimpleSphericalParticles::IDDataType const t, char const* p, unsigned int const s = 0) {
            switch (t) {
            case SimpleSphericalParticles::IDDATA_UINT32: {
                this->id_acc_ = std::make_shared<Accessor_Impl<unsigned int>>(reinterpret_cast<char const*>(p), s);
            } break;
            case SimpleSphericalParticles::IDDATA_UINT64: {
                this->id_acc_ = std::make_shared<Accessor_Impl<uint64_t>>(reinterpret_cast<char const*>(p), s);
            } break;
            case SimpleSphericalParticles::IDDATA_NONE:
            default: {
                this->id_acc_ = std ::make_shared<Accessor_Idx>();
            }
            }
        }

        std::shared_ptr<Accessor> const& GetXAcc() const {
            return this->x_acc_;
        }

        std::shared_ptr<Accessor> const& GetYAcc() const {
            return this->y_acc_;
        }

        std::shared_ptr<Accessor> const& GetZAcc() const {
            return this->z_acc_;
        }

        std::shared_ptr<Accessor> const& GetRAcc() const {
            return this->r_acc_;
        }

        std::shared_ptr<Accessor> const& GetCRAcc() const {
            return this->cr_acc_;
        }

        std::shared_ptr<Accessor> const& GetCGAcc() const {
            return this->cg_acc_;
        }

        std::shared_ptr<Accessor> const& GetCBAcc() const {
            return this->cb_acc_;
        }

        std::shared_ptr<Accessor> const& GetCAAcc() const {
            return this->ca_acc_;
        }

        std::shared_ptr<Accessor> const& GetDXAcc() const {
            return this->dx_acc_;
        }

        std::shared_ptr<Accessor> const& GetDYAcc() const {
            return this->dy_acc_;
        }

        std::shared_ptr<Accessor> const& GetDZAcc() const {
            return this->dz_acc_;
        }

        std::shared_ptr<Accessor> const& GetIDAcc() const {
            return this->id_acc_;
        }

    private:
        std::shared_ptr<Accessor> x_acc_ = std::make_shared<Accessor_0>();
        std::shared_ptr<Accessor> y_acc_ = std::make_shared<Accessor_0>();
        std::shared_ptr<Accessor> z_acc_ = std::make_shared<Accessor_0>();
        std::shared_ptr<Accessor> r_acc_ = std::make_shared<Accessor_0>();
        std::shared_ptr<Accessor> cr_acc_ = std::make_shared<Accessor_0>();
        std::shared_ptr<Accessor> cg_acc_ = std::make_shared<Accessor_0>();
        std::shared_ptr<Accessor> cb_acc_ = std::make_shared<Accessor_0>();
        std::shared_ptr<Accessor> ca_acc_ = std::make_shared<Accessor_0>();
        std::shared_ptr<Accessor> dx_acc_ = std::make_shared<Accessor_0>();
        std::shared_ptr<Accessor> dy_acc_ = std::make_shared<Accessor_0>();
        std::shared_ptr<Accessor> dz_acc_ = std::make_shared<Accessor_0>();
        std::shared_ptr<Accessor> id_acc_ = std::make_shared<Accessor_0>();
    };

    /** possible values of accumulated data sizes over all vertex coordinates */
    static unsigned int VertexDataSize[5];

    /** possible values of accumulated data sizes over all color elements */
    static unsigned int ColorDataSize[8];

    /** possible values of data sizes over all directional dimensions */
    static unsigned int DirDataSize[2];

    /** possible values of data sizes of the id */
    static unsigned int IDDataSize[3];

    /**
     * Ctor
     */
    SimpleSphericalParticles();

    /**
     * Copy ctor
     *
     * @param src The object to clone from
     */
    SimpleSphericalParticles(const SimpleSphericalParticles& src);

    /**
     * Dtor
     */
    ~SimpleSphericalParticles();

    /**
     * Answer the colour data type
     *
     * @return The colour data type
     */
    inline ColourDataType GetColourDataType() const {
        return this->colDataType;
    }

    /**
     * Answer the colour data pointer
     *
     * @return The colour data pointer
     */
    inline const void* GetColourData() const {
        return this->colPtr;
    }

    /**
     * Answer the colour data stride.
     * It represents the distance to the succeeding colour.
     *
     * @return The colour data stride in byte.
     */
    inline unsigned int GetColourDataStride() const {
        return this->colStride == ColorDataSize[this->colDataType] ? 0 : this->colStride;
    }

    /**
     * Answer the direction data type
     *
     * @return The direction data type
     */
    inline DirDataType GetDirDataType() const {
        return this->dirDataType;
    }

    /**
     * Answer the direction data pointer
     *
     * @return The direction data pointer
     */
    inline const void* GetDirData() const {
        return this->dirPtr;
    }

    /**
     * Answer the direction data stride
     *
     * @return The direction data stride
     */
    inline unsigned int GetDirDataStride() const {
        return this->dirStride == DirDataSize[this->dirDataType] ? 0 : this->dirStride;
    }

    /**
     * Answer the number of stored objects
     *
     * @return The number of stored objects
     */
    inline UINT64 GetCount() const {
        return this->count;
    }

    /**
     * Answer the global colour
     *
     * @return The global colour as a pointer to four unsigned bytes
     *         storing the RGBA colour components
     */
    inline const unsigned char* GetGlobalColour() const {
        return this->col;
    }

    /**
     * Answer the global radius
     *
     * @return The global radius
     */
    inline float GetGlobalRadius() const {
        return this->radius;
    }

    /**
     * Answer the global particle type
     *
     * @return the global type
     */
    inline unsigned int GetGlobalType() const {
        return this->particleType;
    }

    /**
     * Answer the maximum colour index value to be mapped
     *
     * @return The maximum colour index value to be mapped
     */
    inline float GetMaxColourIndexValue() const {
        return this->maxColI;
    }

    /**
     * Answer the minimum colour index value to be mapped
     *
     * @return The minimum colour index value to be mapped
     */
    inline float GetMinColourIndexValue() const {
        return this->minColI;
    }

    /**
     * Answer the vertex data type
     *
     * @return The vertex data type
     */
    inline VertexDataType GetVertexDataType() const {
        return this->vertDataType;
    }

    /**
     * Answer the vertex data pointer
     *
     * @return The vertex data pointer
     */
    inline const void* GetVertexData() const {
        return this->vertPtr;
    }

    /**
     * Answer the vertex data stride.
     * It represents the distance to the succeeding vertex.
     *
     * @return The vertex data stride in byte.
     */
    inline unsigned int GetVertexDataStride() const {
        return this->vertStride == VertexDataSize[this->vertDataType] ? 0 : this->vertStride;
    }

    /**
     * Answer the id data type
     *
     * @return The id data type
     */
    inline IDDataType GetIDDataType() const {
        return this->idDataType;
    }

    /**
     * Answer the id data pointer
     *
     * @return The id data pointer
     */
    inline const void* GetIDData() const {
        return this->idPtr;
    }

    /**
     * Answer the id data stride.
     * It represents the distance to the succeeding id.
     *
     * @return The id data stride in byte.
     */
    inline unsigned int GetIDDataStride() const {
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

        this->par_store_->SetColorData(t, reinterpret_cast<char const*>(p), this->colStride, this->col[0], this->col[1],
            this->col[2], this->col[3]);
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
     * Sets the direction data
     *
     * @param t The type of the direction data
     * @param p The pointer to the direction data (must not be NULL if t
     *          is not 'DIRDATA_NONE'
     * @param s The stride of the direction data
     */
    void SetDirData(DirDataType t, const void* p, unsigned int s = 0) {
        ASSERT((p != NULL) || (t == DIRDATA_NONE));
        this->dirDataType = t;
        this->dirPtr = p;
        this->dirStride = s == 0 ? DirDataSize[t] : s;

        this->par_store_->SetDirData(t, reinterpret_cast<char const*>(p), this->dirStride);
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
        this->dirDataType = DIRDATA_NONE;
        this->dirPtr = nullptr; // DO NOT DELETE
        this->idDataType = IDDATA_NONE;
        this->idPtr = nullptr; // DO NOT DELETE

        this->par_store_->SetVertexData(VERTDATA_NONE, nullptr);
        this->par_store_->SetColorData(COLDATA_NONE, nullptr);
        this->par_store_->SetDirData(DIRDATA_NONE, nullptr);
        this->par_store_->SetIDData(IDDATA_NONE, nullptr);

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

        this->par_store_->SetColorData(this->colDataType, reinterpret_cast<char const*>(this->colPtr), this->colStride,
            this->col[0], this->col[1], this->col[2], this->col[3]);
    }

    /**
     * Sets the global radius
     *
     * @param r The global radius
     */
    void SetGlobalRadius(float r) {
        this->radius = r;
        this->par_store_->SetVertexData(
            this->vertDataType, reinterpret_cast<char const*>(this->vertPtr), this->vertStride, this->radius);
    }

    /**
     * Sets the global particle type
     *
     * @param t The global type
     */
    void SetGlobalType(unsigned int t) {
        this->particleType = t;
    }

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

        this->par_store_->SetVertexData(t, reinterpret_cast<char const*>(p), this->vertStride, this->radius);
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

        this->par_store_->SetIDData(t, reinterpret_cast<char const*>(p), this->idStride);
    }

    /**
     * Reports existence of IDs.
     *
     * @return true, if the particles have IDs.
     */
    inline bool HasID() const {
        return this->idDataType != IDDATA_NONE;
    }

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
     * Get instance of particle store call the accessors.
     *
     * @return Instance of particle store.
     */
    ParticleStore const& GetParticleStore() const {
        return *this->par_store_;
    }

    /**
     * Disable NULL-checks in case we have an OpenGL-VAO
     * @param disable flag to disable/enable the checks
     */
    void disableNullChecksForVAOs(bool disable = true) {
        disabledNullChecks = disable;
    }

    /**
     * Defines wether we transport VAOs instead of real data
     * @param vao flag to disable/enable the checks
     */
    void SetIsVAO(bool vao) {
        this->isVAO = vao;
    }

    /**
     * Disable NULL-checks in case we have an OpenGL-VAO
     * @param disable flag to disable/enable the checks
     */
    bool IsVAO() {
        return this->isVAO;
    }

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
    void SetClusterInfos(ClusterInfos* infos) {
        this->clusterInfos = infos;
    }

    /**
     * gets the local ClusterInfos-struct
     */
    ClusterInfos* GetClusterInfos() {
        return this->clusterInfos;
    }

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
    vislib::math::Cuboid<float> GetBBox() const {
        return this->wsBBox;
    }

    void SetBBox(vislib::math::Cuboid<float> const& bbox) {
        this->wsBBox = bbox;
    }

private:
    /** The global colour */
    unsigned char col[4];

    /** The colour data type */
    ColourDataType colDataType;

    /** The colour data pointer */
    const void* colPtr;

    /** The colour data stride */
    unsigned int colStride;

    /** The direction data type */
    DirDataType dirDataType;

    /** The direction data pointer */
    const void* dirPtr;

    /** The direction data stride */
    unsigned int dirStride;

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

protected:
    /** Instance of the particle store */
    std::shared_ptr<ParticleStore> par_store_ = std::make_shared<ParticleStore>();
};

} // namespace megamol::geocalls
