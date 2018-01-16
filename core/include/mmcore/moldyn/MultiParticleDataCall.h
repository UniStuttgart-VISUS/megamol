/*
 * MultiParticleDataCall.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MULTIPARTICLEDATACALL_H_INCLUDED
#define MEGAMOLCORE_MULTIPARTICLEDATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/moldyn/AbstractParticleDataCall.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/assert.h"
#include "vislib/Map.h"
#include "vislib/Array.h"

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
        class VertexData_Detail {
        public:
            virtual float const GetXf() const = 0;
            virtual float const GetYf() const = 0;
            virtual float const GetZf() const = 0;
            virtual float const GetRf() const = 0;
            virtual short const GetXs() const = 0;
            virtual short const GetYs() const = 0;
            virtual short const GetZs() const = 0;
            virtual void SetBasePtr(void const* ptr) = 0;
            virtual VertexData_Detail& Clone() const = 0;
            virtual ~VertexData_Detail() = default;
        };

        template<class T, bool hasRad>
        class VertexData_Impl : public VertexData_Detail {
        public:
            VertexData_Impl() = default;

            VertexData_Impl(VertexData_Impl const& rhs)
                : basePtr{rhs.basePtr} { }

            virtual float const GetXf() const override {
                return GetX<float>();
            }

            virtual short const GetXs() const override {
                return GetX<short>();
            }

            template<class R>
            std::enable_if_t<std::is_same<T, R>::value, R> const GetX() const {
                return this->basePtr[0];
            }

            template<class R>
            std::enable_if_t<!std::is_same<T, R>::value, R> const GetX() const {
                return static_cast<R>(this->basePtr[0]);
            }

            virtual float const GetYf() const override {
                return GetY<float>();
            }

            virtual short const GetYs() const override {
                return GetY<short>();
            }

            template<class R>
            std::enable_if_t<std::is_same<T, R>::value, R> const GetY() const {
                return this->basePtr[1];
            }

            template<class R>
            std::enable_if_t<!std::is_same<T, R>::value, R> const GetY() const {
                return static_cast<R>(this->basePtr[1]);
            }

            virtual float const GetZf() const override {
                return GetZ<float>();
            }

            virtual short const GetZs() const override {
                return GetZ<short>();
            }

            template<class R>
            std::enable_if_t<std::is_same<T, R>::value, R> const GetZ() const {
                return this->basePtr[2];
            }

            template<class R>
            std::enable_if_t<!std::is_same<T, R>::value, R> const GetZ() const {
                return static_cast<R>(this->basePtr[2]);
            }

            virtual float const GetRf() const override {
                return GetR<hasRad>();
            }

            template<bool hasRad_v>
            std::enable_if_t<hasRad_v, float> const GetR() const {
                return this->basePtr[3];
            }

            template<bool hasRad_v>
            std::enable_if_t<!hasRad_v, float> const GetR() const {
                return 0.0f;
            }

            virtual void SetBasePtr(void const* ptr) override {
                this->basePtr = reinterpret_cast<T const*>(ptr);
            }

            virtual VertexData_Impl& Clone() const override {
                return VertexData_Impl{*this};
            }
        private:
            T const* basePtr;
        };

        class VertexData_Base {
        public:
            VertexData_Base(VertexData_Detail& impl, void const* basePtr)
                : pimpl{impl} {
                pimpl.SetBasePtr(basePtr);
            }

            float const GetXf() const {
                return pimpl.GetXf();
            }

            float const GetYf() const {
                return pimpl.GetYf();
            }

            float const GetZf() const {
                return pimpl.GetZf();
            }

            float const GetRf() const {
                return pimpl.GetRf();
            }

            short const GetXs() const {
                return pimpl.GetXs();
            }

            short const GetYs() const {
                return pimpl.GetYs();
            }

            short const GetZs() const {
                return pimpl.GetZs();
            }
        private:
            VertexData_Detail& pimpl;
        };

        class ColorData_Detail {
        public:
            virtual uint8_t const GetRu8() const = 0;
            virtual uint8_t const GetGu8() const = 0;
            virtual uint8_t const GetBu8() const = 0;
            virtual uint8_t const GetAu8() const = 0;
            virtual float const GetRf() const = 0;
            virtual float const GetGf() const = 0;
            virtual float const GetBf() const = 0;
            virtual float const GetAf() const = 0;
            virtual float const GetIf() const = 0;
            virtual void SetBasePtr(void const* ptr) = 0;
            virtual ColorData_Detail& Clone() const = 0;
            virtual ~ColorData_Detail() = default;
        };

        template<class T, bool hasAlpha, bool isI>
        class ColorData_Impl : public ColorData_Detail {
        public:
            ColorData_Impl() = default;

            ColorData_Impl(ColorData_Impl const& rhs)
                : basePtr{rhs.basePtr} { }

            virtual uint8_t const GetRu8() const override {
                return GetR<uint8_t, isI>();
            }

            virtual float const GetRf() const override {
                return GetR<float, isI>();
            }

            template<class R, bool isI_v>
            std::enable_if_t<std::is_same<T, R>::value && !isI_v, R> const GetR() const {
                return this->basePtr[0];
            }

            template<class R, bool isI_v>
            std::enable_if_t<!std::is_same<T, R>::value && !isI_v, R> const GetR() const {
                return static_cast<R>(this->basePtr[0]);
            }

            template<class R, bool isI_v>
            std::enable_if_t<isI_v, R> const GetR() const {
                return static_cast<R>(0.0);
            }

            virtual uint8_t const GetGu8() const override {
                return GetG<uint8_t, isI>();
            }

            virtual float const GetGf() const override {
                return GetG<float, isI>();
            }

            template<class R, bool isI_v>
            std::enable_if_t<std::is_same<T, R>::value && !isI_v, R> const GetG() const {
                return this->basePtr[1];
            }

            template<class R, bool isI_v>
            std::enable_if_t<!std::is_same<T, R>::value && !isI_v, R> const GetG() const {
                return static_cast<R>(this->basePtr[1]);
            }

            template<class R, bool isI_v>
            std::enable_if_t<isI_v, R> const GetG() const {
                return static_cast<R>(0.0);
            }

            virtual uint8_t const GetBu8() const override {
                return GetB<uint8_t, isI>();
            }

            virtual float const GetBf() const override {
                return GetB<float, isI>();
            }

            template<class R, bool isI_v>
            std::enable_if_t<std::is_same<T, R>::value && !isI_v, R> const GetB() const {
                return this->basePtr[2];
            }

            template<class R, bool isI_v>
            std::enable_if_t<!std::is_same<T, R>::value && !isI_v, R> const GetB() const {
                return static_cast<R>(this->basePtr[2]);
            }

            template<class R, bool isI_v>
            std::enable_if_t<isI_v, R> const GetB() const {
                return static_cast<R>(0.0);
            }

            virtual uint8_t const GetAu8() const override {
                return GetA<uint8_t, hasAlpha, isI>();
            }

            virtual float const GetAf() const override {
                return GetA<float, hasAlpha, isI>();
            }

            template<class R, bool hasAlpha_v, bool isI_v>
            std::enable_if_t<std::is_same<T, R>::value && hasAlpha_v && !isI_v, R> const GetA() const {
                return this->basePtr[3];
            }

            template<class R, bool hasAlpha_v, bool isI_v>
            std::enable_if_t<!std::is_same<T, R>::value && hasAlpha_v && !isI_v, R> const GetA() const {
                return static_cast<R>(this->basePtr[3]);
            }

            template<class R, bool hasAlpha_v, bool isI_v>
            std::enable_if_t<!hasAlpha_v && !isI_v, R> const GetA() const {
                return static_cast<R>(0.0);
            }

            template<class R, bool hasAlpha_v, bool isI_v>
            std::enable_if_t<isI_v, R> const GetA() const {
                return static_cast<R>(0.0);
            }

            virtual float const GetIf() const override {
                return GetI<float, isI>();
            }

            template<class R, bool isI_v>
            std::enable_if_t<std::is_same<T, R>::value && isI_v, R> const GetI() const {
                return this->basePtr[0];
            }

            template<class R, bool isI_v>
            std::enable_if_t<!std::is_same<T, R>::value && isI_v, R> const GetI() const {
                return static_cast<R>(this->basePtr[0]);
            }

            template<class R, bool isI_v>
            std::enable_if_t<!isI_v, R> const GetI() const {
                return static_cast<R>(0.0);
            }

            virtual void SetBasePtr(void const* ptr) override {
                this->basePtr = reinterpret_cast<T const*>(ptr);
            }

            virtual ColorData_Impl& Clone() const override {
                return ColorData_Impl{*this};
            }
        private:
            T const* basePtr;
        };

        class ColorData_Base {
        public:
            ColorData_Base(ColorData_Detail& impl, void const* basePtr)
                : pimpl{impl} {
                pimpl.SetBasePtr(basePtr);
            }

            uint8_t const GetRu8() const {
                return pimpl.GetRu8();
            }
            uint8_t const GetGu8() const {
                return pimpl.GetGu8();
            }
            uint8_t const GetBu8() const {
                return pimpl.GetBu8();
            }
            uint8_t const GetAu8() const {
                return pimpl.GetAu8();
            }
            float const GetRf() const {
                return pimpl.GetRf();
            }
            float const GetGf() const {
                return pimpl.GetGf();
            }
            float const GetBf() const {
                return pimpl.GetBf();
            }
            float const GetAf() const {
                return pimpl.GetAf();
            }
            float const GetIf() const {
                return pimpl.GetIf();
            }
        private:
            ColorData_Detail& pimpl;
        };

        class IDData_Detail {
        public:
            virtual uint32_t const GetIDu32() const = 0;
            virtual uint64_t const GetIDu64() const = 0;
            virtual void SetBasePtr(void const* ptr) = 0;
            virtual IDData_Detail& Clone() const = 0;
            virtual ~IDData_Detail() = default;
        };

        template<class T>
        class IDData_Impl : public IDData_Detail {
        public:
            IDData_Impl() = default;

            IDData_Impl(IDData_Impl const& rhs)
                : basePtr{rhs.basePtr} { }

            virtual uint32_t const GetIDu32() const override {
                return GetID<uint32_t>();
            }

            virtual uint64_t const GetIDu64() const override {
                return GetID<uint64_t>();
            }

            template<class R>
            std::enable_if_t<std::is_same<T, R>::value, R> const GetID() const {
                return this->basePtr[0];
            }

            template<class R>
            std::enable_if_t<!std::is_same<T, R>::value, R> const GetID() const {
                return static_cast<R>(this->basePtr[0]);
            }

            virtual void SetBasePtr(void const* ptr) override {
                this->basePtr = reinterpret_cast<T const*>(ptr);
            }

            virtual IDData_Impl& Clone() const override {
                return IDData_Impl{*this};
            }
        private:

            T const* basePtr;
        };

        class IDData_Base {
        public:
            IDData_Base(IDData_Detail& impl, void const* basePtr)
                : pimpl{impl} {
                pimpl.SetBasePtr(basePtr);
            }

            uint32_t const GetIDu32() const {
                return pimpl.GetIDu32();
            }
            uint64_t const GetIDu64() const {
                return pimpl.GetIDu64();
            }
        private:

            IDData_Detail& pimpl;
        };

        /** Struct holding pointers into data streams for a specific particle */
        struct particle_t {
            VertexData_Base const& vert;
            ColorData_Base  const& col;
            IDData_Base     const& id;
            /*void const* vertPtr;
            void const* colPtr;
            void const* idPtr;*/
        };

        /** possible values for the vertex data */
        enum VertexDataType {
            VERTDATA_NONE = 0, //< indicates that this object is void
            VERTDATA_FLOAT_XYZ = 1, //< use global radius
            VERTDATA_FLOAT_XYZR = 2,
            VERTDATA_SHORT_XYZ = 3 //< quantized positions and global radius
        };

        /** possible values for the colour data */
        enum ColourDataType {
            COLDATA_NONE = 0, //< use global colour
            COLDATA_UINT8_RGB = 1,
            COLDATA_UINT8_RGBA = 2,
            COLDATA_FLOAT_RGB = 3,
            COLDATA_FLOAT_RGBA = 4,
            COLDATA_FLOAT_I = 5 //< single float value to be mapped by a transfer function
        };

        /** possible values for the id data */
        enum IDDataType {
            IDDATA_NONE = 0,
            IDDATA_UINT32 = 1,
            IDDATA_UINT64 = 2
        };

        /** possible values of accumulated data sizes over all vertex coordinates */
        static unsigned int VertexDataSize[4];

        /** possible values of accumulated data sizes over all color elements */
        static unsigned int ColorDataSize[6];

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
        inline ColourDataType GetColourDataType(void) const {
            return this->colDataType;
        }

        /**
         * Answer the colour data pointer
         *
         * @return The colour data pointer
         */
        inline const void * GetColourData(void) const {
            return this->colPtr;
        }

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
        inline UINT64 GetCount(void) const {
            return this->count;
        }

        /**
         * Answer the global colour
         *
         * @return The global colour as a pointer to four unsigned bytes
         *         storing the RGBA colour components
         */
        inline const unsigned char * GetGlobalColour(void) const {
            return this->col;
        }

        /**
         * Answer the global radius
         *
         * @return The global radius
         */
        inline float GetGlobalRadius(void) const {
            return this->radius;
        }

        /**
         * Answer the global particle type
         *
         * @return the global type
         */
        inline unsigned int GetGlobalType(void) const {
            return this->particleType;
        }

        /**
         * Answer the maximum colour index value to be mapped
         *
         * @return The maximum colour index value to be mapped
         */
        inline float GetMaxColourIndexValue(void) const {
            return this->maxColI;
        }

        /**
         * Answer the minimum colour index value to be mapped
         *
         * @return The minimum colour index value to be mapped
         */
        inline float GetMinColourIndexValue(void) const {
            return this->minColI;
        }

        /**
         * Answer the vertex data type
         *
         * @return The vertex data type
         */
        inline VertexDataType GetVertexDataType(void) const {
            return this->vertDataType;
        }

        /**
         * Answer the vertex data pointer
         *
         * @return The vertex data pointer
         */
        inline const void * GetVertexData(void) const {
            return this->vertPtr;
        }

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
        inline IDDataType GetIDDataType(void) const {
            return this->idDataType;
        }

        /**
         * Answer the id data pointer
         *
         * @return The id data pointer
         */
        inline const void * GetIDData(void) const {
            return this->idPtr;
        }

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
        void SetColourData(ColourDataType t, const void *p,
                unsigned int s = 0) {
        //    ASSERT((p != NULL) || (t == COLDATA_NONE));
            this->colDataType = t;
            this->colPtr = p;
            this->colStride = s == 0 ? ColorDataSize[t] : s;

            switch (this->colDataType) {
            case COLDATA_UINT8_RGB:
                this->colorAccessor.reset(new ColorData_Impl<uint8_t, false, false>{});
                break;
            case COLDATA_UINT8_RGBA:
                this->colorAccessor.reset(new ColorData_Impl<uint8_t, true, false>{ });
                break;
            case COLDATA_FLOAT_RGB:
                this->colorAccessor.reset(new ColorData_Impl<float, false, false>{ });
                break;
            case COLDATA_FLOAT_RGBA:
                this->colorAccessor.reset(new ColorData_Impl<float, true, false>{ });
                break;
            case COLDATA_FLOAT_I:
                this->colorAccessor.reset(new ColorData_Impl<float, false, true>{ });
                break;
            case COLDATA_NONE:
            default:
                this->colorAccessor.reset();
            }
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
        void SetGlobalColour(unsigned int r, unsigned int g,
                unsigned int b, unsigned int a = 255) {
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
        void SetGlobalRadius(float r) {
            this->radius = r;
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
        void SetVertexData(VertexDataType t, const void *p,
                unsigned int s = 0) {
            ASSERT(this->disabledNullChecks || (p != NULL) || (t == VERTDATA_NONE));
            this->vertDataType = t;
            this->vertPtr = p;
            this->vertStride = s == 0 ? VertexDataSize[t] : s;

            switch (this->vertDataType) {
            case VERTDATA_FLOAT_XYZ:
                this->vertexAccessor.reset(new VertexData_Impl<float, false>{});
                break;
            case VERTDATA_FLOAT_XYZR:
                this->vertexAccessor.reset(new VertexData_Impl<float, true>{});
                break;
            case VERTDATA_SHORT_XYZ:
                this->vertexAccessor.reset(new VertexData_Impl<short, false>{});
                break;
            case VERTDATA_NONE:
            default:
                this->vertexAccessor.reset();
            }
        }

        /**
         * Sets the ID data
         *
         * @param t The type of the ID data
         * @param p The pointer to the ID data (must not be NULL if t
         *          is not 'IDDATA_NONE'
         * @param s The stride of the ID data
         */
        void SetIDData(IDDataType t, const void *p,
            unsigned int s = 0) {
            ASSERT(this->disabledNullChecks || (p != NULL) || (t == IDDATA_NONE));
            this->idDataType = t;
            this->idPtr = p;
            this->idStride = s == 0 ? IDDataSize[t] : s;

            switch (this->idDataType) {
            case IDDATA_UINT32:
                this->idAccessor.reset(new IDData_Impl<uint32_t>{ });
                break;
            case IDDATA_UINT64:
                this->idAccessor.reset(new IDData_Impl<uint64_t>{});
                break;
            case IDDATA_NONE:
            default:
                this->idAccessor.reset();
            }
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
         * Access particle at index without range check.
         *
         * @param idx Index of particle in the streams.
         *
         * @return Struct of pointers to positions of the particle in the streams.
         */
        particle_t const& operator[](size_t idx) const noexcept;

        /**
         * Access particle at index with range check.
         *
         * @param idx Index of particle in the streams.
         *
         * @return Struct of pointers to positions of the particle in the streams.
         *
         * @throws std::out_of_range if idx is larger than particle count.
         */
        particle_t const& At(size_t idx) const;

        /**
         * Disable NULL-checks in case we have an OpenGL-VAO
         * @param disable flag to disable/enable the checks
         */
        void disableNullChecksForVAOs(bool disable = true)
        {
            disabledNullChecks = disable;
        }
        
        /**
        * Defines wether we transport VAOs instead of real data
        * @param vao flag to disable/enable the checks
        */
        void SetIsVAO(bool vao)
        {
            this->isVAO = vao;
        }

        /**
        * Disable NULL-checks in case we have an OpenGL-VAO
        * @param disable flag to disable/enable the checks
        */
        bool IsVAO()
        {
            return this->isVAO;
        }

        /**
        * If we handle clusters this could be useful
        */
        struct ClusterInfos
        {
            /** a map with clusterid to particleids relation*/
            vislib::Map<int, vislib::Array<int>> data;
            /** the map in plain data for upload to gpu */
            unsigned int *plainData;
            /** size of the plain data*/
            size_t sizeofPlainData;
            /** number of clusters*/
            unsigned int numClusters;
            ClusterInfos() : data(), plainData(0), sizeofPlainData(0), numClusters(0) {};
        };
        
        /**
        * Sets the local ClusterInfos-struct
        */
        void SetClusterInfos(ClusterInfos *infos)
        {
            this->clusterInfos = infos;
        }

        /**
        * gets the local ClusterInfos-struct
        */
        ClusterInfos *GetClusterInfos()
        {
            return this->clusterInfos;
        }

        /**
        * Sets the VertexArrayObject, VertexBuffer and ColorBuffer used
        */
        void SetVAOs(unsigned int vao, unsigned int vb, unsigned int cb)
        {
            this->glVAO = vao;
            this->glVB = vb;
            this->glCB = cb;
        }

        /**
        * Gets the VertexArrayObject, VertexBuffer and ColorBuffer used
        */
        void GetVAOs(unsigned int &vao, unsigned int &vb, unsigned int &cb)
        {
            vao = this->glVAO;
            vb = this->glVB;
            cb = this->glCB;
        }



    private:

        /** The global colour */
        unsigned char col[4];

        /** The colour data type */
        ColourDataType colDataType;

        /** The colour data pointer */
        const void *colPtr;

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

        /** The vertex data type */
        VertexDataType vertDataType;

        /** The vertex data pointer */
        const void *vertPtr;

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
        ClusterInfos *clusterInfos;

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
    };


    MEGAMOLCORE_APIEXT template class MEGAMOLCORE_API AbstractParticleDataCall<SimpleSphericalParticles>;


    /**
     * Call for multi-stream particle data.
     */
    class MEGAMOLCORE_API MultiParticleDataCall
        : public AbstractParticleDataCall<SimpleSphericalParticles> {
    public:

        /** typedef for legacy name */
        typedef SimpleSphericalParticles Particles;

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "MultiParticleDataCall";
        }

        /** Ctor. */
        MultiParticleDataCall(void);

        /** Dtor. */
        virtual ~MultiParticleDataCall(void);

        /**
         * Assignment operator.
         * Makes a deep copy of all members. While for data these are only
         * pointers, the pointer to the unlocker object is also copied.
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         */
        MultiParticleDataCall& operator=(const MultiParticleDataCall& rhs);

    };


    /** Description class typedef */
    typedef factories::CallAutoDescription<MultiParticleDataCall>
        MultiParticleDataCallDescription;


} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_MULTIPARTICLEDATACALL_H_INCLUDED */
