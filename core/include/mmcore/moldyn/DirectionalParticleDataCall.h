/*
 * DirectionalParticleDataCall.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_DIRECTIONALPARTICLEDATACALL_H_INCLUDED
#define MEGAMOLCORE_DIRECTIONALPARTICLEDATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/moldyn/AbstractParticleDataCall.h"
#include "MultiParticleDataCall.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/assert.h"


namespace megamol {
namespace core {
namespace moldyn {


    /**
     * Class holding all data of a single directed particle type
     */
    class MEGAMOLCORE_API DirectionalParticles
        : public SimpleSphericalParticles {
    public:
        class DirData_Detail {
        public:
            virtual float const GetDirXf() const = 0;
            virtual float const GetDirYf() const = 0;
            virtual float const GetDirZf() const = 0;
            virtual void SetBasePtr(void const* ptr) = 0;
            virtual std::unique_ptr<DirData_Detail> Clone() const = 0;
            virtual ~DirData_Detail() = default;
        };

        class DirData_None : public DirData_Detail {
        public:
            DirData_None() = default;

            DirData_None(DirData_None const& rhs) = default;

            virtual float const GetDirXf() const override {
                return 0.0f;
            }

            virtual float const GetDirYf() const override {
                return 0.0f;
            }

            virtual float const GetDirZf() const override {
                return 0.0f;
            }


            virtual void SetBasePtr(void const* ptr) override { }

            virtual std::unique_ptr<DirData_Detail> Clone() const override {
                return std::unique_ptr<DirData_Detail>{new DirData_None{*this}};
            }
        };

        template<class T>
        class DirData_Impl : public DirData_Detail {
        public:
            DirData_Impl() = default;

            DirData_Impl(DirData_Impl const& rhs)
                : basePtr{rhs.basePtr} {

            }

            virtual float const GetDirXf() const override {
                return GetDirX<float>();
            }

            template<class R>
            std::enable_if_t<std::is_same<T, R>::value, R> const GetDirX() const {
                return this->basePtr[0];
            }

            template<class R>
            std::enable_if_t<!std::is_same<T, R>::value, R> const GetDirX() const {
                return static_cast<R>(this->basePtr[0]);
            }

            virtual float const GetDirYf() const override {
                return GetDirY<float>();
            }

            template<class R>
            std::enable_if_t<std::is_same<T, R>::value, R> const GetDirY() const {
                return this->basePtr[1];
            }

            template<class R>
            std::enable_if_t<!std::is_same<T, R>::value, R> const GetDirY() const {
                return static_cast<R>(this->basePtr[1]);
            }

            virtual float const GetDirZf() const override {
                return GetDirZ<float>();
            }

            template<class R>
            std::enable_if_t<std::is_same<T, R>::value, R> const GetDirZ() const {
                return this->basePtr[2];
            }

            template<class R>
            std::enable_if_t < !std::is_same<T, R>::value , R > const GetDirZ() const {
                return static_cast<R>(this->basePtr[2]);
            }

            virtual void SetBasePtr(void const* ptr) override {
                this->basePtr = reinterpret_cast<T const*>(ptr);
            }

            virtual std::unique_ptr<DirData_Detail> Clone() const {
                return std::unique_ptr<DirData_Detail>{new DirData_Impl{*this}};
            }
        private:
            T const* basePtr;
        };

        class DirData_Base {
        public:
            DirData_Base(std::unique_ptr<DirData_Detail>&& impl, void const* basePtr)
                : pimpl{std::forward<std::unique_ptr<DirData_Detail>>(impl)} {
                pimpl->SetBasePtr(basePtr);
            }

            DirData_Base(DirData_Base const& rhs) = delete;

            DirData_Base(DirData_Base&& rhs)
                : pimpl{std::forward<std::unique_ptr<DirData_Detail>>(rhs.pimpl)} { }

            DirData_Base& operator=(DirData_Base const& rhs) = delete;

            DirData_Base& operator=(DirData_Base&& rhs) {
                pimpl = std::move(rhs.pimpl);
            }

            float const GetDirXf() const {
                return pimpl->GetDirXf();
            }

            float const GetDirYf() const {
                return pimpl->GetDirYf();
            }

            float const GetDirZf() const {
                return pimpl->GetDirZf();
            }
        private:
            std::unique_ptr<DirData_Detail> pimpl;
        };

        struct dir_particle_t : public particle_t {
            dir_particle_t(VertexData_Base&& v, ColorData_Base&& c, IDData_Base&& i, DirData_Base&& d)
                : particle_t{std::forward<VertexData_Base>(v), std::forward<ColorData_Base>(c), std::forward<IDData_Base>(i)}
                , dir{std::forward<DirData_Base>(d)} { }

            dir_particle_t(particle_t&& par, DirData_Base&& d)
                : particle_t{std::forward<particle_t>(par)}
                , dir{std::forward<DirData_Base>(d)} { }

            DirData_Base dir;
        };

        /** possible values for the direction data */
        enum DirDataType {
            DIRDATA_NONE = 0,
            DIRDATA_FLOAT_XYZ = 1
        };

        /** possible values of data sizes over all directional dimensions */
        static unsigned int DirDataSize[2];

        /**
         * Ctor
         */
        DirectionalParticles(void);

        /**
         * Copy ctor
         *
         * @param src The object to clone from
         */
        DirectionalParticles(const DirectionalParticles& src);

        /**
         * Dtor
         */
        ~DirectionalParticles(void);

        /**
         * Answer the direction data type
         *
         * @return The direction data type
         */
        inline DirDataType GetDirDataType(void) const {
            return this->dirDataType;
        }

        /**
         * Answer the direction data pointer
         *
         * @return The direction data pointer
         */
        inline const void * GetDirData(void) const {
            return this->dirPtr;
        }

        /**
         * Answer the direction data stride
         *
         * @return The direction data stride
         */
        inline unsigned int GetDirDataStride(void) const {
            return this->dirStride == DirDataSize[this->dirDataType] ? 0 : this->dirStride;
        }

        /**
         * Sets the direction data
         *
         * @param t The type of the direction data
         * @param p The pointer to the direction data (must not be NULL if t
         *          is not 'DIRDATA_NONE'
         * @param s The stride of the direction data
         */
        void SetDirData(DirDataType t, const void *p, unsigned int s = 0) {
            ASSERT((p != NULL) || (t == DIRDATA_NONE));
            this->dirDataType = t;
            this->dirPtr = p;
            this->dirStride = s == 0 ? DirDataSize[t] : s;

            switch (t) {
            case DIRDATA_FLOAT_XYZ:
                this->dirAccessor.reset(new DirData_Impl<float>{});
                break;
            case DIRDATA_NONE:
            default:
                this->dirAccessor.reset(new DirData_None{});
            }
        }

        /**
         * Sets the number of objects stored and resets all data pointers!
         *
         * @param cnt The number of stored objects
         */
        void SetCount(UINT64 cnt) {
            this->dirDataType = DIRDATA_NONE;
            this->dirPtr = NULL; // DO NOT DELETE
            SimpleSphericalParticles::SetCount(cnt);
        }

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to 'this'
         */
        DirectionalParticles& operator=(const DirectionalParticles& rhs);

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         *
         * @return 'true' if 'this' and 'rhs' are equal.
         */
        bool operator==(const DirectionalParticles& rhs) const;

        /**
         * Access particle at index without range check.
         *
         * @param idx Index of particle in the streams.
         *
         * @return Struct of pointers to positions of the particle in the streams.
         */
        inline dir_particle_t operator[](size_t idx) const noexcept {
            auto that = dynamic_cast<SimpleSphericalParticles const*>(this);

            return dir_particle_t{
                (*that)[idx],
                DirData_Base{this->dirAccessor->Clone(),
                this->dirPtr != nullptr ? static_cast<char const*>(this->dirPtr) + idx * this->dirStride : nullptr}
            };
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
        inline dir_particle_t const& At(size_t idx) const {
            if (idx < this->GetCount()) {
                return this->operator[](idx);
            } else {
                throw std::out_of_range("Idx larger than particle count.");
            }
        }

    private:

        /** The direction data type */
        DirDataType dirDataType;

        /** The direction data pointer */
        const void *dirPtr;

        /** The direction data stride */
        unsigned int dirStride;

        /** Polymorphic dir access object */
        std::unique_ptr<DirData_Detail> dirAccessor;

    };


    MEGAMOLCORE_APIEXT template class MEGAMOLCORE_API AbstractParticleDataCall<DirectionalParticles>;


    /**
     * Call for multi-stream particle data.
     */
    class MEGAMOLCORE_API DirectionalParticleDataCall
        : public AbstractParticleDataCall<DirectionalParticles> {
    public:

        /** typedef for legacy name */
        typedef DirectionalParticles Particles;

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "DirectionalParticleDataCall";
        }

        /** Ctor. */
        DirectionalParticleDataCall(void);

        /** Dtor. */
        virtual ~DirectionalParticleDataCall(void);

        /**
         * Assignment operator.
         * Makes a deep copy of all members. While for data these are only
         * pointers, the pointer to the unlocker object is also copied.
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         */
        DirectionalParticleDataCall& operator=(const DirectionalParticleDataCall& rhs);

    };


    /** Description class typedef */
    typedef factories::CallAutoDescription<DirectionalParticleDataCall>
        DirectionalParticleDataCallDescription;


} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_DIRECTIONALPARTICLEDATACALL_H_INCLUDED */
