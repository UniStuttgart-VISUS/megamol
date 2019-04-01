/*
 * DirectionalParticleDataCall.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_DIRECTIONALPARTICLEDATACALL_H_INCLUDED
#define MEGAMOLCORE_DIRECTIONALPARTICLEDATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <type_traits>

#include "MultiParticleDataCall.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/moldyn/AbstractParticleDataCall.h"
#include "vislib/assert.h"


namespace megamol {
namespace core {
namespace moldyn {


/**
 * Class holding all data of a single directed particle type
 */
class MEGAMOLCORE_API DirectionalParticles : public SimpleSphericalParticles {
public:
    /** possible values for the direction data */
    enum DirDataType { DIRDATA_NONE = 0, DIRDATA_FLOAT_XYZ = 1 };

    /** possible values of data sizes over all directional dimensions */
    static unsigned int DirDataSize[2];

    /**
     * This class holds the accessors to the current data.
     */
    class DirectionalParticleStore : public ParticleStore {
    public:
        explicit DirectionalParticleStore() = default;

        virtual ~DirectionalParticleStore() = default;

        void SetDirData(DirectionalParticles::DirDataType const t, char const* p, unsigned int const s = 0) {
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

        std::shared_ptr<Accessor> const& GetDXAcc() const { return this->dx_acc_; }

        std::shared_ptr<Accessor> const& GetDYAcc() const { return this->dy_acc_; }

        std::shared_ptr<Accessor> const& GetDZAcc() const { return this->dz_acc_; }

    private:
        std::shared_ptr<Accessor> dx_acc_ = std::make_shared<Accessor_0>();
        std::shared_ptr<Accessor> dy_acc_ = std::make_shared<Accessor_0>();
        std::shared_ptr<Accessor> dz_acc_ = std::make_shared<Accessor_0>();
    };

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
    inline DirDataType GetDirDataType(void) const { return this->dirDataType; }

    /**
     * Answer the direction data pointer
     *
     * @return The direction data pointer
     */
    inline const void* GetDirData(void) const { return this->dirPtr; }

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
    void SetDirData(DirDataType t, const void* p, unsigned int s = 0) {
        ASSERT((p != NULL) || (t == DIRDATA_NONE));
        this->dirDataType = t;
        this->dirPtr = p;
        this->dirStride = s == 0 ? DirDataSize[t] : s;

        std::dynamic_pointer_cast<DirectionalParticleStore, ParticleStore>(this->par_store_)
            ->SetDirData(t, reinterpret_cast<char const*>(p), this->dirStride);
    }

    /**
     * Sets the number of objects stored and resets all data pointers!
     *
     * @param cnt The number of stored objects
     */
    void SetCount(UINT64 cnt) {
        this->dirDataType = DIRDATA_NONE;
        this->dirPtr = NULL; // DO NOT DELETE
        std::dynamic_pointer_cast<DirectionalParticleStore, ParticleStore>(this->par_store_)
            ->SetDirData(DIRDATA_NONE, nullptr);
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
     * Get instance of particle store call the accessors.
     *
     * @return Instance of particle store.
     */
    DirectionalParticleStore const& GetParticleStore() const {
        return *std::dynamic_pointer_cast<DirectionalParticleStore, ParticleStore>(this->par_store_);
    }

private:
    /** The direction data type */
    DirDataType dirDataType;

    /** The direction data pointer */
    const void* dirPtr;

    /** The direction data stride */
    unsigned int dirStride;

    /** Instance of the particle store */
    // std::shared_ptr<DirectionalParticleStore> par_store_;
};


MEGAMOLCORE_APIEXT template class MEGAMOLCORE_API AbstractParticleDataCall<DirectionalParticles>;


/**
 * Call for multi-stream particle data.
 */
class MEGAMOLCORE_API DirectionalParticleDataCall : public AbstractParticleDataCall<DirectionalParticles> {
public:
    /** typedef for legacy name */
    typedef DirectionalParticles Particles;

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) { return "DirectionalParticleDataCall"; }

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
typedef factories::CallAutoDescription<DirectionalParticleDataCall> DirectionalParticleDataCallDescription;


} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_DIRECTIONALPARTICLEDATACALL_H_INCLUDED */
