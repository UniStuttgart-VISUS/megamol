/*
 * AbstractParticleDataCall.h
 *
 * Copyright (C) VISUS 2011 (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTPARTICLEDATACALL_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTPARTICLEDATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/AbstractGetData3DCall.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/assert.h"
#include "vislib/Array.h"


namespace megamol {
namespace core {
namespace moldyn {


    /**
     * Call for multi-stream particle data.
     *
     * template parameter T is the particle class
     *
     * note: Do not use MEGAMOLCORE_API here(!)
     *       Because it is a template which is incomplete here.
     *       The type is exported as API when it is completed, e.g. "MultiParticleDataCall.h"
     */
    template<class T>
    class /* MEGAMOLCORE_API */ AbstractParticleDataCall : public AbstractGetData3DCall {
    public:

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call to get multi-stream particle sphere data";
        }

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void) {
            return AbstractGetData3DCall::FunctionCount();
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char * FunctionName(unsigned int idx) {
            return AbstractGetData3DCall::FunctionName(idx);
        }

        /**
         * Accesses the particles of list item 'idx'
         *
         * @param idx The zero-based index of the particle list to return
         *
         * @return The requested particle list
         */
        T& AccessParticles(unsigned int idx) {
            return this->lists[idx];
        }

        /**
         * Accesses the particles of list item 'idx'
         *
         * @param idx The zero-based index of the particle list to return
         *
         * @return The requested particle list
         */
        const T& AccessParticles(unsigned int idx) const {
            return this->lists[idx];
        }

        /**
         * Answer the number of particle lists
         *
         * @return The number of particle lists
         */
        inline unsigned int GetParticleListCount(void) const {
            return static_cast<unsigned int>(this->lists.Count());
        }

        /**
         * Sets the number of particle lists. All list items are in undefined
         * states afterward.
         *
         * @param cnt The new number of particle lists
         */
        void SetParticleListCount(unsigned int cnt) {
            this->lists.SetCount(cnt);
        }

        /**
         * Assignment operator.
         * Makes a deep copy of all members. While for data these are only
         * pointers, the pointer to the unlocker object is also copied.
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         */
        AbstractParticleDataCall<T>& operator=(const AbstractParticleDataCall<T>& rhs);

    protected:

        /** Ctor. */
        AbstractParticleDataCall(void);

        /** Dtor. */
        virtual ~AbstractParticleDataCall(void);

    private:

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */
        /** Array of lists of particles */
        vislib::Array<T> lists;
#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */

    };


    /*
     * AbstractParticleDataCall<T>::AbstractParticleDataCall
     */
    template <class T>
    AbstractParticleDataCall<T>::AbstractParticleDataCall(void)
            : AbstractGetData3DCall(), lists() {
        // Intentionally empty
    }


    /*
     * AbstractParticleDataCall<T>::~AbstractParticleDataCall
     */
    template <class T>
    AbstractParticleDataCall<T>::~AbstractParticleDataCall(void) {
        this->Unlock();
        this->lists.Clear();
    }


    /*
     * AbstractParticleDataCall<T>::operator=
     */
    template <class T>
    AbstractParticleDataCall<T>& AbstractParticleDataCall<T>::operator=(
            const AbstractParticleDataCall<T>& rhs) {
        AbstractGetData3DCall::operator =(rhs);
        this->lists.SetCount(rhs.lists.Count());
        for (SIZE_T i = 0; i < this->lists.Count(); i++) {
            this->lists[i] = rhs.lists[i];
        }
        return *this;
    }


} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTPARTICLEDATACALL_H_INCLUDED */
