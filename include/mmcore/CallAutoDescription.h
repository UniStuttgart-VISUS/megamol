/*
 * CallAutoDescription.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CALLAUTODESCRIPTION_H_INCLUDED
#define MEGAMOLCORE_CALLAUTODESCRIPTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CallDescription.h"


namespace megamol {
namespace core {


    /**
     * Description class for the call T
     */
    template<class T>
    class CallAutoDescription : public CallDescription {
    public:

        /** Ctor. */
        CallAutoDescription(void) : CallDescription() {
            // intentionally empty
        }

        /** Dtor. */
        virtual ~CallAutoDescription(void) {
            // intentionally empty
        }

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        virtual const char *ClassName(void) const {
            return T::ClassName();
        }

        /**
         * Clones this object
         *
         * @return The new clone
         */
        virtual CallDescription *Clone(void) const {
            return new CallAutoDescription<T>();
        }

        /**
         * Creates a new call object.
         *
         * @return The newly created call object.
         */
        virtual Call * CreateCall(void) const {
            return this->describeCall(new T());
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        virtual const char *Description(void) const {
            return T::Description();
        }

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        virtual unsigned int FunctionCount(void) const {
            return T::FunctionCount();
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        virtual const char * FunctionName(unsigned int idx) const {
            return T::FunctionName(idx);
        }

        /**
         * Answers whether this description is describing the class of
         * 'call'.
         *
         * @param call The call to test.
         *
         * @return 'true' if 'call' is described by this description,
         *         'false' otherwise.
         */
        virtual bool IsDescribing(const Call * call) const {
            return dynamic_cast<const T*>(call) != NULL;
        }

        /**
         * Assignment crowbar
         *
         * @param tar The targeted object
         * @param src The source object
         */
        virtual void AssignmentCrowbar(Call * tar, const Call * src) const {
            T* t = dynamic_cast<T*>(tar);
            const T* s = dynamic_cast<const T*>(src);
            *t = *s;
        }

    };


} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CALLAUTODESCRIPTION_H_INCLUDED */
