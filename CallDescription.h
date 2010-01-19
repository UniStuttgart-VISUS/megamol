/*
 * CallDescription.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CALLDESCRIPTION_H_INCLUDED
#define MEGAMOLCORE_CALLDESCRIPTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "api/MegaMolCore.std.h"
#include "ObjectDescription.h"


namespace megamol {
namespace core {

    /** forward declaration of Call */
    class Call;


    /**
     * Description class for calls
     */
    class MEGAMOLCORE_API CallDescription : public ObjectDescription {
    public:

        /** Ctor. */
        CallDescription(void);

        /** Dtor. */
        virtual ~CallDescription(void);

        /**
         * Answer the class name of the module described.
         *
         * @return The class name of the module described.
         */
        virtual const char *ClassName(void) const = 0;

        /**
         * Creates a new call object.
         *
         * @return The newly created call object.
         */
        virtual Call * CreateCall(void) const = 0;

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        virtual const char *Description(void) const = 0;

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        virtual unsigned int FunctionCount(void) const = 0;

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        virtual const char * FunctionName(unsigned int idx) const = 0;

        /**
         * Answers whether this description is describing the class of
         * 'call'.
         *
         * @param call The call to test.
         *
         * @return 'true' if 'call' is described by this description,
         *         'false' otherwise.
         */
        virtual bool IsDescribing(const Call * call) const = 0;

    protected:

        /**
         * Describes the call 'call'. Must be called for all calles created by
         * their describtion object before they are returned by 'CreateCall'.
         *
         * @param call The call to be described.
         *
         * @return 'call'
         */
        Call * describeCall(Call * call) const;

    };


} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CALLDESCRIPTION_H_INCLUDED */
