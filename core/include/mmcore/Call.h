/*
 * Call.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CALL_H_INCLUDED
#define MEGAMOLCORE_CALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/api/MegaMolCore.std.h"


namespace megamol {
namespace core {

    /** Forward declaration of description and slots */
    class CalleeSlot;
    class CallerSlot;
    namespace factories {
        class CallDescription;
    }


    /**
     * Base class of rendering graph calls
     */
    class MEGAMOLCORE_API Call {
    public:

        /** The description generates the function map */
        friend class ::megamol::core::factories::CallDescription;

        /** Callee slot is allowed to map functions */
        friend class CalleeSlot;

        /** The caller slot registeres itself in the call */
        friend class CallerSlot;

        /** Ctor. */
        Call(void);

        /** Dtor. */
        virtual ~Call(void);

        /**
         * Calls function 'func'.
         *
         * @param func The function to be called.
         *
         * @return The return value of the function.
         */
        bool operator()(unsigned int func = 0);

        /**
         * Answers the callee slot this call is connected to.
         *
         * @return The callee slot this call is connected to.
         */
        inline const CalleeSlot * PeekCalleeSlot(void) const {
            return this->callee;
        }

        /**
         * Answers the caller slot this call is connected to.
         *
         * @return The caller slot this call is connected to.
         */
        inline const CallerSlot * PeekCallerSlot(void) const {
            return this->caller;
        }

        inline void SetClassName(const char *name) {
            this->className = name;
        }

        inline const char * ClassName() const {
            return this->className;
        }

    private:

        /** The callee connected by this call */
        CalleeSlot *callee;

        /** The caller connected by this call */
        CallerSlot *caller;

        const char *className;

        /** The function id mapping */
        unsigned int *funcMap;

    };


} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CALL_H_INCLUDED */
