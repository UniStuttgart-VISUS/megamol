/*
 * CallFrame.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CALLFRAME_H_INCLUDED
#define MEGAMOLCORE_CALLFRAME_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Call.h"
#include "CallAutoDescription.h"

namespace megamol {
namespace core {
namespace protein {


    /**
     * Description:
     * ...
     *
     * 
     * RMSRenderer -> ProteinRenderer
     *             -> ProteinRenderer
     *
     *
     */

    class CallFrame : public Call 
    {
    public:

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "CallFrame";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call to tell renderer which frame should be requested from protein data";
        }

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void) {
            return 1;
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char * FunctionName(unsigned int idx) {
            return "CallFrame";
        }

        /** Ctor. */
        CallFrame(void);

        /** Dtor. */
        virtual ~CallFrame(void);

        /**
         * Answer if there is a new request which is not treated yet.
         * 
         * @return 'True' if there is a new request, 'false' otherwise.
         */
        bool NewRequest(void);

        /**
         * Set the ID for a new requested frame.
         * 
         * @param frmID ID of requested frame.
         */
        void SetFrameRequest(unsigned int frmID);

        /**
         * Answer the ID of the requested frame.
         * 
         * @return ID of current requested frame.
         */
        unsigned int GetFrameRequest(void);

    private:

        /** frame ID of the actual requested frame */
        unsigned int m_frameID;
        /** 'true' if frame ID has changed since last 'get' call */
        bool m_newRequest;

    };

    /** Description class typedef */
    typedef CallAutoDescription<CallFrame> CallFrameDescription;


} /* end namespace protein */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CALLFRAME_H_INCLUDED */
