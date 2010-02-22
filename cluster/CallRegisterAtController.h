/*
 * CallRegisterAtController.h
 *
 * Copyright (C) 2009 - 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CALLREGISTERATCONTROLLER_H_INCLUDED
#define MEGAMOLCORE_CALLREGISTERATCONTROLLER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Call.h"
#include "CallAutoDescription.h"


namespace megamol {
namespace core {
namespace cluster {

    /** forward declaration */
    class ClusterControllerClient;


    /**
     * Call for registering a module at the cluster controller
     */
    class CallRegisterAtController : public Call {
    public:

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "CallRegisterAtController";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call for registering a module at the cluster controller";
        }

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void) {
            return 2;
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char * FunctionName(unsigned int idx) {
            switch (idx) {
                case 0: return "register";
                case 1: return "unregister";
                default: return NULL;
            }
        }

        /**
         * Ctor.
         */
        CallRegisterAtController(void);

        /**
         * ~Dtor.
         */
        virtual ~CallRegisterAtController(void);

        /**
         * Gets the client to be un-/registered
         *
         * @return The client to be un-/registered
         */
        inline ClusterControllerClient * Client(void) {
            return this->client;
        }

        /**
         * Gets the client to be un-/registered
         *
         * @return The client to be un-/registered
         */
        inline const ClusterControllerClient * Client(void) const {
            return this->client;
        }

        /**
         * Sets the client to be un-/registered
         *
         * @param c The client to be un-/registered
         */
        inline void SetClient(ClusterControllerClient *c) {
            this->client = c;
        }

    private:

        /** The client to be un-/registered */
        ClusterControllerClient *client;

    };


    /** Description class typedef */
    typedef CallAutoDescription<CallRegisterAtController>
        CallRegisterAtControllerDescription;


} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CALLREGISTERATCONTROLLER_H_INCLUDED */
