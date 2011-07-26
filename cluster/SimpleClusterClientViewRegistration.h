/*
 * SimpleClusterClientViewRegistration.h
 *
 * Copyright (C) 2009 - 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_SIMPLECLUSTERCLIENTVIEWREGISTRATION_H_INCLUDED
#define MEGAMOLCORE_SIMPLECLUSTERCLIENTVIEWREGISTRATION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Call.h"
#include "CallAutoDescription.h"


namespace megamol {
namespace core {
namespace cluster {


    /**
     * Call for registering a module at the cluster controller
     */
    class SimpleClusterClientViewRegistration : public Call {
    public:

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "SimpleClusterClientViewRegistration";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call for registering a simple cluster view at the simple cluster client";
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
            return "register";
        }

        /**
         * Ctor.
         */
        SimpleClusterClientViewRegistration(void);

        /**
         * ~Dtor.
         */
        virtual ~SimpleClusterClientViewRegistration(void);

        /**
         * Get the client end point
         *
         * @return The client end point
         */
        inline class SimpleClusterClient * GetClient(void) {
            return this->client;
        }

        /**
         * Get the client end point
         *
         * @return The client end point
         */
        inline const class SimpleClusterClient * GetClient(void) const {
            return this->client;
        }

        /**
         * Get the heartbeat end point
         *
         * @return The heartbeat end point
         */
        inline class SimpleClusterHeartbeat * GetHeartbeat(void) {
            return this->heartbeat;
        }

        /**
         * Get the heartbeat end point
         *
         * @return The heartbeat end point
         */
        inline const class SimpleClusterHeartbeat * GetHeartbeat(void) const {
            return this->heartbeat;
        }

        /**
         * Get the view end point
         *
         * @return The view end point
         */
        inline class SimpleClusterView * GetView(void) {
            return this->view;
        }

        /**
         * Get the view end point
         *
         * @return The view end point
         */
        inline const class SimpleClusterView * GetView(void) const {
            return this->view;
        }

        /**
         * Set the client end point
         *
         * @param client The client end point
         */
        inline void SetClient(class SimpleClusterClient *client) {
            this->client = client;
        }

        /**
         * Set the heartbeat end point
         *
         * @param heartbeat The heartbeat end point
         */
        inline void SetHeartbeat(class SimpleClusterHeartbeat *heartbeat) {
            this->heartbeat = heartbeat;
        }

        /**
         * Set the view end point
         *
         * @param view The view end point
         */
        inline void SetView(class SimpleClusterView *view) {
            this->view = view;
        }

    private:

        /** The client end */
        class SimpleClusterClient *client;

        /** The view end */
        class SimpleClusterView *view;

        class SimpleClusterHeartbeat *heartbeat;

    };


    /** Description class typedef */
    typedef CallAutoDescription<SimpleClusterClientViewRegistration>
        SimpleClusterClientViewRegistrationDescription;


} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_SIMPLECLUSTERCLIENTVIEWREGISTRATION_H_INCLUDED */
