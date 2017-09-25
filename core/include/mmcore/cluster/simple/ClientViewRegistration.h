/*
 * ClientViewRegistration.h
 *
 * Copyright (C) 2009 - 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CLUSTER_SIMPLE_CLIENTVIEWREGISTRATION_H_INCLUDED
#define MEGAMOLCORE_CLUSTER_SIMPLE_CLIENTVIEWREGISTRATION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"

#include "vislib/net/SimpleMessageDispatchListener.h"


namespace megamol {
namespace core {
namespace cluster {
namespace simple {


    /**
     * Call for registering a module at the cluster controller
     */
    class ClientViewRegistration : public Call {
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
        ClientViewRegistration(void);

        /**
         * ~Dtor.
         */
        virtual ~ClientViewRegistration(void);

        /**
         * Get the client end point
         *
         * @return The client end point
         */
        inline class Client * GetClient(void) {
            return this->client;
        }

        /**
         * Get the client end point
         *
         * @return The client end point
         */
        inline const class Client * GetClient(void) const {
            return this->client;
        }

        /**
         * Get the heartbeat end point
         *
         * @return The heartbeat end point
         */
        inline class Heartbeat * GetHeartbeat(void) {
            return this->heartbeat;
        }

        /**
         * Get the heartbeat end point
         *
         * @return The heartbeat end point
         */
        inline const class Heartbeat * GetHeartbeat(void) const {
            return this->heartbeat;
        }

        /**
         * If raw message dispatching was requested and is possible, return
         * the view as message listener.
         *
         * @return The view end point.
         */
        vislib::net::SimpleMessageDispatchListener *
            GetRawMessageDispatchListener(void);

        /**
         * Get the view end point
         *
         * @return The view end point
         */
        inline class View * GetView(void) {
            return this->view;
        }

        /**
         * Get the view end point
         *
         * @return The view end point
         */
        inline const class View * GetView(void) const {
            return this->view;
        }

        /**
         * Set the client end point
         *
         * @param client The client end point
         */
        inline void SetClient(class Client *client) {
            this->client = client;
        }

        /**
         * Set the heartbeat end point
         *
         * @param heartbeat The heartbeat end point
         */
        inline void SetHeartbeat(class Heartbeat *heartbeat) {
            this->heartbeat = heartbeat;
        }

        /**
         * Enable or disable raw message dispatching from the client to a view 
         * that implements vislib::net::SimpleMessageDispatchListener.
         *
         * @param isRawMessageDispatching true for enabling the function, false
         *                                otherwise.
         *
         */
        inline void SetIsRawMessageDispatching(
                const bool isRawMessageDispatching) {
            this->isRawMessageDispatching = isRawMessageDispatching;
        }

        /**
         * Set the view end point
         *
         * @param view The view end point
         */
        inline void SetView(class View *view) {
            this->view = view;
        }

    private:

        /** The client end */
        class Client *client;

        /** The view end */
        class View *view;

        /** The heartbeat end */
        class Heartbeat *heartbeat;

        /** Enable dispatching of raw messages to compatible clients. */
        bool isRawMessageDispatching;
    };


    /** Description class typedef */
    typedef factories::CallAutoDescription<ClientViewRegistration>
        ClientViewRegistrationDescription;


} /* end namespace simple */
} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CLUSTER_SIMPLE_CLIENTVIEWREGISTRATION_H_INCLUDED */
