/*
 * SimpleClusterClient.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_SIMPLECLUSTERCLIENT_H_INCLUDED
#define MEGAMOLCORE_SIMPLECLUSTERCLIENT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Module.h"
#include "CalleeSlot.h"
#include "param/ParamSlot.h"
#include "vislib/Array.h"
#include "vislib/Socket.h"
#include "vislib/Thread.h"
//#include "vislib/Serialiser.h"
//#include "vislib/String.h"


namespace megamol {
namespace core {
namespace cluster {


    /**
     * Abstract base class of override rendering views
     */
    class SimpleClusterClient : public Module {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "SimpleClusterClient";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Simple Powerwall-Fusion Client";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /**
         * Disallow usage in quickstarts
         *
         * @return false
         */
        static bool SupportQuickstart(void) {
            return false;
        }

        /** Ctor. */
        SimpleClusterClient(void);

        /** Dtor. */
        virtual ~SimpleClusterClient(void);

        /**
         * Unregisters a view
         *
         * @param view The view to unregister
         */
        void Unregister(class SimpleClusterView *view);

    protected:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

    private:

        /**
         * The udp receiver thread code
         *
         * @param ctxt The user context object
         *
         * @return The return value
         */
        static DWORD udpReceiverLoop(void *ctxt);

        /**
         * Callback called when views register
         *
         * @param call The incoming call
         *
         * @return The return value
         */
        bool onViewRegisters(Call& call);

        /**
         * Callback used when the UdpPort changes
         *
         * @param slot this->udpPortSlot
         *
         * @return True
         */
        bool onUdpPortChanged(param::ParamSlot& slot);

        /** The slot views may register at */
        CalleeSlot registerViewSlot;

        /** registered views */
        vislib::Array<class SimpleClusterView *> views;

        /** The port used for udp communication */
        param::ParamSlot udpPortSlot;

        /** The socket listening for udp packages */
        vislib::net::Socket udpInSocket;

        /** The udp receiver thread */
        vislib::sys::Thread udpReceiver;

    };


} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_SIMPLECLUSTERCLIENT_H_INCLUDED */
