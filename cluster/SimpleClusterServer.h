/*
 * SimpleClusterServer.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_SIMPLECLUSTERSERVER_H_INCLUDED
#define MEGAMOLCORE_SIMPLECLUSTERSERVER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "job/AbstractJob.h"
#include "Module.h"
#include "CallerSlot.h"
#include "param/ParamSlot.h"
#include "cluster/SimpleClusterDatagram.h"
//#include "vislib/Array.h"
#include "vislib/IPEndPoint.h"
#include "vislib/Socket.h"
//#include "vislib/Thread.h"
//#include "vislib/Serialiser.h"
//#include "vislib/String.h"


namespace megamol {
namespace core {
namespace cluster {


    /**
     * Abstract base class of override rendering views
     */
    class SimpleClusterServer : public Module, public job::AbstractJob {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "SimpleClusterServer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Simple Powerwall-Fusion Server";
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
        SimpleClusterServer(void);

        /** Dtor. */
        virtual ~SimpleClusterServer(void);

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

        /**
         * Answers whether or not this job is still running.
         *
         * @return 'true' if this job is still running, 'false' if it has
         *         finished.
         */
        virtual bool IsRunning(void) const;

        /**
         * Starts the job thread.
         *
         * @return true if the job has been successfully started.
         */
        virtual bool Start(void);

        /**
         * Terminates the job thread.
         *
         * @return true to acknowledge that the job will finish as soon
         *         as possible, false if termination is not possible.
         */
        virtual bool Terminate(void);

    private:

        /**
         * Callback called when the rendering node instances should be shut down
         *
         * @param slot clusterShutdownBtnSlot
         *
         * @return true
         */
        bool onShutdownClusterClicked(param::ParamSlot& slot);

        /**
         * Callback called when the udp target updates
         *
         * @param slot udpTargetSlot
         *
         * @return true
         */
        bool onUdpTargetUpdated(param::ParamSlot& slot);

        /**
         * Callback called when the view name updates
         *
         * @param slot viewnameSlot
         *
         * @return true
         */
        bool onViewNameUpdated(param::ParamSlot& slot);

        /**
         * Informs somebody that the view has been disconnected
         */
        void disconnectView(void);

        /**
         * Informs somebody that a new view has been connected
         */
        void newViewConnected(void);

        /**
         * Sends a datagram
         *
         * @param datagram The datagram
         */
        void sendUDPDiagram(SimpleClusterDatagram& datagram);

        /** The parameter slot holding the name of the view module to be use */
        param::ParamSlot viewnameSlot;

        /** The status of the connection to the view module */
        int viewConStatus;

        /** The slot connecting to the view to be synchronized */
        CallerSlot viewSlot;

        /** The udp target */
        param::ParamSlot udpTargetSlot;

        /** The port used for udp communication */
        param::ParamSlot udpTargetPortSlot;

        /** The udp target */
        vislib::net::IPEndPoint udpTarget;

        /** The socket listening for udp packages */
        vislib::net::Socket udpSocket;

        /** The button to shutdown the rendering instances */
        param::ParamSlot clusterShutdownBtnSlot;

    };


} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_SIMPLECLUSTERSERVER_H_INCLUDED */
