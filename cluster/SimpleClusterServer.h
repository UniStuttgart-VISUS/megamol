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
#include "cluster/SimpleClusterCommUtil.h"
//#include "vislib/Array.h"
#include "vislib/CommServer.h"
#include "vislib/CommServerListener.h"
#include "vislib/IPEndPoint.h"
#include "vislib/RunnableThread.h"
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
    class SimpleClusterServer : public Module, public job::AbstractJob,
        public vislib::net::CommServerListener {
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

        /**
         * The server will call this method when a new client connected. The
         * listener can decide whether it wants to take ownership of the
         * communication channel 'channel' by returning true. If no listener 
         * accepts the new connection, the server will terminate the new 
         * connection by closing it.
         *
         * Note that no other listeners will be informed after the first one
         * has accepted the connection by returning true. This first 
         * listener is regarded as new owner of 'channel' by the server.
         *
         * Subclasses must not throw exceptions within this method.
         *
         * Subclasses should return as soon as possible from this method.
         *
         * @param src     The server that made the new channel.
         * @param channel The new communication channel.
         * 
         * @return true if the listener takes ownership of 'channel'. The 
         *         server will not use the channel again. If the method 
         *         returns false, the listener should not use the socket, 
         *         because the server remains its owner.
         */
        virtual bool OnNewConnection(const vislib::net::CommServer& src,
            vislib::SmartRef<vislib::net::AbstractCommChannel> channel) throw();

    private:

        ///**
        // * Class managing a connected client
        // */
        //class Client {
        //public:

        //    /** Ctor */
        //    Client(void);

        //    /** Dtor */
        //    virtual ~Client(void);

        //};

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

        /**
         * Stops the server
         */
        void stopServer(void);

        /**
         * Callback triggered when the parameter value changes
         *
         * @param slot The calling slot
         *
         * @return True
         */
        bool onServerRunningChanged(param::ParamSlot& slot);

        /**
         * Callback triggered when the parameter value changes
         *
         * @param slot The calling slot
         *
         * @return True
         */
        bool onServerEndPointChanged(param::ParamSlot& slot);

        /**
         * Callback triggered when the parameter value changes
         *
         * @param slot The calling slot
         *
         * @return True
         */
        bool onServerReconnectClicked(param::ParamSlot& slot);

        /**
         * Callback triggered when the parameter value changes
         *
         * @param slot The calling slot
         *
         * @return True
         */
        bool onServerRestartClicked(param::ParamSlot& slot);

        /**
         * gets the configured server end point
         *
         * @param outEP The end point
         *
         * @return True on success
         */
        bool getServerEndPoint(vislib::net::IPEndPoint& outEP);

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

        /** The name of the rendering cluster */
        param::ParamSlot clusterNameSlot;

        /** The server running flag */
        param::ParamSlot serverRunningSlot;

        /** The server endpoint slot */
        param::ParamSlot serverEndPointAddrSlot;

        /** The server endpoint port slot */
        param::ParamSlot serverEndPointPortSlot;

        /** Button to send the clients a reconnect message */
        param::ParamSlot serverReconnectSlot;

        /** Button to restart the TCP server */
        param::ParamSlot serverRestartSlot;

        /** The server thread */
        vislib::sys::RunnableThread<vislib::net::CommServer> serverThread;

        ///** The endpoint to run the server on */
        //vislib::net::IPEndPoint serverEndPoint;

        ///** The server */
        //vislib::sys::RunnableThread<vislib::net::TcpServer> server;

        ///** The thread lock for the clients list */
        //vislib::sys::CriticalSection clientsLock;

        ///** The connected clients */
        //vislib::Array<> clients;

    };


} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_SIMPLECLUSTERSERVER_H_INCLUDED */
