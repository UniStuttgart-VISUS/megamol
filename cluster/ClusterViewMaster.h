/*
 * ClusterViewMaster.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CLUSTERVIEWMASTER_H_INCLUDED
#define MEGAMOLCORE_CLUSTERVIEWMASTER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CallerSlot.h"
#include "cluster/ClusterControllerClient.h"
#include "Module.h"
#include "param/ParamSlot.h"
#include "vislib/AbstractServerEndPoint.h"
#include "vislib/CommServer.h"
#include "vislib/CommServerListener.h"
#include "vislib/RunnableThread.h"
#include "vislib/SmartRef.h"


namespace megamol {
namespace core {
namespace cluster {


    /**
     * Abstract base class of override rendering views
     */
    class ClusterViewMaster : public Module, public ClusterControllerClient,
        public vislib::net::CommServerListener {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "ClusterViewMaster";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Master view controller module for distributed, tiled rendering";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor. */
        ClusterViewMaster(void);

        /** Dtor. */
        virtual ~ClusterViewMaster(void);

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
         * A message has been received.
         *
         * @param hPeer The peer which sent the message
         * @param msgType The type value of the message
         * @param msgBody The data of the message
         */
        virtual void OnUserMsg(const ClusterController::PeerHandle& hPeer,
            const UINT32 msgType, const BYTE *msgBody);

        /**
         * This method is called every time a message is received.
         *
         * This method should return very quickly and should not perform
         * excessive work as it is executed in the discovery thread.
         *
         * The return value of the method can be used to stop the message
         * dispatcher, e. g. if an exit message was received. 
         *
         * Note that the dispatcher will stop if any of the registered listeners
         * returns false.
         *
         * @param src The SimpleMessageDispatcher that received the message.
         * @param msg The message that was received.
         *
         * @return true in order to make the SimpleMessageDispatcher continue
         *         receiving messages, false will cause the dispatcher to
         *         exit.
         */
        virtual bool OnMessageReceived(
            vislib::net::SimpleMessageDispatcher& src,
            const vislib::net::AbstractSimpleMessage& msg) throw();

        /**
         * Reacts on changes of the view name parameter
         *
         * @param slot Must be 'viewNameSlot'
         *
         * @return 'true' to reset the dirty flag.
         */
        bool onViewNameChanged(param::ParamSlot& slot);

        /**
         * Update callback when the control communication address changes
         *
         * @param address The new address string to be used
         */
        virtual void OnCtrlCommAddressChanged(const vislib::TString& address);

        /**
         * This method is called once a network error occurs.
         *
         * This method should return very quickly and should not perform
         * excessive work as it is executed in the server thread.
         *
         * The return value of the method can be used to stop the server. The 
         * default implementation returns true for continuing after an error.
         *
         * Note that the server will stop if any of the registered listeners 
         * returns false.
         *
         * @param src       The CommServer which caught the communication error.
         * @param exception The exception that was caught (this exception
         *                  represents the error that occurred).
         *
         * @return true in order to make the CommServer continue listening, 
         *         false will cause the server to exit.
         */
        virtual bool OnServerError(const vislib::net::CommServer& src,
            const vislib::Exception& exception) throw();

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

        /**
         * The server will call this method when it left the server loop and
         * is about to exit.
         *
         * Subclasses must not throw exceptions within this method.
         *
         * Subclasses should return as soon as possible from this method.
         *
         * @param serv The server that exited.
         */
        virtual void OnServerExited(const vislib::net::CommServer& src) throw();

        /**
         * The server will call this method immediately before entering the 
         * server loop, but after the communication channel was put into
         * listening state.
         *
         * Subclasses must not throw exceptions within this method.
         *
         * Subclasses should return as soon as possible from this method.
         *
         * @param serv The server that started.
         */
        virtual void OnServerStarted(const vislib::net::CommServer& src) throw();

    private:

        /**
         * Answer the default port
         *
         * @return The default port
         */
        UINT16 defaultPort(void) const;

        /**
         * Answer the default IP address, including port, to run the server socket on
         *
         * @return The default IP address, including port
         */
        vislib::TString defaultServerAddress(void) const;

        /** The name of the view to be used */
        param::ParamSlot viewNameSlot;

        /** The slot connecting to the view to be used */
        CallerSlot viewSlot;

        /** Server of the control message communication channel */
        vislib::sys::RunnableThread<vislib::net::CommServer> commCtrlServer;

        /** Flag indicating that the commCtrlServer is shutting down */
        bool commCtrlServerShutdown;

        ///** The communication channel for control commands */
        //vislib::SmartRef<vislib::net::AbstractServerEndPoint> commChnlCtrl;

        ///** The communication channel for camera updates */
        //vislib::SmartRef<vislib::net::AbstractServerEndPoint> commChnlCam;

    };


} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CLUSTERVIEWMASTER_H_INCLUDED */
