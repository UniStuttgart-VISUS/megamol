/*
 * Client.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CLUSTER_SIMPLE_CLIENT_H_INCLUDED
#define MEGAMOLCORE_CLUSTER_SIMPLE_CLIENT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Module.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/Array.h"
#include "vislib/sys/RunnableThread.h"
#include "vislib/net/SimpleMessageDispatcher.h"
#include "vislib/net/SimpleMessageDispatchListener.h"
#include "vislib/net/Socket.h"
#include "vislib/net/TcpCommChannel.h"
#include "vislib/sys/Thread.h"
//#include "vislib/Serialiser.h"
//#include "vislib/String.h"


namespace megamol {
namespace core {
namespace cluster {
namespace simple {


    /**
     * Abstract base class of override rendering views
     */
    class Client : public Module, public vislib::net::SimpleMessageDispatchListener {
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
        Client(void);

        /** Dtor. */
        virtual ~Client(void);

        /**
         * Unregisters a heartbeat job
         *
         * @param view The heartbeat job to unregister
         */
        void Unregister(class Heartbeat *heartbeat);

        /**
         * Unregisters a view
         *
         * @param view The view to unregister
         */
        void Unregister(class View *view);

        /**
         * Continue setup
         *
         * @param i The setup continuation index
         */
        void ContinueSetup(int i = 0);

        /**
         * Informs the server that this client wants (no) direct camera updates
         *
         * @param yes True if this client wants direct camera updates
         */
        void SetDirectCamSync(bool yes);

        /**
         * Requests a "TimeCameraUpdate"
         *
         * @return true If the request was sent, false if the client does not seem to be connected to a server
         */
        bool RequestTCUpdate(void);

        /**
         * Get the Server Address.
         *
         * @return The server address.
         */
        inline const vislib::StringA ConServerAddress() const {
            return conServerAddr;
        }

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
         * A message from the TCP-Server was received
         *
         * @param src The dispatcher
         * @param msg The received message
         *
         * @return True to continue receiving
         */
        virtual bool OnMessageReceived(vislib::net::SimpleMessageDispatcher& src,
            const vislib::net::AbstractSimpleMessage& msg) throw();

        /**
         * Communication failure
         *
         * @param src The dispatcher
         * @param exception The exception
         *
         * @return False because everything is lost
         */
        virtual bool OnCommunicationError(vislib::net::SimpleMessageDispatcher& src,
            const vislib::Exception& exception) throw();

        /**
         * This method is called immediately after the message dispatcher loop
         * was left and the dispatching method is being exited.
         *
         * This method should return very quickly and should not perform
         * excessive work as it is executed in the discovery thread.
         *
         * @param src The SimpleMessageDispatcher that exited.
         */
        virtual void OnDispatcherExited(vislib::net::SimpleMessageDispatcher& src) throw();

        /**
         * This method is called immediately before the message dispatcher loop
         * is entered, but after the dispatcher was initialised. This method
         * can be used to release references to the communication channel that
         * the caller has and does not need any more.
         *
         * This method should return very quickly and should not perform
         * excessive work as it is executed in the discovery thread.
         *
         * @param src The SimpleMessageDispatcher that exited.
         */
        virtual void OnDispatcherStarted(vislib::net::SimpleMessageDispatcher& src) throw();

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

        /**
         * Sends a simple message
         *
         * @param msg The simple message
         */
        void send(const vislib::net::AbstractSimpleMessage& msg);

        /** The slot views may register at */
        CalleeSlot registerViewSlot;

        /** registered views */
        vislib::Array<class View *> views;

        /** registered heartbeat jobs */
        vislib::Array<class Heartbeat *> heartbeats;

        /** The port used for udp communication */
        param::ParamSlot udpPortSlot;

        /** The socket listening for udp packages */
        vislib::net::Socket udpInSocket;

        /** The udp receiver thread */
        vislib::sys::Thread udpReceiver;

        /** The name of the rendering cluster */
        param::ParamSlot clusterNameSlot;

        /** The address to echo broadcast udp messages */
        param::ParamSlot udpEchoBAddrSlot;

        /** the tcp communication channel */
        vislib::SmartRef<vislib::net::TcpCommChannel> tcpChan;

        /** The TCP communication */
        vislib::sys::RunnableThread<vislib::net::SimpleMessageDispatcher> tcpSan;

        /** The address of the connected server */
        vislib::StringA conServerAddr;

    };


} /* end namespace simple */
} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CLUSTER_SIMPLE_CLIENT_H_INCLUDED */
