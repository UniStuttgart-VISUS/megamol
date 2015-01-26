/*
 * Server.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CLUSTER_SIMPLE_SERVER_H_INCLUDED
#define MEGAMOLCORE_CLUSTER_SIMPLE_SERVER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "job/AbstractJob.h"
#include "Module.h"
#include "CallerSlot.h"
#include "param/ParamSlot.h"
#include "param/ParamUpdateListener.h"
#include "cluster/simple/CommUtil.h"
#include "vislib/net/CommServer.h"
#include "vislib/net/CommServerListener.h"
#include "vislib/sys/CriticalSection.h"
#include "vislib/net/IPEndPoint.h"
#include "vislib/PtrArray.h"
#include "vislib/sys/RunnableThread.h"
#include "vislib/SmartRef.h"
#include "vislib/net/Socket.h"
#include "vislib/net/SimpleMessageDispatcher.h"
#include "vislib/net/SimpleMessageDispatchListener.h"
#include "vislib/String.h"


namespace megamol {
namespace core {
namespace cluster {
namespace simple {


    /**
     * Abstract base class of override rendering views
     */
    class Server : public Module, public job::AbstractJob,
        public vislib::net::CommServerListener, public param::ParamUpdateListener {
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
        Server(void);

        /** Dtor. */
        virtual ~Server(void);

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
            vislib::SmartRef<vislib::net::AbstractCommClientChannel> channel) throw();

        /**
         * Callback called when a parameter is updated
         *
         * @param slot The parameter updated
         */
        virtual void ParamUpdated(param::ParamSlot& slot);

    private:

        /**
         * Class managing a connected client
         */
        class Client : public vislib::net::SimpleMessageDispatchListener {
        public:

            /**
             * Ctor
             *
             * @param parent The parent server
             * @param channel The communication channel
             */
            Client(Server& parent, vislib::SmartRef<vislib::net::AbstractCommClientChannel> channel);

            /** Dtor */
            virtual ~Client(void);

            /** Prepare the client to be terminated by closing the connections */
            void Close(void);

            /**
             * This method is called once a communication error occurs.
             *
             * This method should return very quickly and should not perform
             * excessive work as it is executed in the discovery thread.
             *
             * The return value of the method can be used to stop the message
             * dispatcher. The default implementation returns true for continuing
             * after an error.
             *
             * Note that the dispatcher will stop if any of the registered listeners
             * returns false.
             *
             * @param src       The SimpleMessageDispatcher which caught the 
             *                  communication error.
             * @param exception The exception that was caught (this exception
             *                  represents the error that occurred).
             *
             * @return true in order to make the SimpleMessageDispatcher continue
             *         receiving messages, false will cause the dispatcher to
             *         exit.
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
            virtual bool OnMessageReceived(vislib::net::SimpleMessageDispatcher& src,
                const vislib::net::AbstractSimpleMessage& msg) throw();

            /**
             * Answer whether or not this client is running
             *
             * @return True if this client is running
             */
            inline bool IsRunning(void) const {
                return this->dispatcher.IsRunning();
            }

            /**
             * Sends a simple message
             *
             * @param msg The simple message
             */
            inline void Send(const vislib::net::AbstractSimpleMessage& msg) {
                this->send(msg);
            }

            /**
             * Gets the flag whether or not this client wants camera updates
             */
            inline bool WantCameraUpdates(void) const {
                return this->wantCamUpdates;
            }

        private:

            /**
             * Sends a simple message
             *
             * @param msg The simple message
             */
            void send(const vislib::net::AbstractSimpleMessage& msg);

            /** The parent object */
            Server& parent;

            /** The dispatcher thread */
            vislib::sys::RunnableThread<vislib::net::SimpleMessageDispatcher> dispatcher;

            /** Flag marking an imminent termination */
            bool terminationImminent;

            /** The name of the computer connected */
            vislib::StringA name;

            /** Flag whether or not this client wants camera updates */
            bool wantCamUpdates;

            /** The camera sync number of the camera information last transmitted */
            unsigned int lastTCSyncNumber;

        };

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
        void sendUDPDiagram(Datagram& datagram);

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
         * Callback triggered when the parameter value changes
         *
         * @param slot The calling slot
         *
         * @return True
         */
        bool onServerStartStopClicked(param::ParamSlot& slot);

        /**
         * The thread function for camera updates
         *
         * @param userData
         *
         * @return 0
         */
        static DWORD cameraUpdateThread(void *userData);

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

        /** The server running flag */
        param::ParamSlot serverStartSlot;

        /** The server running flag */
        param::ParamSlot serverStopSlot;

        /** The server port slot */
        param::ParamSlot serverPortSlot;

        /** Button to send the clients a reconnect message */
        param::ParamSlot serverReconnectSlot;

        /** Button to restart the TCP server */
        param::ParamSlot serverRestartSlot;

        /** The name of this server */
        param::ParamSlot serverNameSlot;

        /** Restricts the server to a single client at a time. */
        param::ParamSlot singleClientSlot;

        /** The server thread */
        vislib::sys::RunnableThread<vislib::net::CommServer> serverThread;

        /** The thread lock for the clients list */
        vislib::sys::CriticalSection clientsLock;

        /** The connected clients */
        vislib::PtrArray<Client> clients; /* *HAZARD* TODO: Fixme -> executing code in deleted objects */

        /** The thread to update the camera settings */
        vislib::sys::Thread camUpdateThread;

        /** Use the force luke */
        bool camUpdateThreadForce;

        /** Client receivers have access */
        friend class Client;

    };


} /* end namespace simple */
} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CLUSTER_SIMPLE_SERVER_H_INCLUDED */
