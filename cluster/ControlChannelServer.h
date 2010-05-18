/*
 * ControlChannelServer.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CONTROLCHANNELSERVER_H_INCLUDED
#define MEGAMOLCORE_CONTROLCHANNELSERVER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "cluster/ControlChannel.h"
#include "vislib/AbstractSimpleMessage.h"
#include "vislib/CommServer.h"
#include "vislib/CommServerListener.h"
#include "vislib/CriticalSection.h"
#include "vislib/Listenable.h"
#include "vislib/RunnableThread.h"
#include "vislib/SmartRef.h"
#include "vislib/TcpCommChannel.h"


namespace megamol {
namespace core {
namespace cluster {

    /**
     * class for control communication channel end points
     */
    class ControlChannelServer : public vislib::Listenable<ControlChannelServer>,
        protected vislib::net::CommServerListener, protected ControlChannel::Listener {
    public:

        /**
         * Class for listener object
         */
        class Listener : public vislib::Listenable<ControlChannelServer>::Listener {
        public:

            /** Ctor */
            Listener(void) {
            }

            /** Dtor */
            virtual ~Listener(void) {
            }

            /**
             * Informs that the control channel is now connected an can send and receive messages
             *
             * @param sender The sending object
             */
            virtual void OnControlChannelServerStarted(ControlChannelServer& server) {
            }

            /**
             * Informs that the control channel is no longer connected.
             *
             * @param sender The sending object
             */
            virtual void OnControlChannelServerStopped(ControlChannelServer& server) {
            }

            /**
             * Informs that the control channel is now connected an can send and receive messages
             *
             * @param sender The sending object
             * @param channel The control channel
             */
            virtual void OnControlChannelConnect(ControlChannelServer& server, ControlChannel& channel) {
            }

            /**
             * Informs that the control channel is no longer connected.
             *
             * @param sender The sending object
             * @param channel The control channel
             */
            virtual void OnControlChannelDisconnect(ControlChannelServer& server, ControlChannel& channel) {
            }

            /**
             * A message has been received over the control channel.
             *
             * @param sender The sending object
             * @param channel The control channel
             * @param msg The received message
             */
            virtual void OnControlChannelMessage(ControlChannelServer& server, ControlChannel& channel,
                    const vislib::net::AbstractSimpleMessage& msg) {
            }

        };

        /**
         * Ctor
         */
        ControlChannelServer(void);

        /**
         * Dtor.
         */
        virtual ~ControlChannelServer(void);

        /**
         * Answer wether the server is running
         *
         * @return true if the server is running
         */
        bool IsRunning(void) const;

        /**
         * Starts the server on the specified local ip end point
         *
         * @param ep The local ip end point to start the server on
         */
        void Start(vislib::net::IPEndPoint& ep);

        /**
         * Stops the server
         */
        void Stop(void);

        /**
         * Sends a message to all nodes in the cluster.
         *
         * @param msg The message to be send
         */
        void MultiSendMessage(const vislib::net::AbstractSimpleMessage& msg);

    protected:

        /**
         * Informs that the control channel is no longer connected.
         *
         * @param sender The sending object
         */
        virtual void OnControlChannelDisconnect(cluster::ControlChannel& sender);

        /**
         * A message has been received over the control channel.
         *
         * @param sender The sending object
         * @param msg The received message
         */
        virtual void OnControlChannelMessage(cluster::ControlChannel& sender, const vislib::net::AbstractSimpleMessage& msg);

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
        virtual bool OnServerError(const vislib::net::CommServer& src, const vislib::Exception& exception) throw();

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
        virtual bool OnNewConnection(const vislib::net::CommServer& src, vislib::SmartRef<vislib::net::AbstractCommChannel> channel) throw();

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

        /** The lock object to access the clients list */
        vislib::sys::CriticalSection clientsLock;

        /** The list of active control channels */
        vislib::SingleLinkedList<ControlChannel> clients;

        /** The comm channel to use */
        vislib::SmartRef<vislib::net::TcpCommChannel> commChannel;

        /** The server accepting incoming connections */
        vislib::sys::RunnableThread<vislib::net::CommServer> server;

    };


} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CONTROLCHANNELSERVER_H_INCLUDED */
