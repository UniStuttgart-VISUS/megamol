/*
 * TcpServer.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_TCPSERVER_H_INCLUDED
#define VISLIB_TCPSERVER_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/Socket.h"          // Must be first.
#include "vislib/SocketAddress.h"
#include "vislib/CriticalSection.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/Runnable.h"


namespace vislib {
namespace net {


    /**
     * This class implements a simple TCP server that informs registered 
     * listener objects about incoming connections.
     *
     * The listeners can take ownership of an incoming connection. Once the 
     * first listener has taken ownership of the connection, other listeners
     * are not informed any more and the server does not modify the client
     * socket in the future. If no listener takes ownership, the server will
     * immediately close the new connection.
     *
     * It is intended to run TcpServer in a separate vislib::sys::Thread.
     */
    class TcpServer : public vislib::sys::Runnable {

    public:

        /**
         * This is the base class for an object that waits for connections that
         * the TcpServer accepts.
         */
        class Listener {

        public:

            /** Dtor. */
            virtual ~Listener(void);

            /**
             * The server will call this method when a new client connected. The
             * listener can decide whether it wants to take ownership of the
             * client socket 'socket' by returning true. If no listener accepts
             * the new client socket, the server will terminate the new 
             * connection.
             *
             * Note that no other listeners will be informed after the first one
             * has accepted the connection by returning true. This first 
             * listener is regarded as new owner of 'socket' by the TcpServer.
             *
             * Subclasses must not throw exceptions within this method.
             *
             * Subclasses should return as soon as possible from this method.
             *
             * @param socket The socket of the new connection.
             * @param addr   The address of the peer node that 'socket' is
             *               connected to.
             * 
             * @return true if the listener takes ownership of 'socket'. The 
             *         server will not use the socket again. If the method 
             *         returns false, the listener should not use the socket, 
             *         because the server remains its owner.
             */
            virtual bool OnNewConnection(Socket& socket, 
                const SocketAddress& addr) throw() = 0;

            /**
             * The server will call this method when it left the server loop and
             * is about to exit.
             *
             * Subclasses must not throw exceptions within this method.
             *
             * Subclasses should return as soon as possible from this method.
             */
            virtual void OnServerStopped(void) throw();

        protected:

            /** Ctor. */
            Listener(void);
        }; /* end class Listener */

        /** Ctor. */
        TcpServer(void);

        /** Dtor. */
        ~TcpServer(void);

        /**
         * Set a new connection listener object.
         *
         * The object does not take ownership of 'listener'. The callee is
         * responsible for ensuring that 'listener' exists as long it is the
         * registered connection listener of the TcpServer.
         * 
         * @param listener Pointer to the new connection listener. It is safe
         *                 to pass a NULL pointer.
         */
        void AddListener(Listener *listener);

        /**
         * Remove 'listener' from the list of objects informed about new 
         * connections. Possible multiple instances of 'listener' registered
         * will all be removed. The TcpServer will therefore not use the object
         * designated by 'listener' after the method returns.
         *
         * @param listener Pointer to the listener to remove. It is safe to
         *                 pass a NULL pointer.
         */
        void RemoveListener(Listener *listener);

        /**
         * Start the server. 'userData' must be a pointer to the server address.
         *
         * @param userData Pointer to a SocketAddress.
         *
         * @return 0 in case of success, an error code otherwise.
         *
         * @throws IllegalParamException if 'userData' is NULL.
         */
        virtual DWORD Run(void *userData);

        /**
         * Start the server.
         *
         * @param serverAddr The address to bind the server to.
         *
         * @return 0 in case of success, an error code otherwise.
         */
        virtual DWORD Run(const SocketAddress& serverAddr);

        /**
         * Terminate the server.
         *
         * @return true if the server acknowledges termination, false otherwise.
         */
        virtual bool Terminate(void);

    private:

        /** List of potential listeners. */
        typedef SingleLinkedList<Listener *> ListenerList;

        bool fireNewConnection(Socket& socket, const SocketAddress& addr);

        void fireServerStopped(void);

        /** The server socket. */
        Socket socket;

        /**
         * This object receives the notifications about new client 
         * connections.
         */
        ListenerList listeners;

        /** Lock for the 'listeners' member. */
        mutable vislib::sys::CriticalSection lock;

    };
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_TCPSERVER_H_INCLUDED */
