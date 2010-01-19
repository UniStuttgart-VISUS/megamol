/*
 * RenderMaster.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_RENDERMASTER_H_INCLUDED
#define MEGAMOLCORE_RENDERMASTER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "job/AbstractJobThread.h"
#include "Module.h"
#include "ViewInstance.h"
#include "param/ParamSlot.h"
#include "special/RenderNetMsg.h"
//#include "vislib/assert.h"
#include "vislib/CriticalSection.h"
#include "vislib/Runnable.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/SmartPtr.h"
#include "vislib/String.h"
#include "vislib/TcpServer.h"
#include "vislib/Thread.h"


namespace megamol {
namespace core {
namespace special {


    /**
     * Class implementing the cluster rendering master job
     */
    class RenderMaster : public job::AbstractJobThread,
        public Module, vislib::net::TcpServer::Listener {
    public:

        /**
         * Class handling one connection per object
         */
        class Connection {
        public:

            /**
             * Ctor.
             */
            Connection(void);

            /**
             * Copy Ctor.
             *
             * @param src The object to clone from.
             */
            Connection(const Connection& src);

            /**
             * Ctor.
             *
             * This is the only ctor which creates a receiver thread.
             *
             * @param owner The owning object
             * @param name The name of the connected client
             * @param socket The socket to use
             */
            Connection(RenderMaster *owner, const vislib::StringA& name,
                const vislib::net::Socket& socket);

            /**
             * Dtor.
             */
            ~Connection(void);

            /**
             * Closes the connection
             *
             * @param join Waits for the receiver thread to join
             */
            void Close(bool join);

            /**
             * Sends a message through this connection.
             *
             * @param msg The message to be sent
             */
            void Send(const RenderNetMsg& msg);

            /**
             * Assignment operator
             *
             * @param rhs The right hand side operand
             *
             * @return reference to 'this'
             */
            Connection& operator=(const Connection& rhs);

            /**
             * Test for equality
             *
             * @param rhs The right hand side operand
             *
             * @return 'true' if 'this' and 'rhs' are equal
             */
            bool operator==(const Connection& rhs) const;

        private:

            /**
             * The receiver thread loop
             *
             * @param userData Pointer to a Connection object holding all
             *                 relevant information. This object is owned by
             *                 the thread itself.
             *
             * @return 0 (not used)
             */
            static DWORD receive(void *userData);

            /** The name of the connected client */
            vislib::StringA name;

            /** The owning object */
            RenderMaster *owner;

            /** The receiver thread */
            vislib::sys::Thread *receiver;

            /** The socket to use */
            vislib::net::Socket socket;

        };

        /** Allow connections to transfer data */
        friend class Connection;

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "RenderMaster";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "The cluster rendering master thread";
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
         * Ctor
         */
        RenderMaster();

        /**
         * Dtor
         */
        virtual ~RenderMaster();

    private:

        /**
         * The tcp server will call this method when a new client connected.
         * This method accepts the connection under reserve, since the
         * handshake protocol has yet to be done.
         *
         * @param socket The socket of the new connection.
         * @param addr   The address of the peer node that 'socket' is
         *               connected to.
         * 
         * @return true
         */
        virtual bool OnNewConnection(vislib::net::Socket& socket,
            const vislib::net::IPEndPoint& addr) throw();

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
         * Perform the work of a thread.
         *
         * @param userData A pointer to user data that are passed to the thread,
         *                 if it started.
         *
         * @return The application dependent return code of the thread. This 
         *         must not be STILL_ACTIVE (259).
         */
        virtual DWORD Run(void *userData);

        /**
         * Closes all connections.
         */
        void closeAllConnections(void);

        /**
         * Handles an incoming network message
         *
         * @param con The calling connection
         * @param msg The incoming message
         *
         * @return 'true' if the message was handled, 'false' if not.
         */
        bool HandleMessage(Connection& con, RenderNetMsg& msg);

        /**
         * Creates graph setup data for 'masterView' and stories it in 
         * 'outBuf'.
         *
         * @param outBuf The buffer to receive the data
         */
        void makeGraphSetupData(vislib::RawStorage& outBuf);

        /** The network adapter to run the server on */
        param::ParamSlot serverAdapSlot;

        /** The network port to run the server on */
        param::ParamSlot serverPortSlot;

        /** Flag indicating if the server is running or not */
        param::ParamSlot serverUpSlot;

        /** The name of the view instance to be used as master */
        param::ParamSlot masterViewNameSlot;

        /** The list of active connections */
        vislib::SingleLinkedList<Connection> connections;

        /** The synchronization object for the connection list */
        vislib::sys::CriticalSection connLock;

        /** The view instance to be synchronized */
        ViewInstance *masterView;

    };


} /* end namespace special */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_RENDERMASTER_H_INCLUDED */
