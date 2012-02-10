/*
 * SocketTest.h
 *
 * Copyright (C) 2011 by University of Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MMPROTEINPLUGIN_SOCKETTEST_H_INCLUDED
#define MMPROTEINPLUGIN_SOCKETTEST_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "param/ParamSlot.h"
#include "vislib/CriticalSection.h"
#include "vislib/Socket.h"
#include "vislib/Runnable.h"
#include "vislib/RawStorage.h"
#include "vislib/Array.h"

namespace megamol {
namespace protein {

    /** 
     * TODO
     */

    class SocketTest : public vislib::sys::Runnable
    {
    public:

        /** Ctor */
        SocketTest(void);

        /** Dtor */
        virtual ~SocketTest(void);

        /**
         * Startup callback of the thread. The Thread class will call that 
         * before Run().
         *
         * @param config A pointer to the Configuration, which specifies the
         *               settings of the connector.
         */
        virtual void OnThreadStarting(void *config);

        /**
         * Perform the work of a thread.
         *
         * @param config A pointer to the Configuration, which specifies the
         *               settings of the connector.
         *
         * @return The application dependent return code of the thread. This 
         *         must not be STILL_ACTIVE (259).
         */
        virtual DWORD Run(void *config);

        /**
         * Abort the work of the connector by forcefully closing the underlying
         * communication channel.
         *
         * @return true.
         */
        virtual bool Terminate(void);


        /**
         * Checks whether or not the socket is functional.
         *
         * @return 'true' if valid, 'false' otherwise.
         */
        bool IsSocketFunctional(void) { 
            return this->socketValidity;
        }

        /**
         * Sets the host and the port for the socket connection.
         * This MUST be called before calling Start for the thread, or the socket
         * will never get set up correctly.
         *
         * @param inHost Host name or IP.
         * @param inPort Port number.
         */
        void Initialize( int inPort);

        /**
         * Releases socket resources.
         */
        void release(void);

    private:

        /**
         * Starts the socket connection with the given host and port; performs handshaking.
         * Sets the socket validity flag to true if it succeeds.
         *
         * @param host String representing either name or IP of the machine running MDDriver.
         * @param port Port number to communicate with on the server machine.
         * @return 'true' on success, 'false' otherwise.
         */
        bool startSocket( int port);

        // -------------------- variables --------------------

        /** The socket for MD Driver connection */
        vislib::net::Socket socket;

        /** The socket status */
        bool socketValidity;

        /** The port for the socket connection */
        int port;

    };

} // end namespace protein
} // end namespace megamol

#endif // MMPROTEINPLUGIN_SOCKETTEST_H_INCLUDED