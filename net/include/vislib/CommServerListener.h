/*
 * CommServerListener.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_COMMSERVERLISTENER_H_INCLUDED
#define VISLIB_COMMSERVERLISTENER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/AbstractCommClientChannel.h"


/* Forward declarations. */
namespace vislib {
    class Exception;
}
namespace vislib {
namespace net {
    class CommServer;
}
}


namespace vislib {
namespace net {

    /**
     * This is the listener class for CommServers. Classes interested in 
     * connections to such servers should implements the interface defined
     * by this class.
     */
    class CommServerListener {

    public:

        /** Dtor. */
        virtual ~CommServerListener(void);

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
        virtual bool OnServerError(const CommServer& src,
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
        virtual bool OnNewConnection(const CommServer& src,
            SmartRef<AbstractCommClientChannel> channel) throw() = 0;

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
        virtual void OnServerExited(const CommServer& src) throw();

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
        virtual void OnServerStarted(const CommServer& src) throw();

    protected:

        /** Ctor. */
        CommServerListener(void);

    };
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_COMMSERVERLISTENER_H_INCLUDED */

