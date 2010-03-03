/*
 * CommServer.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_COMMSERVER_H_INCLUDED
#define VISLIB_COMMSERVER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractServerEndPoint.h"
#include "vislib/CommServerListener.h"
#include "vislib/CriticalSection.h"
#include "vislib/Runnable.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/StackTrace.h"
#include "vislib/StringConverter.h"


namespace vislib {
namespace net {


    /**
     * This class implements a server for VISlib communication channels.
     */
    class CommServer : public vislib::sys::Runnable {

    public:

        /** Ctor. */
        CommServer(void);

        /** Dtor. */
        virtual ~CommServer(void);

        /**
         * Add a new CommServerListener to be informed about events of this 
         * server.
         *
         * The caller remains owner of the memory designated by 'listener' and
         * must ensure that the object exists as long as the listener is 
         * registered.
         *
         * This method is thread-safe.
         *
         * @param listener The listener to register. This must not be NULL.
         */
        void AddListener(CommServerListener *listener);


        /**
         * Configure the server using the specified end point object and the
         * specified address. 
         *
         * Once the server is started, it will bind the given end point to
         * the given address. The end point passed to the method therefore 
         * should be unused. The format of the address string must be compatible
         * to the format of the addresses used by the given end point.
         *
         * Please note, that the Configure method does not yet bind the end 
         * point, this will be done in the server thread.
         *
         * @param serverEndPoint The end point to be used for the server. The
         *                       CommServer object will increase the reference
         *                       count of this object and release it on its
         *                       own destruction.
         * @param bindAddress    A string representation of the address to bind
         *                       the server to.
         */
        inline void Configure(AbstractServerEndPoint *serverEndPoint,
                const char *bindAddress) {
            VLSTACKTRACE("CommServer::Configure", __FILE__, __LINE__);
            this->Configure(serverEndPoint, A2W(bindAddress));
        }

        /**
         * Configure the server using the specified end point object and the
         * specified address. 
         *
         * Once the server is started, it will bind the given end point to
         * the given address. The end point passed to the method therefore 
         * should be unused. The format of the address string must be compatible
         * to the format of the addresses used by the given end point.
         *
         * Please note, that the Configure method does not yet bind the end 
         * point, this will be done in the server thread.
         *
         * @param serverEndPoint The end point to be used for the server. The
         *                       CommServer object will increase the reference
         *                       count of this object and release it on its
         *                       own destruction.
         * @param bindAddress    A string representation of the address to bind
         *                       the server to.
         */
        inline void Configure(SmartRef<AbstractServerEndPoint>& serverEndPoint,
                const char *bindAddress) {
            VLSTACKTRACE("CommServer::Configure", __FILE__, __LINE__);
            this->Configure(serverEndPoint, A2W(bindAddress));
        }

        /**
         * Configure the server using the specified end point object and the
         * specified address. 
         *
         * Once the server is started, it will bind the given end point to
         * the given address. The end point passed to the method therefore 
         * should be unused. The format of the address string must be compatible
         * to the format of the addresses used by the given end point.
         *
         * Please note, that the Configure method does not yet bind the end 
         * point, this will be done in the server thread.
         *
         * @param serverEndPoint The end point to be used for the server. The
         *                       CommServer object will increase the reference
         *                       count of this object and release it on its
         *                       own destruction.
         * @param bindAddress    A string representation of the address to bind
         *                       the server to.
         */
        void Configure(AbstractServerEndPoint *serverEndPoint, 
            const wchar_t *bindAddress);

        /**
         * Configure the server using the specified end point object and the
         * specified address. 
         *
         * Once the server is started, it will bind the given end point to
         * the given address. The end point passed to the method therefore 
         * should be unused. The format of the address string must be compatible
         * to the format of the addresses used by the given end point.
         *
         * Please note, that the Configure method does not yet bind the end 
         * point, this will be done in the server thread.
         *
         * @param serverEndPoint The end point to be used for the server. The
         *                       CommServer object will increase the reference
         *                       count of this object and release it on its
         *                       own destruction.
         * @param bindAddress    A string representation of the address to bind
         *                       the server to.
         */
        void Configure(SmartRef<AbstractServerEndPoint>& serverEndPoint,
            const wchar_t *bindAddress);


        /**
         * Configure the server using the specified end point object and the
         * specified address. 
         *
         * Once the server is started, it will bind the given end point to
         * the given address. The end point passed to the method therefore 
         * should be unused. The format of the address string must be compatible
         * to the format of the addresses used by the given end point.
         *
         * Please note, that the Configure method does not yet bind the end 
         * point, this will be done in the server thread.
         *
         * @param serverEndPoint The end point to be used for the server. The
         *                       CommServer object will increase the reference
         *                       count of this object and release it on its
         *                       own destruction.
         * @param bindAddress    A string representation of the address to bind
         *                       the server to.
         */
        inline void Configure(AbstractServerEndPoint *serverEndPoint,
                const StringW& bindAddress) {
            VLSTACKTRACE("CommServer::Configure", __FILE__, __LINE__);
            this->Configure(serverEndPoint, bindAddress.PeekBuffer());
        }

        /**
         * Configure the server using the specified end point object and the
         * specified address. 
         *
         * Once the server is started, it will bind the given end point to
         * the given address. The end point passed to the method therefore 
         * should be unused. The format of the address string must be compatible
         * to the format of the addresses used by the given end point.
         *
         * Please note, that the Configure method does not yet bind the end 
         * point, this will be done in the server thread.
         *
         * @param serverEndPoint The end point to be used for the server. The
         *                       CommServer object will increase the reference
         *                       count of this object and release it on its
         *                       own destruction.
         * @param bindAddress    A string representation of the address to bind
         *                       the server to.
         */
        inline void Configure(SmartRef<AbstractServerEndPoint>& serverEndPoint,
                const StringA& bindAddress) {
            VLSTACKTRACE("CommServer::Configure", __FILE__, __LINE__);
            this->Configure(serverEndPoint, bindAddress.PeekBuffer());
        }

        /**
         * Configure the server using the specified end point object and the
         * specified address. 
         *
         * Once the server is started, it will bind the given end point to
         * the given address. The end point passed to the method therefore 
         * should be unused. The format of the address string must be compatible
         * to the format of the addresses used by the given end point.
         *
         * Please note, that the Configure method does not yet bind the end 
         * point, this will be done in the server thread.
         *
         * @param serverEndPoint The end point to be used for the server. The
         *                       CommServer object will increase the reference
         *                       count of this object and release it on its
         *                       own destruction.
         * @param bindAddress    A string representation of the address to bind
         *                       the server to.
         */
        inline void Configure(AbstractServerEndPoint *serverEndPoint,
                const StringA& bindAddress) {
            VLSTACKTRACE("CommServer::Configure", __FILE__, __LINE__);
            this->Configure(serverEndPoint, bindAddress.PeekBuffer());
        }

        /**
         * Configure the server using the specified end point object and the
         * specified address. 
         *
         * Once the server is started, it will bind the given end point to
         * the given address. The end point passed to the method therefore 
         * should be unused. The format of the address string must be compatible
         * to the format of the addresses used by the given end point.
         *
         * Please note, that the Configure method does not yet bind the end 
         * point, this will be done in the server thread.
         *
         * @param serverEndPoint The end point to be used for the server. The
         *                       CommServer object will increase the reference
         *                       count of this object and release it on its
         *                       own destruction.
         * @param bindAddress    A string representation of the address to bind
         *                       the server to.
         */
        inline void Configure(SmartRef<AbstractServerEndPoint>& serverEndPoint,
                const StringW& bindAddress) {
            VLSTACKTRACE("CommServer::Configure", __FILE__, __LINE__);
            this->Configure(serverEndPoint, bindAddress.PeekBuffer());
        }

        /**
         * Answer the address the server will bind to or is bound to.
         *
         * @return The bind address of the server.
         */
        inline StringA GetBindAddressA(void) const {
            return StringA(this->bindAddress);
        }

        /**
         * Answer the address the server will bind to or is bound to.
         *
         * @return The bind address of the server.
         */
        inline const StringW& GetBindAddressW(void) const {
            return this->bindAddress;
        }

        /**
         * Removes, if registered, 'listener' from the list of objects informed
         * about events events.
         * 
         * The caller remains owner of the memory designated by 'listener'.
         *
         * This method is thread-safe.
         *
         * @param listener The listener to be removed. Nothing happens, if the
         *                 listener was not registered.
         */
        void RemoveListener(CommServerListener *listener);

        /**
         * Perform the work of a thread.
         *
         * @param reserved Unused pointer. Should be NULL.
         *
         * @return The application dependent return code of the thread. This 
         *         must not be STILL_ACTIVE (259).
         */
        virtual DWORD Run(void *reserved);

        /**
         * Abort the work of the server by forcefully closing the underlying
         * communication channel.
         *
         * @return true.
         */
        virtual bool Terminate(void);

    private:

        /** A thread-safe list for the message listeners. */
        typedef SingleLinkedList<CommServerListener *, 
            sys::CriticalSection> ListenerList;

        void createNewTcpServerEndPoint(void);

        /**
         * Inform all registered listener about a connection that was 
         * established.
         *
         * This method is thread-safe.
         *
         * @param channel The client communication channel.
         *
         * @return true if the connection was accepted, false otherwise.
         */
        bool fireNewConnection(SmartRef<AbstractCommChannel>& channel);

        /**
         * Inform all registered listener about an exception that was caught.
         *
         * This method is thread-safe.
         *
         * @param exception The exception that was caught.
         *
         * @return Accumulated (ANDed) return values of all listeners.
         */
        bool fireServerError(const vislib::Exception& exception);

        /**
         * Inform all registered listener about that the server is exiting.
         *
         * This method is thread-safe.
         */
        void fireServerExited(void);

        /**
         * Inform all registered listener about that the server is starting.
         *
         * This method is thread-safe.
         */
        void fireServerStarted(void);

        /** The list of listeners. */
        ListenerList listeners;

        /** The address to bind the server to. */
        StringW bindAddress;

        /** The end point the server is using for waiting for clients. */
        SmartRef<AbstractServerEndPoint> serverEndPoint;

    };
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_COMMSERVER_H_INCLUDED */
