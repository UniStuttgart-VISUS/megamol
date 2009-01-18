/*
 * AsyncSocketContext.h
 *
 * Copyright (C) 2006 - 2009 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2009 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ASYNCSOCKETCONTEXT_H_INCLUDED
#define VISLIB_ASYNCSOCKETCONTEXT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#ifdef _WIN32
#include <winsock2.h>
#endif /* _WIN32 */

#include "vislib/AbstractAsyncContext.h"
#include "vislib/assert.h"
#include "vislib/Event.h"
#include "vislib/IPEndPoint.h"
#include "vislib/Socket.h"


namespace vislib {
namespace net {

    /* Forward declarations. */
    class AsyncSocket;


    /**
     * This class is used for handling asynchronous socket operations.
     */
    class AsyncSocketContext : public vislib::sys::AbstractAsyncContext {

    public:

        /**
         * This function pointer defines the callback that is called once
         * the operation was completed.
         */
        typedef vislib::sys::AbstractAsyncContext::AsyncCallback AsyncCallback;

        /** Ctor. */
        AsyncSocketContext(AsyncCallback callback = NULL, 
            void *userContext = NULL);

        /**
         * Copy ctor.
         *
         * @param rhs The object to be cloned.
         */
        inline AsyncSocketContext(const AsyncSocketContext& rhs) : Super(rhs) {}

        /** Dtor. */
        virtual ~AsyncSocketContext(void);

        /**
         * Wait for the operation associated with this context to complete.
         *
         * @throws SystemException If the operation failed.
         */
        virtual void Wait(void);

    protected:

        /** Superclass typedef. */
        typedef vislib::sys::AbstractAsyncContext Super;

#ifdef _WIN32
        /**
         * Provide access to the WSAOVERLAPPED structure of the operating system
         * that is associated with this request. 
         *
         * This WSAOVERLAPPED structure is used to initiate the asynchronous
         * Winsock operation.
         *
         * @return A pointer to the WSAOVERLAPPED structure.
         */
        operator WSAOVERLAPPED *(void) {
            ASSERT(sizeof(WSAOVERLAPPED) == sizeof(OVERLAPPED));
            return (Super::operator OVERLAPPED *());
        }
#endif /* _WIN32 */

        virtual void notifyCompleted(const DWORD cntData, 
            const DWORD errorCode);

        inline void setDgramParams(AsyncSocket *socket, 
                const IPEndPoint *dgramAddr, const void *data, 
                const SIZE_T cntData, const INT flags, const INT timeout) {
            this->socket = socket;
            this->dgramAddr = const_cast<IPEndPoint *>(dgramAddr);
            this->data = const_cast<void *>(data);
            this->cntData = cntData;
            this->flags = flags;
            this->timeout = timeout;
        }

        inline void setStreamParams(AsyncSocket *socket, const void *data, 
                const SIZE_T cntData, const INT flags, const INT timeout) {
            this->socket = socket;
            this->dgramAddr = NULL;
            this->data = const_cast<void *>(data);
            this->cntData = cntData;
            this->flags = flags;
            this->timeout = timeout;
        }

    private:

        DWORD errorCode;

        SIZE_T cntData;
               
        void *data; 

        IPEndPoint *dgramAddr;

        INT flags;

        INT timeout;

        AsyncSocket *socket;


        /** Allow access to protected cast operation to the socket. */
        friend class AsyncSocket;
    };
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ASYNCSOCKETCONTEXT_H_INCLUDED */
