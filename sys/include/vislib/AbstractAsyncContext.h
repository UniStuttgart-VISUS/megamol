/*
 * AbstractAsyncContext.h
 *
 * Copyright (C) 2006 - 2009 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2009 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTASYNCCONTEXT_H_INCLUDED
#define VISLIB_ABSTRACTASYNCCONTEXT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#ifdef _WIN32
#include <windows.h>
#endif /* _WIN32 */


namespace vislib {
namespace sys {


    /**
     * This is the super-class for context data that are required for 
     * asynchrounous system operations like socket calls.
     *
     * The AbstractAsyncContext is not thread-safe. It must not be modified
     * after an asynchronous operation has been started, i. e. the callback
     * and the user context must not be changed.
     */
    class AbstractAsyncContext {

    public:

        /**
         * This function pointer defines the callback that is called once
         * the operation was completed.
         */
        typedef void (* AsyncCallback)(AbstractAsyncContext *context);

        /** Dtor. */
        virtual ~AbstractAsyncContext(void);

        /**
         * Answer the user data pointer stored in the context.
         *
         * @return The user data pointer.
         */
        inline void *GetUserContext(void) {
            return this->userContext;
        }

        /**
         * Answer the user data pointer stored in the context.
         *
         * @return The user data pointer.
         */
        inline const void *GetUserContext(void) const {
            return this->userContext;
        }

        /**
         * Reset the state of the context. If a context object is reused for
         * more than one asynchronous operation, Reset() must be called before
         * every operation.
         *
         * Reset() must not be called between the begin and end of an 
         * asynchronous operation. The results of the call are undefined in this
         * case.
         */
        virtual void Reset(void);

        /**
         * Set a new callback function that is called once the operation 
         * completes.
         *
         * It is safe to set a NULL pointer. In this case, no notification call 
         * is made.
         *
         * This method must not be called between the begin and end of an
         * asynchronous operation. The results of the call are undefined in this
         * case.
         *
         * @param callback The new callback function.
         */
        inline void SetCallback(AsyncCallback callback) {
            this->callback = callback;
        }

        /**
         * Set a new user data pointer. The caller remains owner of the memory
         * designated by 'userContext' and must ensure that the data exists as
         * long as the AbstractAsyncContext exists.
         *
         * This method must not be called between the begin and end of an
         * asynchronous operation. The results of the call are undefined in this
         * case.
         *
         * @param userData The new user data pointer.
         */
        inline void SetUserContext(void *userContext) {
            this->userContext = userContext;
        }

        /**
         * Wait for the operation associated with this context to complete.
         *
         * @throws SystemException If the operation failed.
         */
        virtual void Wait(void) = 0;

    protected:

        /** 
         * Create a new context.
         *
         * @param callback    The callback function to be called once the 
         *                    operation completed.
         * @param userContext An arbitrary user context pointer.
         */
        AbstractAsyncContext(AsyncCallback callback, void *userContext);

        /**
         * Copy ctor.
         *
         * @param rhs The object to be cloned.
         */
        inline AbstractAsyncContext(const AbstractAsyncContext& rhs) {
            this->Reset();
            *this = rhs;
        }

        /**
         * Call the callback, if set.
         *
         * @return true if a callback was set and called, false otherwise.
         */
        virtual bool notifyCompleted(void);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        AbstractAsyncContext& operator =(const AbstractAsyncContext& rhs);

#ifdef _WIN32
        /**
         * Provide access to the OVERLAPPED structure of the operating system 
         * that is associated with this request. 
         *
         * This OVERLAPPED structure is used to initiate the asynchronous
         * operations on Windows platforms.
         *
         * @return A pointer to the OVERLAPPED structure.
         */
        operator OVERLAPPED *(void) {
            return &this->overlapped;
        }
#endif /* _WIN32 */

        /** The callback function to be called once the operation completed. */
        AsyncCallback callback;

    private:

#ifdef _WIN32
        /** The OVERLAPPED structure that is used by Win32 API functions. */
        OVERLAPPED overlapped;
#endif /* _WIN32 */

        /**
         * A user-defined pointer to any context that might be required for
         * completing an asynchronous operation.
         */
        void *userContext;
    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTASYNCCONTEXT_H_INCLUDED */
