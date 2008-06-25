/*
 * RunnableThread.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_RUNNABLETHREAD_H_INCLUDED
#define VISLIB_RUNNABLETHREAD_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Runnable.h"
#include "vislib/Thread.h"


namespace vislib {
namespace sys {


    /**
     * This template is for faciliating thread usage. It provides a thread class
     * that holds its own runnable provided that the runnable instance has a
     * default ctor. This combination of Runnable and Thread implies that the
     * created thread instance cannot be used for any other Runnable than the 
     * one instance it created itself.
     *
     * The template parameter T must be a Runnable-derived class with a default
     * constructor.
     */
    template<class T> class RunnableThread : public Thread, public T {

    public:

        /** Ctor. */
        RunnableThread(void);

        /** Dtor. */
        virtual ~RunnableThread(void);

        /**
         * Ask the runnable to abort as soon as possible.
         *
         * @return true to acknowledge that the Runnable will finish as soon
         *         as possible, false if termination is not possible.
         */
        virtual bool Terminate(void);

        /**
         * Terminate the thread. See documentation of Thread::Terminate
         *
         * @param forceTerminate If true, the thread is terminated immediately,
         *                       if false, the thread has the possibility to do
         *                       some cleanup and finish in a controllend 
         *                       manner. 'forceTerminate' must be true, if the
         *                       thread has been constructed using a 
         *                       RunnableFunc.
         * @param exitCode       If 'forceTerminate' is true, this value will be
         *                       used as exit code of the thread. If 
         *                       'forceTerminate' is false, this value will be
         *                       ignored.
         * 
         * @returns true, if the thread has been terminated, false, otherwise.
         *
         * @throws IllegalStateException If 'forceTerminate' is false and the
         *                               thread has been constructed using a 
         *                               RunnableFunc.
         * @throws SystemException       If terminating the thread forcefully
         *                               failed.
         */
        bool Terminate(const bool forceTerminate, const int exitCode = 0);

    private:

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        RunnableThread(const RunnableThread& rhs);

        /**
         * Forbidden assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         *
         * @throws IllegalParamException If (this != &rhs).
         */
        RunnableThread& operator =(const RunnableThread& rhs);

    };


    /*
     * vislib::sys::RunnableThread<T>::RunnableThread
     */
    template<class T> RunnableThread<T>::RunnableThread(void) 
#ifdef _WIN32
#pragma warning(disable: 4355)
#endif /* _WIN32 */
            : Thread(this) {
#ifdef _WIN32
#pragma warning(default: 4355)
#endif /* _WIN32 */
    }


    /*
     * vislib::sys::RunnableThread<T>::~RunnableThread
     */
    template<class T> RunnableThread<T>::~RunnableThread(void) {
    }


    /*
     * vislib::sys::RunnableThread<T>::RunnableThread
     */
    template<class T> 
    RunnableThread<T>::RunnableThread(const RunnableThread& rhs) 
            : Thread(rhs), T(rhs) {
    }

    /*
     * RunnableThread<T>::Terminate
     */
    template<class T> 
    bool RunnableThread<T>::Terminate(void) {
        return T::Terminate();
    }


    /*
     * RunnableThread<T>::Terminate
     */
    template<class T> 
    bool RunnableThread<T>::Terminate(const bool forceTerminate, 
            const int exitCode) {
        return Thread::Terminate(forceTerminate, exitCode);
    }


    /*
     * vislib::sys::RunnableThread<T>::operator =
     */
    template<class T> RunnableThread<T>& RunnableThread<T>::operator =(
            const RunnableThread& rhs) {
        Thread::operator =(rhs);
        T::operator =(rhs);
        return *this;
    }

} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_RUNNABLETHREAD_H_INCLUDED */

