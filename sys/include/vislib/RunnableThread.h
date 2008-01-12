/*
 * RunnableThread.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_RUNNABLETHREAD_H_INCLUDED
#define VISLIB_RUNNABLETHREAD_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
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
    template<class T> class RunnableThread : public Thread {

    public:

        /** Ctor. */
        RunnableThread(void);

        /** Dtor. */
        virtual ~RunnableThread(void);

        /**
         * Answer the Runnable instance the thread is running.
         *
         * @return The runnable the thread is running.
         */
        inline T& GetRunnableInstance(void) {
            return this->runnableInstance;
        }

        /**
         * Answer the Runnable instance the thread is running.
         *
         * @return The runnable the thread is running.
         */
        inline const T& GetRunnableInstance(void) const {
            return this->runnableInstance;
        }

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

        /** 
         * The instance of the runnable to start. The 'runnable' member in the
         * superclass will point to this variable.
         */
        T runnableInstance;

    };


    /*
     * vislib::sys::RunnableThread<T>::RunnableThread
     */
    template<class T> RunnableThread<T>::RunnableThread(void) 
#ifdef _WIN32
#pragma warning(disable: 4355)
#endif /* _WIN32 */
            : Thread(&this->runnableInstance) {
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
    RunnableThread<T>::RunnableThread(const RunnableThread& rhs) : Thread(rhs) {
    }


    /*
     * vislib::sys::RunnableThread<T>::operator =
     */
    template<class T> RunnableThread<T>& RunnableThread<T>::operator =(
            const RunnableThread& rhs) {
        Thread::operator =(rhs);
        return *this;
    }

} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_RUNNABLETHREAD_H_INCLUDED */

