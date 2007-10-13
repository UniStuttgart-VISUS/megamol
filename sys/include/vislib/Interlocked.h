/*
 * Interlocked.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_INTERLOCKED_H_INCLUDED
#define VISLIB_INTERLOCKED_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#ifdef _WIN32
#include <windows.h>
#else /* _WIN32 */
//#include <asm/atomic.h>
#endif /* _WIN32 */

#include "vislib/UnsupportedOperationException.h"
#include "vislib/types.h"


namespace vislib {
namespace sys {


    /**
     * This class provides interlocked operations on both, Windows and Linux 
     * platforms.
     *
     * The interlocked functions perform an atomic operations on the destination
     * variables. They provide a simple mechanism for synchronising access to a
     * variable that is shared by multiple threads. The threads of different 
     * processes can use this mechanism if the variable is in shared memory.
     *
     * The parameters for interlocked functions must be aligned on a 32-bit 
     * boundary; otherwise, the functions will behave unpredictably on 
     * multiprocessor x86 systems and any non-x86 systems.
     *
     * Windows: These functions should not be used on memory allocated with the 
     * PAGE_NOCACHE modifier, as this may cause hardware faults on some 
     * processor architectures. To ensure ordering between reads and writer 
     * to PAGE_NOCACHE memory, use explicit memory barriers in your code.
     */
    class Interlocked {

    public:

        /** Dtor. */
        ~Interlocked(void);

        /**
         * Perform an atomic comparison of the specified 32-bit values and 
         * exchange them, based on the outcome of the comparison. If the 
         * variable at 'address' and 'comparand' are equal, the variable at
         * 'address' is exchanged with 'exchange'. Otherwise, nothing is node.
         *
         * @param address   Pointer to the destination value. The sign is 
         *                  ignored.
         * @param exchange  Exchange value. The sign is ignored. 
         * @param comparand Value to compare to 'address'. The sign is ignored.
         *
         * @return The initial value of the variable designated by 'address'.
         */
#ifdef _WIN32
        __forceinline static INT32 CompareExchange(INT32 *address, 
                const INT32 exchange, const INT comparand) {
            return ::InterlockedCompareExchange(
                reinterpret_cast<LONG *>(address), exchange, comparand);
#else /* _WIN32 */
        inline static INT32 CompareExchange(INT32 *address, 
                const INT32 exchange, const INT comparand) {
#ifdef atomic_cmpxchg
            INT32 retval = atomic_cmpxchg(reinterpret_cast<atomic_t *>(address),
                comparand, exchange);
            return (retval == exchange) ? comparand : retval;
#else /* atomic_cmpxchg */
            throw UnsupportedOperationException("TODO: Kernel does not support "
                "atomic_cmpxchg. Consider inline assembler in VISlib", 
                __FILE__, __LINE__);
#endif /* atomic_cmpxchg */
#endif /* _WIN32 */
        }

        /**
         * Decrement the value of the specified 32-bit variable designated by 
         * and return the resulting value.
         *
         * @param address Pointer to the variable to be decremented.
         *
         * @return The new value of the variable.
         */
#ifdef _WIN32
        __forceinline static INT32 Decrement(INT32 *address) {
            return ::InterlockedDecrement(reinterpret_cast<LONG *>(address));
#else /* _WIN32 */
        inline INT32 static Decrement(INT32 *address) {
#ifdef atomic_dec_return
            return atomic_dec_return(reinterpret_cast<atomic_t *>(address));
#else /* atomic_dec_return */
            throw UnsupportedOperationException("TODO: Kernel does not support "
                "atomic_dec_return.", __FILE__, __LINE__);
#endif /* atomic_dec_return */
#endif /* _WIN32 */
        }

        /**
         * Atomically exchange a pair of 32-bit values.
         *
         * @param address Address of the target variable.
         * @param value   The new value of the variable.
         *
         * @return The old value of the variable designated by 'address'.
         */
#ifdef _WIN32
        __forceinline static INT32 Exchange(INT32 *address, const INT32 value) {
            return ::InterlockedExchange(reinterpret_cast<LONG *>(address), 
                value);
#else /* _WIN32 */
        inline static INT32 Exchange(INT32 *address, const INT32 value) {
#ifdef atomic_cmpxchg
            // TODO: This implementation is crazy. Search for a real solution. 
            // The problem is that Linux does not want us to know the old value
            // at 'address' when using the libc xchg function.
            INT32 old, retval;
            
            old = retval = value;
            while (true) {
                old = atomic_cmpxchg(reinterpret_cast<atomic_t *>(address),
                    old, value);
                if (old == value) {
                    return retval;
                } else {
                    retval = old;
                }
            }
#else /* atomic_cmpxchg */
            throw UnsupportedOperationException("TODO: Kernel does not support "
                "atomic_cmpxchg. Consider inline assembler in VISlib", 
                __FILE__, __LINE__);
#endif /* atomic_cmpxchg */
#endif /* _WIN32 */
        }

        /**
         * Perform an atomic addition of a 32-bit increment value to a 32-bit 
         * addend variable. 
         *
         * @param address Address of the addend variable.
         * @param value   Value to be added to the variable at 'address'.
         *
         * @return Value of variable designated by 'address' prior to the 
         *         operation.
         */
#ifdef _WIN32
        __forceinline static INT32 ExchangeAdd(INT32 *address, 
                const INT32 value) {
            return ::InterlockedExchangeAdd(reinterpret_cast<LONG *>(address),
                value);
#else /* _WIN32 */
        inline static INT32 ExchangeAdd(INT32 *address, const INT32 value) {
#ifdef atomic_add_return
            INT32 retval = atomic_add_return(value, 
                reinterpret_cast<atomic_t *>(address));
            return (retval - value);
#else /* atomic_add_return */
            INT32 retval;
            __asm__ __volatile__("lock; xaddl %0, (%1)"
                : "=r" (retval) : "r" (address), "0" (value) : "memory");
            return retval;

            throw UnsupportedOperationException("TODO: Kernel does not support "
                "atomic_add_return.", __FILE__, __LINE__);
#endif /* atomic_add_return */
#endif /* _WIN32 */
        }

        /**
         * Perform an atomic subtraction of a 32-bit increment value to a 32-bit
         * variable. 
         *
         * @param address Address of the variable to modify.
         * @param value   Value to be subtracted from the variable at 'address'.
         *
         * @return Value of variable designated by 'address' prior to the 
         *         operation.
         */
#ifdef _WIN32
        __forceinline static INT32 ExchangeSub(INT32 *address, 
                const INT32 value) {
            return ::InterlockedExchangeAdd(reinterpret_cast<LONG *>(address),
                -value);
#else /* _WIN32 */
        inline static INT32 ExchangeSub(INT32 *address, const INT32 value) {
#ifdef atomic_sub_return
            INT32 retval = atomic_sub_return(value,
                reinterpret_cast<atomic_t *>(address));
            return (retval + value);
#else /* atomic_sub_return */
            throw UnsupportedOperationException("TODO: Kernel does not support "
                "atomic_sub_return.", __FILE__, __LINE__);
#endif /* atomic_sub_return */
#endif /* _WIN32 */
        }

        /**
         * Increment the value of the specified 32-bit variable designated by 
         * and return the resulting value.
         *
         * @param address Pointer to the variable to be incremented.
         *
         * @return The new value of the variable.
         */
#ifdef _WIN32
        __forceinline static INT32 Increment(INT32 *address) {
            return ::InterlockedIncrement(reinterpret_cast<LONG *>(address));
#else /* _WIN32 */
        inline static INT32 Increment(INT32 *address) {
#ifdef atomic_inc_return
            return atomic_inc_return(reinterpret_cast<atomic_t *>(address));
#else /* atomic_inc_return */
            throw UnsupportedOperationException("TODO: Kernel does not support "
                "atomic_inc_return.", __FILE__, __LINE__);
#endif /* atomic_inc_return */
#endif /* _WIN32 */
        }

    private:

        /**
         * Disallow instances. 
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        Interlocked(void);

    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_INTERLOCKED_H_INCLUDED */
