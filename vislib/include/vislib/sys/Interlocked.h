/*
 * Interlocked.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_INTERLOCKED_H_INCLUDED
#define VISLIB_INTERLOCKED_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#ifdef _WIN32
#include <windows.h>
#else /* _WIN32 */
//#include <asm/atomic.h>
#endif /* _WIN32 */

#include "vislib/assert.h"
#include "vislib/forceinline.h"
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
     *
     * x64: The Windows version of 64-bit Interlocked functions required Windows
     * XP SP2 or above (_WIN32_WINNT >= 0x0502). On Linux, there is currently no
     * 64-bit support.
     */
    class Interlocked {

    public:

        /** Dtor. */
        ~Interlocked(void);

        /**
         * Perform an atomic comparison of the specified 32-bit values and 
         * exchange them, based on the outcome of the comparison. If the 
         * variable at 'address' and 'comparand' are equal, the variable at
         * 'address' is exchanged with 'exchange'. Otherwise, nothing is done.
         *
         * @param address   Pointer to the destination value. The sign is 
         *                  ignored.
         * @param exchange  Exchange value. The sign is ignored. 
         * @param comparand Value to compare to 'address'. The sign is ignored.
         *
         * @return The initial value of the variable designated by 'address'.
         */
        VISLIB_FORCEINLINE static INT32 CompareExchange(volatile INT32 *address,
                const INT32 exchange, const INT32 comparand) {
#ifdef _WIN32
            ASSERT(sizeof(INT32) == sizeof(LONG));
            return ::InterlockedCompareExchange(
                reinterpret_cast<volatile LONG *>(address), 
                static_cast<LONG>(exchange),
                static_cast<LONG>(comparand));
#else /* _WIN32 */
            INT32 retval;
            __asm__ __volatile__ ("lock; cmpxchgl %2, %0"
                : "=m" (*address), "=a" (retval)
                : "r" (exchange), "m" (*address), "a" (comparand)
                : "memory");
            return retval;
#endif /* _WIN32 */
        }

        /**
         * Perform an atomic comparison of the specified 64-bit values and 
         * exchange them, based on the outcome of the comparison. If the 
         * variable at 'address' and 'comparand' are equal, the variable at
         * 'address' is exchanged with 'exchange'. Otherwise, nothing is done.
         *
         * @param address   Pointer to the destination value. The sign is 
         *                  ignored.
         * @param exchange  Exchange value. The sign is ignored. 
         * @param comparand Value to compare to 'address'. The sign is ignored.
         *
         * @return The initial value of the variable designated by 'address'.
         *
         * @throws UnsupportedOperationException If the platform does not 
         *                                       support 64 bit interlocked
         *                                       operations.
         */
        VISLIB_FORCEINLINE static INT64 CompareExchange(volatile INT64 *address,
                const INT64 exchange, const INT64 comparand) {
#ifdef _WIN32
            ASSERT(sizeof(INT64) == sizeof(LONGLONG));
#if (defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0502))
            return ::InterlockedCompareExchange64(
                reinterpret_cast<volatile LONGLONG *>(address),
                static_cast<LONGLONG>(exchange),
                static_cast<LONGLONG>(comparand));
#else /* (defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0502)) */
            throw UnsupportedOperationException("Interlocked::CompareExchange",
                __FILE__, __LINE__);
#endif /* (defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0502)) */
#else /* _WIN32 */
            throw UnsupportedOperationException("Interlocked::CompareExchange",
                __FILE__, __LINE__);
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
        VISLIB_FORCEINLINE static INT32 Decrement(volatile INT32 *address) {
#ifdef _WIN32
            ASSERT(sizeof(INT32) == sizeof(LONG));
            return ::InterlockedDecrement(
                reinterpret_cast<volatile LONG *>(address));
#else /* _WIN32 */
            return (Interlocked::ExchangeAdd(address, -1) - 1);
#endif /* _WIN32 */
        }

        /**
         * Decrement the value of the specified 64-bit variable designated by 
         * and return the resulting value.
         *
         * @param address Pointer to the variable to be decremented.
         *
         * @return The new value of the variable.
         *
         * @throws UnsupportedOperationException If the platform does not 
         *                                       support 64 bit interlocked
         *                                       operations.
         */
        VISLIB_FORCEINLINE static INT64 Decrement(volatile INT64 *address) {
            ASSERT(sizeof(INT64) == sizeof(LONGLONG));
#ifdef _WIN32
#if (defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0502))
            return ::InterlockedDecrement64(
                reinterpret_cast<volatile LONGLONG *>(address));
#else /* (defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0502)) */
            throw UnsupportedOperationException("Interlocked::Decrement",
                __FILE__, __LINE__);
#endif /* (defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0502)) */
#else /* _WIN32 */
            throw UnsupportedOperationException("Interlocked::Decrement",
                __FILE__, __LINE__);
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
        VISLIB_FORCEINLINE static INT32 Exchange(volatile INT32 *address,
                const INT32 value) {
#ifdef _WIN32
            ASSERT(sizeof(INT32) == sizeof(LONG));
            return ::InterlockedExchange(
                reinterpret_cast<volatile LONG *>(address),
                static_cast<INT32>(value));
#else /* _WIN32 */
            INT32 old;
            do {
                old = *address;
            } while (Interlocked::CompareExchange(address, value, old) != old);
            return old;
#endif /* _WIN32 */
        }

        /**
         * Atomically exchange a pair of 64-bit values.
         *
         * @param address Address of the target variable.
         * @param value   The new value of the variable.
         *
         * @return The old value of the variable designated by 'address'.
         *
         * @throws UnsupportedOperationException If the platform does not 
         *                                       support 64 bit interlocked
         *                                       operations.
         */
        VISLIB_FORCEINLINE static INT64 Exchange(volatile INT64 *address,
                const INT64 value) {
#ifdef _WIN32
            ASSERT(sizeof(INT64) == sizeof(LONGLONG));
#if (defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0502))
            return ::InterlockedExchange64(
                reinterpret_cast<volatile LONGLONG *>(address),
                static_cast<LONGLONG>(value));
#else /* (defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0502)) */
            throw UnsupportedOperationException("Interlocked::Exchange",
                __FILE__, __LINE__);
#endif /* (defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0502)) */
#else /* _WIN32 */
            INT64 old;
            do {
                old = *address;
            } while (Interlocked::CompareExchange(address, value, old) != old);
            return old;
#endif /* _WIN32 */
        }

        /**
         * Atomically exchange a pair of pointers.
         *
         * @param address Address of the target variable.
         * @param value   The new value of the variable.
         *
         * @return The old value of the variable designated by 'address'.
         */
        VISLIB_FORCEINLINE static void *Exchange(void **address,
                const void *value) {
#ifdef _WIN32
#pragma warning(disable: 4311)
#pragma warning(disable: 4312)
            return InterlockedExchangePointer(address, 
                const_cast<void *>(value));
#pragma warning(default: 4312)
#pragma warning(default: 4311)
#else /* _WIN32 */
#if ((defined(__LP64__) || defined(_LP64) || defined(__x86_64__)) \
&& ((__LP64__ != 0) || (_LP64 != 0) || (__x86_64__ != 0)))
            return reinterpret_cast<void *>(Interlocked::Exchange(
                reinterpret_cast<volatile INT64 *>(address), 
                reinterpret_cast<INT64>(value)));
#else /* #if ((defined(__LP64__) || defined(_LP64) || ... */
            return reinterpret_cast<void *>(Interlocked::Exchange(
                reinterpret_cast<volatile INT32 *>(address), 
                reinterpret_cast<INT32>(value)));
#endif /* #if ((defined(__LP64__) || defined(_LP64) || ... */
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
        VISLIB_FORCEINLINE static INT32 ExchangeAdd(volatile INT32 *address, 
                const INT32 value) {
#ifdef _WIN32
            ASSERT(sizeof(INT32) == sizeof(LONG));
            return ::InterlockedExchangeAdd(
                reinterpret_cast<volatile LONG *>(address),
                static_cast<LONG>(value));
#else /* _WIN32 */
            INT32 retval;
            __asm__ __volatile__("lock; xaddl %0, %1"
                : "=r" (retval), "=m" (*address)
                : "0" (value), "m" (*address)
                : "memory");
            return retval;
#endif /* _WIN32 */
        }

        /**
         * Perform an atomic addition of a 64-bit increment value to a 64-bit 
         * addend variable. 
         *
         * @param address Address of the addend variable.
         * @param value   Value to be added to the variable at 'address'.
         *
         * @return Value of variable designated by 'address' prior to the 
         *         operation.
         *
         * @throws UnsupportedOperationException If the platform does not 
         *                                       support 64 bit interlocked
         *                                       operations.
         */
        VISLIB_FORCEINLINE static INT64 ExchangeAdd(volatile INT64 *address,
                const INT64 value) {
#ifdef _WIN32
            ASSERT(sizeof(INT64) == sizeof(LONGLONG));
#if (defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0502))
            return ::InterlockedExchangeAdd64(
                reinterpret_cast<volatile LONGLONG *>(address),
                static_cast<LONGLONG>(value));
#else /* (defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0502)) */
            throw UnsupportedOperationException("Interlocked::ExchangeAdd",
                __FILE__, __LINE__);
#endif /* (defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0502)) */
#else /* _WIN32 */
            throw UnsupportedOperationException("Interlocked::ExchangeAdd",
                __FILE__, __LINE__);
#endif /* _WIN32 */
        }

        /**
         * Perform an atomic subtraction of a 32-bit decrement value to a 32-bit
         * variable. 
         *
         * @param address Address of the variable to modify.
         * @param value   Value to be subtracted from the variable at 'address'.
         *
         * @return Value of variable designated by 'address' prior to the 
         *         operation.
         */
        VISLIB_FORCEINLINE static INT32 ExchangeSub(volatile INT32 *address, 
                const INT32 value) {
#ifdef _WIN32
            ASSERT(sizeof(INT32) == sizeof(LONG));
            return ::InterlockedExchangeAdd(
                reinterpret_cast<volatile LONG *>(address), 
                static_cast<LONG>(-value));
#else /* _WIN32 */
            return Interlocked::ExchangeAdd(address, -value);
#endif /* _WIN32 */
        }

        /**
         * Perform an atomic subtraction of a 64-bit decrement value to a 64-bit
         * variable. 
         *
         * @param address Address of the variable to modify.
         * @param value   Value to be subtracted from the variable at 'address'.
         *
         * @return Value of variable designated by 'address' prior to the 
         *         operation.
         *
         * @throws UnsupportedOperationException If the platform does not 
         *                                       support 64 bit interlocked
         *                                       operations.
         */
        VISLIB_FORCEINLINE static INT64 ExchangeSub(volatile INT64 *address,
                const INT64 value) {
#ifdef _WIN32
            return Interlocked::ExchangeAdd(address, -value);
#else /* _WIN32 */
            return Interlocked::ExchangeAdd(address, -value);
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
        VISLIB_FORCEINLINE static INT32 Increment(volatile INT32 *address) {
#ifdef _WIN32
            ASSERT(sizeof(INT32) == sizeof(LONG));
            return ::InterlockedIncrement(
                reinterpret_cast<volatile LONG *>(address));
#else /* _WIN32 */
            return (Interlocked::ExchangeAdd(address, 1) + 1);
#endif /* _WIN32 */
        }

        /**
         * Increment the value of the specified 64-bit variable designated by 
         * and return the resulting value.
         *
         * @param address Pointer to the variable to be incremented.
         *
         * @return The new value of the variable.
         *
         * @throws UnsupportedOperationException If the platform does not 
         *                                       support 64 bit interlocked
         *                                       operations.
         */
        VISLIB_FORCEINLINE static INT64 Increment(volatile INT64 *address) {
            ASSERT(sizeof(INT64) == sizeof(LONGLONG));
#ifdef _WIN32
#if (defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0502))
            return ::InterlockedIncrement64(
                reinterpret_cast<volatile LONGLONG *>(address));
#else /* (defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0502)) */
            throw UnsupportedOperationException("Interlocked::Increment",
                __FILE__, __LINE__);
#endif /* (defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0502)) */
#else /* _WIN32 */
            return (Interlocked::ExchangeAdd(address, 1) + 1);
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
