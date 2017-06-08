/*
 * GUID.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_GUID_H_INCLUDED
#define VISLIB_GUID_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#ifdef _WIN32
#include <guiddef.h>
#else /* _WIN32 */
#include <uuid/uuid.h>
#endif /* _WIN32 */

#include "vislib/String.h"
#include "vislib/types.h"

#ifdef _MSC_VER
#pragma comment(lib, "rpcrt4")
#endif /* _MSC_VER */


namespace vislib {


    /**
     * This class represents a Globally Unique Identifier (GUID).
     */
    class GUID {

    public:

        /**
         * Create a zero-initialised GUID.
         */
        GUID(void);

#ifdef _WIN32
        /**
         * Create a GUID object from a system GUID structure.
         *
         * @param guid The GUID structure.
         */
        inline GUID(const ::GUID& guid) {
            *this = guid;
        }

#else /* _WIN32 */
        /**
         * Create a GUID object from a system uuid_t structure.
         *
         * @param guid The uuid_t structure.
         */
        inline GUID(const uuid_t& guid) {
            *this = guid;
        }
#endif /* _WIN32 */

        /**
         * Create a GUID from an array of 16 bytes.
         *
         * @param b An array of 16 bytes.
         */
        GUID(const BYTE b[16]);

        /**
         * Create a GUID from 16 bytes.
         *
         * @param b1  Byte # 1.
         * @param b2  Byte # 2.
         * @param b3  Byte # 3.
         * @param b4  Byte # 4.
         * @param b5  Byte # 5.
         * @param b6  Byte # 6.
         * @param b7  Byte # 7.
         * @param b8  Byte # 8.
         * @param b9  Byte # 9.
         * @param b10 Byte # 10.
         * @param b11 Byte # 11.
         * @param b12 Byte # 12.
         * @param b13 Byte # 13.
         * @param b14 Byte # 14.
         * @param b15 Byte # 15.
         * @param b16 Byte # 16.
         */
        GUID(const BYTE b1, const BYTE b2, const BYTE b3, const BYTE b4,
            const BYTE b5, const BYTE b6, const BYTE b7, const BYTE b8,
            const BYTE b9, const BYTE b10, const BYTE b11, const BYTE b12,
            const BYTE b13, const BYTE b14, const BYTE b15, const BYTE b16);

        /**
         * Create a GUID from an integer, two shorts and eight bytes.
         *
         * @param i   The first four bytes.
         * @param s1  Bytes #5 and 6.
         * @param s2  Bytes #7 and 8.
         * @param b9  Byte # 9.
         * @param b10 Byte # 10.
         * @param b11 Byte # 11.
         * @param b12 Byte # 12.
         * @param b13 Byte # 13.
         * @param b14 Byte # 14.
         * @param b15 Byte # 15.
         * @param b16 Byte # 16.
         */
        GUID(const UINT32 i, const UINT16 s1, const UINT16 s2,
            const BYTE b1, const BYTE b2, const BYTE b3, const BYTE b4,
            const BYTE b5, const BYTE b6, const BYTE b7, const BYTE b8);

        /**
         * Create a GUID from an integer, two shorts and an array of eight 
         * bytes.
         *
         * @param i   The first four bytes.
         * @param s1  Bytes #5 and 6.
         * @param s2  Bytes #7 and 8.
         * @param b9  Byte # 9.
         * @param b   The last eight bytes.
         */
        GUID(const UINT32 i, const UINT16 s1, const UINT16 s2, const BYTE b[8]);

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline GUID(const GUID& rhs) {
            *this = rhs;
        }

        /** Dtor. */
        ~GUID(void);

        /**
         * Make this GUID a new one.
         *
         * @param doNotUseMacAddress If this flag is set, an algorithm will be
         *                           used that does not include the MAC address
         *                           in the GUID. This is the default.
         *
         * @return true in case of success, i. e. the GUID was updated;
         *         false otherwise, e. g. if the MAC address should be used, but
         *         the system does not have an ethernet adapter.
         */
        bool Create(const bool doNotUseMacAddress = true);

        /**
         * Answer whether the GUID is completely filled with zeros.
         *
         * @return true if the GUID is zero, false otherwise.
         */
        bool IsZero(void) const;

        /**
         * Parse the GUID from a string representation.
         *
         * @param str The string representation of the GUID.
         *
         * @return true in case of success, false if 'str' could not be parsed 
         *         as GUID.
         */
        bool Parse(const StringA& str);

        /**
         * Parse the GUID from a string representation.
         *
         * @param str The string representation of the GUID.
         *
         * @return true in case of success, false if 'str' could not be parsed 
         *         as GUID.
         */
        bool Parse(const StringW& str);

        /**
         * Answer a hash code of the GUID.
         *
         * @return A hash code of the GUID.
         */
        UINT32 HashCode(void) const;

        /**
         * Set the whole GUID zero.
         */
        void SetZero(void);

        /**
         * Return a string representation of the GUID.
         *
         * @return A string representation of the GUID.
         *
         * @throws std::bad_alloc In case there is insufficient memory.
         */
        StringA ToStringA(void) const;

        /**
         * Return a string representation of the GUID.
         *
         * @return A string representation of the GUID.
         *
         * @throws std::bad_alloc In case there is insufficient memory.
         */
        StringW ToStringW(void) const;

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        GUID& operator =(const GUID& rhs);

#ifdef _WIN32
        /**
         * Assignment from system GUID structure.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        GUID& operator =(const ::GUID& rhs);

        /**
         * Assignment from pointer to system GUID structure.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        GUID& operator =(const ::GUID *rhs);

#else /* _WIN32 */
        /**
         * Assignment from system uuid_t structure.
         *
         * @param guid The uuid_t structure.
         */
        GUID& operator =(const uuid_t& rhs);
#endif /* _WIN32 */

        /**
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if this object and 'rhs' are equal, false otherwise.
         */
        inline bool operator ==(const GUID& rhs) const {
#ifdef _WIN32
            return (::InlineIsEqualGUID(this->guid, rhs.guid) != 0);
#else /* _WIN32 */
            return (::uuid_compare(this->guid, rhs.guid) == 0);
#endif /* _WIN32 */
        }

        /**
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if this object and 'rhs' are not equal, false otherwise.
         */
        inline bool operator !=(const GUID& rhs) const {
            return !(*this == rhs);
        }

    private:

        /** The GUID structure wrapped by this object. */
#ifdef _WIN32
        ::GUID guid;
#else /* _WIN32 */
        uuid_t guid;
#endif /* _WIN32 */

    };

} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_GUID_H_INCLUDED */
