/*
 * VersionNumber.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_VERSIONNUMBER_H_INCLUDED
#define VISLIB_VERSIONNUMBER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/String.h"


namespace vislib {


    /**
     * This class represets a version number containing four unsigned integer,
     * representing the major version number, minor version number, build 
     * number and revision number.
     */
    class VersionNumber {

    public:
        
        /** type for version number value */
        typedef unsigned short VersionInt;

        /**
         * Ctor.
         *
         * @param majorNumber The major version number.
         * @param minorNumber The minor version number.
         * @param buildNumber The build number.
         * @param revisionNumber The revision number.
         */
        VersionNumber(VersionInt majorNumber = 0, VersionInt minorNumber = 0, 
            VersionInt buildNumber = 0, VersionInt revisionNumber = 0);

        /** 
         * Copy ctor. 
         *
         * @param rhs The source object to copy from
         */
        VersionNumber(const VersionNumber& rhs);

        /**
         * Ctor.
         * If the string could not be parsed successfully, the version is set
         * to 0.0.0.0
         *
         * @see vislib::VersionNumber::Parse
         *
         * @param ver The version number as string.
         */
        VersionNumber(const char *ver);

        /**
         * Ctor.
         * If the string could not be parsed successfully, the version is set
         * to 0.0.0.0
         *
         * @see vislib::VersionNumber::Parse
         *
         * @param ver The version number as string.
         */
        VersionNumber(const wchar_t *ver);

        /** Dtor. */
        ~VersionNumber(void);
        
        /**
         * Answer the major version number.
         *
         * @return The major version number.
         */
        inline VersionInt GetMajorVersionNumber(void) const {
            return this->majorNumber;
        }
        
        /**
         * Answer the minor version number.
         *
         * @return The minor version number.
         */
        inline VersionInt GetMinorVersionNumber(void) const {
            return this->minorNumber;
        }
        
        /**
         * Answer the build number.
         *
         * @return The build number.
         */
        inline VersionInt GetBuildNumber(void) const {
            return this->buildNumber;
        }
        
        /**
         * Answer the revision number.
         *
         * @return The revision number.
         */
        inline VersionInt GetRevisionNumber(void) const {
            return this->revisionNumber;
        }

        /**
         * Sets the version number to the value of a string. The string must
         * contain between one and four positive numbers separated by '.' 
         * characters. This string must not contain any white space characters.
         * However, any trailing characters behinde the last number will be
         * ignored. If not all version numbers could be successfully parsed
         * the one not set, will be set to zero.
         *
         * @param verStr The version string.
         *
         * @return The number of version numbers successfully parsed from the
         *         string. Zero means that the string could not be parsed. One
         *         indicates that only the major version number is set. ...
         *         A value of four indicates that all four version numbers has
         *         been successfully parsed.
         */
        unsigned int Parse(const char *verStr);

        /**
         * Sets the version number to the value of a string. The string must
         * contain between one and four positive numbers separated by '.' 
         * characters. This string must not contain any white space characters.
         * However, any trailing characters behinde the last number will be
         * ignored. If not all version numbers could be successfully parsed
         * the one not set, will be set to zero.
         *
         * @param verStr The version string.
         *
         * @return The number of version numbers successfully parsed from the
         *         string. Zero means that the string could not be parsed. One
         *         indicates that only the major version number is set. ...
         *         A value of four indicates that all four version numbers has
         *         been successfully parsed.
         */
        unsigned int Parse(const wchar_t *verStr);

        /**
         * Sets the version numbers.
         *
         * @param majorNumber The major version number.
         * @param minorNumber The minor version number.
         * @param buildNumber The build number.
         * @param revisionNumber The revision number.
         */
        inline void Set(VersionInt majorNumber, VersionInt minorNumber = 0, 
                VersionInt buildNumber = 0, VersionInt revisionNumber = 0) {
            this->majorNumber = majorNumber;
            this->minorNumber = minorNumber;
            this->buildNumber = buildNumber;
            this->revisionNumber = revisionNumber;
        }

        /**
         * Creates and returns a string holding the version numbers. The string
         * is created with the same format used by 'Parse'.
         *
         * @see vislib::VersionNumber::Parse
         *
         * @param num The number of version number to output. A value of one 
         *            will produce a string containing only the major version
         *            number, a value of two a string with the major and minor
         *            version numbers, ... a value of four or above will 
         *            produce a string containing all four version numbers. A
         *            value of Zero will produce a string with at least the
         *            major version and will stop as soon as the remaining
         *            values all are zero.
         *
         * @return The string holding the version number.
         */
        vislib::StringA ToStringA(unsigned int num = 0) const;

        /**
         * Creates and returns a string holding the version numbers. The string
         * is created with the same format used by 'Parse'.
         *
         * @see vislib::VersionNumber::Parse
         *
         * @param num The number of version number to output. A value of one 
         *            will produce a string containing only the major version
         *            number, a value of two a string with the major and minor
         *            version numbers, ... a value of four or above will 
         *            produce a string containing all four version numbers. A
         *            value of Zero will produce a string with at least the
         *            major version and will stop as soon as the remaining
         *            values all are zero.
         *
         * @return The string holding the version number.
         */
        vislib::StringW ToStringW(unsigned int num = 0) const;

        /**
         * assignment operator
         *
         * @param rhs The right hand side operand.
         *
         * @return Reference to this
         */
        VersionNumber& operator=(const VersionNumber& rhs);

        /**
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if all four version numbers of both operands are equal.
         */
        inline bool operator==(const VersionNumber& rhs) const {
            return (this->majorNumber == rhs.majorNumber) 
                && (this->minorNumber == rhs.minorNumber)
                && (this->buildNumber == rhs.buildNumber) 
                && (this->revisionNumber == rhs.revisionNumber);
        }

        /**
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if at least one of the version numbers of the two 
         *         operands is not equal.
         */
        inline bool operator!=(const VersionNumber& rhs) const {
            return !(*this == rhs);
        }

        /**
         * Test if this version number is less than rhs.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if this version numbers is less than rhs.
         */
        inline bool operator<(const VersionNumber& rhs) const {
            return ((this->majorNumber < rhs.majorNumber) 
                || ((this->majorNumber == rhs.majorNumber)
                && ((this->minorNumber < rhs.minorNumber) 
                || ((this->minorNumber == rhs.minorNumber)
                && ((this->buildNumber < rhs.buildNumber) 
                || ((this->buildNumber == rhs.buildNumber)
                && (this->revisionNumber < rhs.revisionNumber)))))));
        }

        /**
         * Test if this version number is less than or equal rhs.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if this version numbers is less than or equal rhs.
         */
        inline bool operator<=(const VersionNumber& rhs) const {
            return (*this == rhs) || (*this < rhs);
        }

        /**
         * Test if this version number is more than rhs.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if this version numbers is more than rhs.
         */
        inline bool operator>(const VersionNumber& rhs) const {
            return (!(*this == rhs)) && (!(*this < rhs));
        }

        /**
         * Test if this version number is more than or equal rhs.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if this version numbers is more than or equal rhs.
         */
        inline bool operator>=(const VersionNumber& rhs) const {
            return !(*this < rhs);
        }

    private:

        /** major version number */
        VersionInt majorNumber;

        /** minor version number */
        VersionInt minorNumber;

        /** build number */
        VersionInt buildNumber;

        /** revision number */
        VersionInt revisionNumber;

    };
    
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_VERSIONNUMBER_H_INCLUDED */
