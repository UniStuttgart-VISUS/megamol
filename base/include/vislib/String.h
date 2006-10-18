/*
 * String.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 - 2006 by Christoph Mueller. All rights reserved.
 */

#ifndef VISLIB_STRING_H_INCLUDED
#define VISLIB_STRING_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/CharTraits.h"
#include "vislib/memutils.h"
#include "vislib/OutOfRangeException.h"



namespace vislib {

    /**
     * Represents a string. The string class can basically be instantiated for 
     * vislib::CharTraits subclasses that implement all the required operations.
     *
     * @author Christoph Mueller
     */
    template<class T> class String {

    public:

        /** Define a local name for the character type. */
        typedef typename T::Char Char;

        /** Define a local name for the string size. */
        typedef typename T::Size Size;

        /** Index of not found substrings. */
        static const Size INVALID_POS;

        /** Create a new and empty string. */
        String(void);

        /**
         * Create a string with the initial data 'data'.
         *
         * @param data A zero-terminated string used to initialise this object.
         */
        String(const Char *data);

		/**
		 * Craete a string with the initial data 'data'. This constructor 
		 * performns the necessary conversions. It will only be available, if the
		 * appropriate U::Convert() method is implemented in the U character traits
		 * class.
		 *
		 * @param data A zero-terminated string used to initialise this object.
		 */
		template<class U> String(const String<U>& data);

        /**
         * Create a string containing 'cnt' times the character 'c'.
         *
         * @param c   A character.
         * @param cnt A number.
         */
        String(const Char c, const Size cnt);

        /**
         * Create a copy of 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        String(const String& rhs);

        /** Dtor. */
        ~String(void);

        /**
         * Deletes the current content of the string, allocates memory for a
         * string of length cnt, and returns a pointer to this buffer. The 
         * terminating zero character will be set at the end of the buffer, 
         * but the buffer itself will not be initialized.
         *
         * @param newLen The size of the new buffer, excluding terminating 
         *               zero.
         *
         * @return The pointer to the creating internal buffer.
         */
        Char * AllocateBuffer(const Size newLen);

        /**
         * Answer whether this string ends with the string 'str'.
         *
         * @param str The string to be searched at the end.
         *
         * @return true, if this string ends with 'str', false otherwise.
         */
        bool EndsWith(const String& str) const;

        /**
         * Answer a hash code of the string.
         *
         * @return A hash code of the string-
         */
        UINT32 CalcHashCode(void) const;

        /**
         * Prints into this string like sprintf.
         *
         * @param fmt The format string.
         * @param ... Optional parameters.
         */
        void Format(const Char *fmt, ...);

        /**
         * Answer the index of the first occurrence of 'c' in the string. The 
         * search begins with the 'beginningAt'th character. If the character
         * was not found, INVALID_POS is returned.
         *
         * @param c		      The character to be searched.
         * @param beginningAt The index of the first character to be tested.
         *
         * @return The first occurrence of 'c' after or at 'beginningAt' in the
         *         string or INVALID_POS, if the character is not in the string.
         */
        Size IndexOf(const Char c, const Size beginningAt = 0) const;

        /**
         * Answer whether the string is empty.
         *
         * @return true, if this string is empty, false otherwise.
         */
        inline bool IsEmpty(void) const {
            return (this->Length() < 1);
        }

        /**
         * Answer the index of the last occurrence of 'c' in the string.
         *
         * @param c The character to be searched.
         *
         * @return The last occurrence of 'c' in the string or INVALID_POS, if the
         *         character is not in the string.
         */
        Size LastIndexOf(const Char c) const;

        /**
         * Answer the length of the string.
         *
         * @return The number of characters in the string, excluding the
         *         terminating zero.
         */
        inline Size Length(void) const {
            //return T::SafeStringLength(this->data);
            return T::StringLength(this->data);
        }

        /**
         * Provide direct access to the character buffer of this string.
         *
         * @return A pointer to the characters of the string.
         */
        inline const Char *PeekBuffer(void) const {
            return this->data;
        }

        /**
         * Replace all occurrences of 'oldChar' in the string with 'newChar'.
		 *
		 * @param oldChar The character that is to be replaced.
		 * @param newChar The new character.
		 *
		 * @return The number of characters that have been replaced.
         */
        Size Replace(const Char oldChar, const Char newChar);

        ///**
        // * Replace all occurrences of 'oldStr' in the string with 'newStr'.
        // */
        //void Replace(const String oldStr, const String newStr);

        /**
         * Answer whether this string starts with the string 'str'.
         *
         * @param str The string to be searched at the begin.
         *
         * @return true, if this string starts with 'str', false otherwise.
         */
        bool StartsWith(const String& str) const;

        ///**
        // * Answer the substring beginning at 'begin' and reaching to the end of
        // * this string. If 'begin' is after the end of this string, an empty
        // * string is returned.
        // */
        //String Substring(const Size begin) const;

        ///**
        // * Answer the substring beginning at 'begin' and having a length of at
        // * most 'length' characters. If there are less than 'length' characters
        // * between 'begin' and the end of this string, the substring to the end
        // * is returned. If 'begin' is after the end of this string, an empty 
        // * string is returned.
        // */
        //String Substring(const Size begin, const Size length) const;

        ///**
        // * Remove all characters that are in the string 'chars' from the end
        // * of this string. 'chars' must be zero-terminated.
        // */
        //void TrimEnd(const T *chars);

        ///**
        // * Remove all characters that are in the string 'chars' from the start 
        // * of this string. 'chars' must be zero-terminated.
        // */
        //void TrimStart(const T *chars);

        ///**
        // * Remove all characters that are in the string 'chars' from the start
        // * and the end of this string. 'chars' must be zero-terminated.
        // */
        //inline void Trim(const T *chars) {
        //    this->TrimStart(chars);
        //    this->TrimEnd(chars);
        //}

        ///**
        // * Convert all characters to lower case.
        // */
        //inline void ToLowerCase(void) {
        //    T::ToLower(this->data);
        //}

        ///**
        // * Convert all characters to upper case.
        // */
        //inline void ToUpperCase(void) {
        //    T::ToUpper(this->data);
        //}

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return This string.
         */
        String& operator =(const String& rhs);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return This string.
         */
        String& operator =(const Char *rhs);

		/**
         * Conversion assignment.
		 * It will only be available, if the appropriate U::Convert() method 
         * is implemented in the U character traits class.
		 *
		 * @param rhs The right hand side operand.
         *
         * @return This string.
		 */
		template<class U> String& operator =(const String<U>& rhs);


        /**
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if 'rhs' and this string are equal, false otherwise.
         */
        bool operator ==(const String& rhs) const;

        /**
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if 'rhs' and this string are not equal, false 
         *         otherwise.
         */
        inline bool operator !=(const String& rhs) const {
            return !(*this == rhs);
        }

        /**
         * Concatenate this string and 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return A new string that is this string with 'rhs' appended.
         */
        String operator +(const String& rhs);

        /**
         * Concatenate this string and 'rhs'. This string will hold the result.
         *
         * @param rhs The right hand side operand.
         */
        void operator +=(const String& rhs);

        /**
         * Answer the 'i'th character in the string.
         *
         * @param i Index of the character to be returned.
         *
         * @return The character at the 'i'th position in the string.
         *
         * @throws OutOfRangeException If 'i' does not designate a valid 
         *                             character of the string.
         */
        Char operator [](const Size i) const;

        /**
         * Answer a reference to the 'i'th characet in the string.
         *
         * @param i Index of the character to be returned.
         *
         * @return Reference to the character at the 'i'th position in the string.
         *
         * @throws OutOfRangeException If 'i' does not designate a valid 
         *                             character of the string.
         */
        Char& operator [](const Size i);

        /**
         * Provide direct access to the characters of the string.
         *
         * @return A pointer to the characters of the string.
         */
        inline operator const Char *(void) const {
            return this->data;
        }

    private:

        /** The string data. */
        Char *data;

    }; /* end class String */


    /*
     * String<T>::INVALID_POS
     */
    template<class T> const typename String<T>::Size String<T>::INVALID_POS = -1;

    /*
     * String<T>::String
     */
    template<class T> String<T>::String(void) : data(NULL) {
        this->data = new Char[1];
        this->data[0] = 0;
    }


    /*
     * String<T>::String
     */
    template<class T> String<T>::String(const Char *data) : data(NULL) {
        Size newLen = T::SafeStringLength(data) + 1;
        this->data = new Char[newLen];

        if (newLen > 1) {
            ::memcpy(this->data, data, newLen * T::CharSize());
        } else {
            /* 'data' must be a NULL pointer. */
            this->data[0] = 0;
        }
    }

	/*
	 * String<T>::String
	 */
	template<class T> 
	template<class U> String<T>::String(const String<U>& data) {
		Size newLen = data.Length() + 1;
		this->data = new Char[newLen];
		U::Convert(this->data, newLen, 
			static_cast<const typename U::Char *>(data));
	}


    /*
     * String<T>::String
     */
    template<class T> String<T>::String(const Char c, const Size cnt) 
            : data(NULL) {
        this->data = new Char[cnt + 1];

        for (Size i = 0; i < cnt; i++) {
            this->data[i] = c;
        }

        this->data[cnt] = 0;
    }


    /*
     * String<T>::String
     */
    template<class T> String<T>::String(const String& rhs) : data(NULL) {
        Size newLen = rhs.Length() + 1;
        this->data = new Char[newLen];
        ::memcpy(this->data, rhs.data, newLen * T::CharSize());
    }


    /*
     * String<T>::~String
     */
    template<class T> String<T>::~String(void) {
        ARY_SAFE_DELETE(this->data);
    }


    /*
     * String<T>::AllocateBuffer
     */
    template<class T>  
    typename String<T>::Char * String<T>::AllocateBuffer(const Size newLen) {
        ARY_SAFE_DELETE(this->data);
        this->data = new Char[newLen + 1];
        this->data[newLen] = 0;
        return this->data;
    }


    /*
     * String<T>::EndsWith
     */
    template<class T> bool String<T>::EndsWith(const String& str) const {
        Size len1 = this->Length();
        Size len2 = str.Length();

        if (len2 <= len1) {
            for (Size i = (len1 - len2), j = 0; i < len1; i++, j++) {
                if (this->data[i] != str.data[j]) {
                    return false;
                }
            }
            /* No difference found. */

            return true;

        } else {
            /* Cannot end with 'str', if shorter. */
            return false;
        }
    }


    /*
     * String<T>::CalcHashCode
     */
    template<class T> UINT32 String<T>::CalcHashCode(void) const {
        // DJB2 hash function
        UINT32 hash = 0;
        Char c;
        Char *str = this->data;

        while ((c = *str++) != 0) {
            hash = ((hash << 5) + hash) + static_cast<UINT32>(c);
        }

        return hash;
    }

    /*
     * String<T>::Format
     */
    template<class T> void String<T>::Format(const Char *fmt, ...) {
        // TODO
        assert(false);
    }


    /*
     * String<T>::IndexOf
     */
    template<class T> typename String<T>::Size String<T>::IndexOf(const Char c,
			const Size beginningAt) const {
        Size len = this->Length();
        
        for (Size i = beginningAt; i < len; i++) {
            if (this->data[i] == c) {
                return static_cast<int>(i);
            }
        }
        /* Nothing found. */

        return INVALID_POS;
    }


    /*
     * String<T>::LastIndexOf
     */
    template<class T> typename String<T>::Size String<T>::LastIndexOf(
            const Char c) const {
        for (Size i = this->Length(); i >= 0; i--) {
            if (this->data[i] == c) {
                return static_cast<int>(i);
            }
        }
        /* Nothing found. */

        return INVALID_POS;
    }

	/*
	 * String<T>::Replace
	 */
    template<class T> typename String<T>::Size String<T>::Replace(
            const Char oldChar, const Char newChar) {
		Size retval = 0;
		Size len = this->Length();
		
		for (Size i = 0; i < len; i++) {
			if (this->data[i] == oldChar) {
				this->data[i] = newChar;
				retval++;
			}
		}

		return retval;
	}


    /*
     * String<T>::StartsWith
     */
    template<class T> bool String<T>::StartsWith(const String& str) const {
        Size len1 = this->Length();
        Size len2 = str.Length();

        if (len2 <= len1) {
            for (Size i = 0; i < len2; i++) {
                if (this->data[i] != str.data[i]) {
                    return false;
                }
            }
            /* No difference found. */

            return true;

        } else {
            /* Cannot start with 'str', if shorter. */
            return false;
        }
    }


    /*
     * String<T>::operator =
     */
    template<class T> String<T>& String<T>::operator =(const String& rhs) {
        if (this != &rhs) {
            delete[] this->data;

            Size newLen = rhs.Length() + 1;
            this->data = new Char[newLen];

            ::memcpy(this->data, rhs.data, newLen * T::CharSize());
        }

        return *this;
    }


    /*
     * String<T>::operator =
     */
    template<class T> String<T>& String<T>::operator =(const Char *rhs) {
        if (this->data != rhs) {
            delete[] this->data;

            Size newLen = T::SafeStringLength(rhs) + 1;
            this->data = new Char[newLen];

            ::memcpy(this->data, rhs, newLen * T::CharSize());
        }

        return *this;
    }


    /*
     * String<T>::operator =
     */
    template<class T> template<class U> 
    String<T>& String<T>::operator =(const String<U>& rhs) {
        if (static_cast<void*>(this) != static_cast<const void*>(&rhs)) {
            delete[] this->data;

            Size newLen = rhs.Length() + 1;
		    this->data = new Char[newLen];

            U::Convert(this->data, newLen, 
                static_cast<const typename U::Char *>(rhs));
        }

        return *this;
    }


    /*
     * String<T>::operator ==
     */
    template<class T> bool String<T>::operator ==(const String& rhs) const {
        const Char *str1 = this->data;
        const Char *str2 = rhs.data;

        while ((*str1 == *str2) && (*str1 != 0)) {
            str1++;
            str2++;
        }

        return (*str1 == *str2);
    }


    /*
     * String<T>::operator +
     */
    template<class T> String<T> String<T>::operator +(const String& rhs) {
        Size len1 = this->Length();
        Size len2 = rhs.Length();

        String retval;
        delete[] retval.data;   // tricky!
        retval.data = new Char[len1 + len2 + 1];

        ::memcpy(retval.data, this->data, len1 * T::CharSize());
        ::memcpy(retval.data + len1, rhs.data, (len2 + 1) * T::CharSize());

        return retval;
    }


    /*
     * String<T>::operator +=
     */
    template<class T> void String<T>::operator +=(const String& rhs) {
        Size len1 = this->Length();
        Size len2 = rhs.Length();
        Char *str = new Char[len1 + len2 + 1];

        ::memcpy(str, this->data, len1 * T::CharSize());
        ::memcpy(str + len1, rhs.data, (len2 + 1) * T::CharSize());

        delete[] this->data;
        this->data = str;
    }


    /*
     * String<T>::operator []
     */
    template<class T> 
    typename String<T>::Char String<T>::operator [](const Size i) const {
        if ((i >= 0) && (i < this->Length())) {
            return this->data[i];
        } else {
            throw OutOfRangeException(i, 0, 
                static_cast<int>(this->Length() - 1), __FILE__, __LINE__);
        }
    }


    /*
     * String<T>::operator []
     */
    template<class T> 
    typename String<T>::Char& String<T>::operator [](const Size i) {
        if ((i >= 0) && (i < this->Length())) {
            return this->data[i];
        } else {
            throw OutOfRangeException(i, 0, 
                static_cast<int>(this->Length() - 1), __FILE__, __LINE__);
        }
    }
    
    /** Template instantiation for ANSI strings. */
    typedef String<CharTraitsA> StringA;

    /** Template instantiation for wide strings. */
    typedef String<CharTraitsW> StringW;

    /** Template instantiation for TCHARs. */
    typedef String<TCharTraits> TString;

} /* end namespace vislib */

#endif /* VISLIB_STRING_H_INCLUDED */
