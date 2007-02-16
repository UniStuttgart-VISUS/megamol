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
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


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
         * Create a string with the first 'cnt' characters from 'data'. It is 
         * safe to pass a zero-terminated string that is shorter than 'cnt'
         * characters. No padding occurs in this case.
         *
         * @param data A string.
         * @param cnt  The number of characters to read.
         */
        String(const Char *data, const Size& cnt);

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
        Char *AllocateBuffer(const Size newLen);

        /**
         * Concatenate this string and 'rhs'. This string will hold the result.
         *
         * @param rhs The right hand side operand.
         */
        inline void Append(const Char rhs) {
            *this += rhs;
        }

        /**
         * Concatenate this string and 'rhs'. This string will hold the result.
         *
         * @param rhs The right hand side operand.
         */
        inline void Append(const Char *rhs) {
            *this += rhs;
        }

        /**
         * Concatenate this string and 'rhs'. This string will hold the result.
         *
         * @param rhs The right hand side operand.
         */
        inline void Append(const String& rhs) {
            *this += rhs.data;
        }

        /**
         * Remove all characters from the string.
         */
        void Clear(void);

        /**
         * TODO: Document
         */
        inline bool Compare(const Char *rhs) const;

        /**
         * TODO: Document
         */
        inline bool Compare(const String& rhs) const;

        /**
         * TODO: Document
         */
        inline bool CompareInsensitive(const Char *rhs) const;

        /**
         * TODO: Document
         */
        inline bool CompareInsensitive(const String& rhs) const;

        /**
         * Answer whether this string ends with the character 'c'.
         *
         * @param c The charater to be searched at the end.
         *
         * @return true, if this string ends with 'c', false otherwise.
         */
        bool EndsWith(const Char c) const;

        /**
         * Answer whether this string ends with the string 'str'.
         *
         * Note that for 'str' being a NULL pointer, the result will be false.
         *
         * @param str The string to be searched at the end.
         *
         * @return true, if this string ends with 'str', false otherwise.
         */
        bool EndsWith(const Char *str) const;

        /**
         * Answer whether this string ends with the string 'str'.
         *
         * @param str The string to be searched at the end.
         *
         * @return true, if this string ends with 'str', false otherwise.
         */
        inline bool EndsWith(const String& str) const {
            return this->EndsWith(str.data);
        }

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
        Size Find(const Char c, const Size beginningAt = 0) const;

        /**
         * Answer the index of the first occurrence of the string 'str' at or
         * after the 'beginningAt'th character. If the string was not found, 
         * INVALID_POS is returned.
         *
         * @param str         The string to be searched.
         * @param beginningAt The index of the first character to be tested.
         *
         * @return The index of the first occurrence of 'str' at or after
         *         'beginningAt' or INVALID_POS, if 'str' was not found.
         */
        Size Find(const Char *str, const Size beginningAt = 0) const;

        /**
         * Answer the index of the first occurrence of the string 'str' at or
         * after the 'beginningAt'th character. If the string was not found, 
         * INVALID_POS is returned.
         *
         * @param str         The string to be searched.
         * @param beginningAt The index of the first character to be tested.
         *
         * @return The index of the first occurrence of 'str' at or after
         *         'beginningAt' or INVALID_POS, if 'str' was not found.
         */
        inline Size Find(const String& str, const Size beginningAt = 0) const {
            return this->Find(str.data, beginningAt);
        }

        /**
         * Answer the index of the last occurrence of 'c' in the string.
         *
         * @param c           The character to be searched.
         * @param beginningAt This is the index of the first character to be 
         *                    tested (effectively the last position in the 
         *                    string to be tested). INVALID_POS can be used to
         *                    specify that the whole string is searched without
         *                    querying the string for its length.
         *
         * @return The last occurrence of 'c' in the string or INVALID_POS, if 
         *         the character is not in the string.
         */
        Size FindLast(const Char c, const Size beginningAt = INVALID_POS) const;

        /**
         * Answer the index of the last occurrence of 'str' in the string 
         * starting at most at 'beginningAt'. Setting 'beginningAt' to 
         * INVALID_POS actually finds the last occurrence.
         *
         * @param str         The string to be searched.
         * @param beginningAt This is the index of the first character to be 
         *                    tested (effectively the last position in the 
         *                    string to be tested). INVALID_POS can be used to
         *                    specify that the whole string is searched without
         *                    querying the string for its length.
         *
         * @return The begin of the last occurrence of 'str' in the string or
         *         INVALID_POS, if 'str' was not found.
         */
        Size FindLast(const Char *str, 
            const Size beginningAt = INVALID_POS) const;

        /**
         * Answer the index of the last occurrence of 'str' in the string 
         * starting at most at 'beginningAt'. Setting 'beginningAt' to 
         * INVALID_POS actually finds the last occurrence.
         *
         * @param str         The string to be searched.
         * @param beginningAt This is the index of the first character to be 
         *                    tested (effectively the last position in the 
         *                    string to be tested). INVALID_POS can be used to
         *                    specify that the whole string is searched without
         *                    querying the string for its length.
         *
         * @return The begin of the last occurrence of 'str' in the string or
         *         INVALID_POS, if 'str' was not found.
         */
        inline Size FindLast(const String& str, 
                const Size beginningAt = INVALID_POS) const {
            return this->FindLast(str.data, beginningAt);
        }

        /**
         * Prints into this string like sprintf.
         *
         * @param fmt The format string.
         * @param ... Optional parameters.
         */
        void Format(const Char *fmt, ...);

        /**
         * Answer a hash code of the string.
         *
         * @return A hash code of the string-
         */
        UINT32 HashCode(void) const;

        /**
         * Answer whether the string is empty.
         *
         * @return true, if this string is empty, false otherwise.
         */
        inline bool IsEmpty(void) const {
            return (this->Length() < 1);
        }

#ifdef _WIN32
        /**
         * Load a string resource from the string table of the module designated
         * by the instance handle hInst.
         *
         * @param id The ID of the resource to retrieve.
         *
         * @return true in case of success, false, if the resource could not be
         *         found.
         */
        bool Load(const HINSTANCE hInst, const UINT id);

        /**
         * Load a string resource using the instance handle of the current
         * executable.
         *
         * @param id The ID of the resource to retrieve.
         *
         * @return true in case of success, false, if the resource could not be
         *         found.
         */
        inline bool Load(const UINT id) {
            return this->Load(::GetModuleHandle(NULL), id);
        }
#endif /* _WIN32 */

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
         * Concatenate 'rhs' and this string. This string will hold the result.
         *
         * @param rhs The string to be prepended.
         */
        void Prepend(const Char rhs);

        /**
         * Concatenate 'rhs' and this string. This string will hold the result.
         *
         * @param rhs The string to be prepended.
         */
        void Prepend(const Char *rhs);

        /**
         * Concatenate 'rhs' and this string. This string will hold the result.
         *
         * @param rhs The string to be prepended.
         */
        inline void Prepend(const String& rhs) {
            this->Prepend(rhs.data);
        }

        /**
         * Delete at most 'cnt' characters from this string, beginnin at 
         * 'begin'. INVALID_POS for 'cnt' can be used to delete all characters
         * after 'begin'.
         *
         * @param begin The index of the first character to be removed.
         * @param cnt   The number of characters to be removed, INVALID_POS for
         *              all.
         */
        void Remove(const Size begin, const Size cnt = INVALID_POS);

        /**
         * Remove all occurrences of 'str' from the string.
         *
         * @param str The substring to be removed.
         */
        inline void Remove(const Char *str) {
            this->Replace(str, T::EMPTY_STRING);
        }

        /**
         * Remove all occurrences of 'str' from the string.
         *
         * @param str The substring to be removed.
         */
        inline void Remove(const String& str) {
            this->Replace(str.data, T::EMPTY_STRING);
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

        /**
         * Replace all occurrences of 'oldStr' in the string with 'newStr'.
         *
         * @param oldStr The string to be replaced.
         * @param newStr The replacement sequence.
         *
         * @return The number of replacements made.
         */
        Size Replace(const Char *oldStr, const Char *newStr);

        /**
         * Replace all occurrences of 'oldStr' in the string with 'newStr'.
         *
         * @param oldStr The string to be replaced.
         * @param newStr The replacement sequence.
         *
         * @return The number of replacements made.
         */
        inline Size Replace(const String& oldStr, const String& newStr) {
            return this->Replace(oldStr.data, newStr.data);
        }

        /**
         * Replace all occurrences of 'oldChar' in the string with 'newStr'.
         *
         * @param oldChar The character to be replaced.
         * @param newStr  The replacement sequence.
         *
         * @return The number of replacements made.
         */
        inline Size Replace(const Char oldChar, const Char *newStr) {
            Char tmp[2] = { oldChar, static_cast<Char>(0) };
            return this->Replace(tmp, newStr);
        }

        /**
         * Replace all occurrences of 'oldChar' in the string with 'newStr'.
         *
         * @param oldChar The character to be replaced.
         * @param newStr  The replacement sequence.
         *
         * @return The number of replacements made.
         */
        inline Size Replace(const Char oldChar, const String& newStr) {
            return this->Replace(oldChar, newStr.data);
        }

        /**
         * Replace all occurrences of 'oldStr' in the character with 'newChar'.
         *
         * @param oldStr  The string to be replaced.
         * @param newChar The replacement character.
         *
         * @return The number of replacements made.
         */
        inline Size Replace(const Char *oldStr, const Char newChar) {
            Char tmp[2] = { newChar, static_cast<Char>(0) };
            return this->Replace(oldStr, tmp);
        }

        /**
         * Replace all occurrences of 'oldStr' in the character with 'newChar'.
         *
         * @param oldStr  The string to be replaced.
         * @param newChar The replacement character.
         *
         * @return The number of replacements made.
         */
        inline Size Replace(const String& oldStr, const Char newChar) {
            return this->Replace(oldStr.data, newChar);
        }

        /**
         * Answer whether this string starts with the character 'c'.
         *
         * @param c The character to be searched at the begin.
         *
         * @return true, if this string starts with 'c', false otherwise.
         */
        bool StartsWith(const Char c) const;

        /**
         * Answer whether this string starts with the string 'str'.
         *
         * Note that for 'str' being a NULL pointer, the result is always false.
         *
         * @param str The string to be searched at the begin.
         *
         * @return true, if this string starts with 'str', false otherwise.
         */
        bool StartsWith(const Char *str) const;

        /**
         * Answer whether this string starts with the string 'str'.
         *
         * @param str The string to be searched at the begin.
         *
         * @return true, if this string starts with 'str', false otherwise.
         */
        inline bool StartsWith(const String& str) const {
            return this->StartsWith(str.data);
        }

        /**
         * Answer the substring beginning at 'begin' and reaching to the end of
         * this string. If 'begin' is after the end of this string, an empty
         * string is returned.
         *
         * @param begin The index of the first character of the substring.
         *
         * @return The substring beginning at 'begin'.
         */
        String Substring(const Size begin) const;

        /**
         * Answer the substring beginning at 'begin' and having a length of at
         * most 'length' characters. If there are less than 'length' characters
         * between 'begin' and the end of this string, the substring to the end
         * is returned. If 'begin' is after the end of this string, an empty 
         * string is returned.
         *
         * @param begin  The index of the first character of the substring.
         * @param length The length of the substring.
         *
         * @return The substring.
         */
        String Substring(const Size begin, const Size length) const;

        /**
         * Remove all characters that are in the string 'chars' from the start 
         * of this string. 'chars' must be zero-terminated.
         *
         * @param chars A string of characters to be removed from the begin of 
         *              this string.
         */
        void TrimBegin(const Char *chars);

        /**
         * Remove all characters that are in the string 'chars' from the end
         * of this string. 'chars' must be zero-terminated.
         *
         * @param chars A string of characters to be removed from the end of 
         *              this string.
         */
        void TrimEnd(const Char *chars);

        /**
         * Remove all characters that are in the string 'chars' from the start
         * and the end of this string. 'chars' must be zero-terminated.
         *
         * @param chars A string of characters to be removed from the begin and 
         *              end of this string.
         */
        inline void Trim(const Char *chars) {
            this->TrimBegin(chars);
            this->TrimEnd(chars);
        }

        /**
         * Remove all whitespace characters from the start of this string.
         */
        void TrimSpacesBegin(void);

        /**
         * Remove all whitespace characters from the end of this string. 
         */
        void TrimSpacesEnd(void);

        /**
         * Remove all whitespace characters from the start and the end of this
         * string.
         */
        inline void TrimSpaces(void) {
            this->TrimSpacesBegin();
            this->TrimSpacesEnd();
        }

        /**
         * If the string is longer than 'size' characters, truncate it to be 
         * 'size' characters. Note, that the trailing zero is added behind the
         * last character.
         *
         * @param size The maximum size to truncate the string to.
         */
        void Truncate(const Size size);

        // TODO: ToLowerCase
        ///**
        // * Convert all characters to lower case.
        // */
//#pragma deprecated(ToLowerCase)
//        inline void ToLowerCase(void) {
//            //T::ToLower(this->data);
//            // TODO: This is a quick hack and should not be used.
//            Size len = this->Length();
//            for (Size i = 0; i < len; i++) {
//                this->data[i] = T::ToLower(this->data[i]);
//            }
//        }

        // TODO: ToUpperCase
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
        bool operator ==(const Char *rhs) const;

        /**
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if 'rhs' and this string are equal, false otherwise.
         */
        inline bool operator ==(const String& rhs) const {
            return (*this == rhs.data);
        }

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
        String operator +(const Char rhs) const;

        /**
         * Concatenate this string and 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return A new string that is this string with 'rhs' appended.
         */
        String operator +(const Char *rhs) const;

        /**
         * Concatenate this string and 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return A new string that is this string with 'rhs' appended.
         */
        inline String operator +(const String& rhs) const {
            return (*this + rhs.data);
        }

        /**
         * Concatenate this string and 'rhs'. This string will hold the result.
         *
         * @param rhs The right hand side operand.
         */
        void operator +=(const Char rhs);

        /**
         * Concatenate this string and 'rhs'. This string will hold the result.
         *
         * @param rhs The right hand side operand.
         */
        void operator +=(const Char *rhs);

        /**
         * Concatenate this string and 'rhs'. This string will hold the result.
         *
         * @param rhs The right hand side operand.
         */
        inline void operator +=(const String& rhs) {
            *this += rhs.data;
        }

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
    template<class T> 
    const typename String<T>::Size String<T>::INVALID_POS = -1;

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
    template<class T> String<T>::String(const Char *data, const Size& cnt) 
            : data(NULL) {
        Size newLen = T::SafeStringLength(data);

        if (cnt < newLen) {
            newLen = cnt;
        }
            
        this->data = new Char[newLen + 1];
        ::memcpy(this->data, data, newLen * T::CharSize());
        this->data[newLen] = 0;
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
     * String<T>::Clear
     */
    template<class T> void String<T>::Clear(void) {
        delete[] this->data;
        this->data = new Char[1];
        *this->data = 0;
    }


    /*
     * String<T>::Compare
     */
    template<class T> bool String<T>::Compare(const Char *rhs) const {
        return (*this == rhs);
    }


    /*
     * String<T>::Compare
     */
    template<class T> bool String<T>::Compare(const String& rhs) const {
        return (*this == rhs.data);
    }


    /*
     * String<T>::CompareInsensitive
     */
    template<class T> bool String<T>::CompareInsensitive(const Char *rhs) const {
        const Char *str = this->data;

        if (rhs == NULL) {
            /* 
             * NULL pointer can never be equal as we have at least the terminating 
             * zero in our strings. 
             */
            return false;

        } else {
            while ((T::ToLower(*str) == T::ToLower(*rhs)) && (*str != 0)) {
                str++;
                rhs++;
            }

            return (T::ToLower(*str) == T::ToLower(*rhs));
        }
    }


    /*
     * String<T>::CompareInsensitive
     */
    template<class T> bool String<T>::CompareInsensitive(const String& rhs) const {
        return this->CompareInsensitive(rhs.data);
    }


    /*
     * String<T>::EndsWith
     */
    template<class T> bool String<T>::EndsWith(const Char c) const {
        Size len = this->Length();
        return (len > 0) ? (this->data[len - 1] == c) : false;
    }


    /*
     * String<T>::EndsWith
     */
    template<class T> bool String<T>::EndsWith(const Char *str) const {
        Size len1 = this->Length();
        Size len2 = T::SafeStringLength(str);

        if ((str != NULL) && (len2 <= len1)) {
            for (Size i = (len1 - len2), j = 0; i < len1; i++, j++) {
                if (this->data[i] != str[j]) {
                    return false;
                }
            }
            /* No difference found. */

            return true;

        } else {
            /* Cannot end with 'str', if shorter or 'str' is invalid. */
            return false;
        }
    }


    /*
     * String<T>::Find
     */
    template<class T> typename String<T>::Size String<T>::Find(const Char c,
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
     * String<T>::Find
     */
    template<class T> 
    typename String<T>::Size String<T>::Find(const Char *str, 
            const Size beginningAt) const {
        // TODO: Naive implementation, consider Rabin-Karp
        Size strLen = T::SafeStringLength(str);
        Size len = this->Length() - strLen;
        Size j = 0;

        if ((str == NULL) || (len - beginningAt < 0)) {
            /*
             * 'str' is not a valid string or 'str' is longer than search 
             * range, and therecore cannot be contained in this string. 
             */
            return INVALID_POS;
        }
        
        /* 
         * Search the substring. Note: 'len' is the last possible starting 
         * point for a substring match.
         */
        for (Size i = beginningAt; i <= len; i++) {
            for (j = 0; (j < strLen) && (this->data[i + j] == str[j]); j++);

            if (j == strLen) {
                /* Full 'str' was found, so return. */
                return i;
            }
        }

        return INVALID_POS;
    }


    /*
     * String<T>::FindLast
     */
    template<class T> typename String<T>::Size String<T>::FindLast(
            const Char c, const Size beginningAt) const {
        Size len = this->Length() - 1;

        if ((beginningAt > INVALID_POS) && (beginningAt < len)) {
            /* Valid begin is specified. */
            len = beginningAt;
        }

        for (Size i = len; i >= 0; i--) {
            if (this->data[i] == c) {
                return static_cast<int>(i);
            }
        }
        /* Nothing found. */

        return INVALID_POS;
    }


    /*
     * String<T>::FindLast
     */
    template<class T> 
    typename String<T>::Size String<T>::FindLast(const Char *str, 
            const Size beginningAt) const {
        // TODO: Naive implementation, consider Rabin-Karp
        Size strLen = T::SafeStringLength(str) - 1;
        Size len = this->Length() - 1;
        Size j = 0;

        if ((str == NULL) || (len < strLen)) {
            /*
             * 'str' is not a valid string or it is a longer string, that cannot
             * be contained in shorter one. 
             */
            return INVALID_POS;
        }

        if ((beginningAt > INVALID_POS) && (beginningAt < len)) {
            /* Valid begin is specified. */
            len = beginningAt;
        }

        for (Size i = len; i >= 0; i--) {
            for (j = strLen; (j >= 0) && (this->data[i - strLen + j] 
                    == str[j]); j--);

            if (j == -1) {
                return (i - strLen);
            }
        }

        return INVALID_POS;
    }


    /*
     * String<T>::Format
     */
    template<class T> void String<T>::Format(const Char *fmt, ...) {
        va_list argptr;
        Size size = 0;

        /* Determine required buffer size. */
        va_start(argptr, fmt);
        size = T::Format(NULL, 0, fmt, argptr);
        va_end(argptr);

        /* Allocate memory. */
        ASSERT(size >= 0);
        this->AllocateBuffer(size);
        
        /* Write the actual output. */
        va_start(argptr, fmt);
        T::Format(this->data, size + 1, fmt, argptr);
        va_end(argptr);
    }


    /*
     * String<T>::HashCode
     */
    template<class T> UINT32 String<T>::HashCode(void) const {
        // DJB2 hash function
        UINT32 hash = 0;
        Char c;
        Char *str = this->data;

        while ((c = *str++) != 0) {
            hash = ((hash << 5) + hash) + static_cast<UINT32>(c);
        }

        return hash;
    }


#ifdef _WIN32
    /*
     * String<T>::Load
     */
    template<class T>
    bool String<T>::Load(const HINSTANCE hInst, const UINT id) {
        INT bufLen = 128;
        INT charsRead = 0;
        Char *buffer = new Char[bufLen];
        ASSERT(buffer != NULL);

        while ((charsRead = T::StringFromResource(hInst, id, buffer, bufLen)) 
                == bufLen - 1) {
            delete[] buffer;
            bufLen *= 2;
            buffer = new Char[bufLen];
            ASSERT(buffer != NULL);
        }

        if (charsRead > 0) {
            *this = buffer;
        }

        ARY_SAFE_DELETE(buffer);
        return (charsRead > 0);

    }
#endif /* _WIN32 */


    /*
     * String<T>::Prepend
     */
    template<class T> void String<T>::Prepend(const Char rhs) {
        Size len = this->Length();
        Char *str = new Char[len + 2];

        str[0] = rhs;
        ::memcpy(str + 1, this->data, (len + 1) * T::CharSize());

        delete[] this->data;
        this->data = str;
    }


    /*
     * String<T>::Prepend
     */
    template<class T> void String<T>::Prepend(const Char *rhs) {
        if (rhs != NULL) {
            Size len1 = this->Length();
            Size len2 = T::StringLength(rhs);
            Char *str = new Char[len1 + len2 + 1];

            ::memcpy(str, rhs, len2 * T::CharSize());
            ::memcpy(str + len2, this->data, (len1 + 1) * T::CharSize());

            delete[] this->data;
            this->data = str;
        }
    }


    /*
     * String<T>::Remove
     */
    template<class T> void String<T>::Remove(const Size begin, const Size cnt) {
        Size len = this->Length();
        Char *str = NULL;

        if (begin < len) {
            if ((cnt != INVALID_POS) && (begin + cnt <= len)) {
                /* Remove characters from the middle. */
                str = new Char[len - cnt + 1];
                ::memcpy(str, this->data, begin * sizeof(Char));
                ::memcpy(str + begin, this->data + begin + cnt, 
                    (len - begin - cnt + 1) * sizeof(Char));

            } else {
                /* Remove all characters beginning at 'begin'. */
                str = new Char[begin + 1];
                ::memcpy(str, this->data, begin * sizeof(Char));
                str[begin] = 0;
            }

            /* Swap buffers. */
            delete[] this->data;
            this->data = str;
        }
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
     * String<T>::Replace
     */
    template<class T> typename String<T>::Size String<T>::Replace(
            const Char *oldStr, const Char *newStr) {
        Size retval = 0;                            // Number of replacements.
        Size oldLen = T::SafeStringLength(oldStr);  // Length of 'oldStr'.
        Size newLen = T::SafeStringLength(newStr);  // Length of 'newStr'.
        Size nextStart = 0;                         // Next search start.
        Char *str = NULL;                   
        Char *dst = NULL;
        Char *src = NULL;

        if (oldLen < 1) {
            /* String to replace is empty, nothing to do. */
            return 0;
        }

        /* Count number of replacements. */
        while ((nextStart = this->Find(oldStr, nextStart)) != INVALID_POS) {
            retval++;
            nextStart += oldLen;
        }

        if (retval < 1) {
            /* String to replace was not found, nothing to do. */
            return 0;
        }
        /* Must replace at least one occurrence at this point. */

        /* Copy the data into new buffer 'str'. */
        str = new Char[this->Length() - retval * oldLen + retval * newLen + 1];

        nextStart = 0;
        src = this->data;
        dst = str;
        while ((nextStart = this->Find(oldStr, nextStart)) != INVALID_POS) {
            while (src < this->data + nextStart) {
                *dst++ = *src++;
            }
            ::memcpy(dst, newStr, newLen * sizeof(Char));
            dst += newLen;
            src += oldLen;
            nextStart += oldLen;
        }
        while ((*dst++ = *src++) != 0);

        delete[] this->data;
        this->data = str;
        return retval;
    }


    /*
     * String<T>::StartsWith
     */
    template<class T> bool String<T>::StartsWith(const Char c) const {
        return (this->data[0] == c);
    }


    /*
     * String<T>::StartsWith
     */
    template<class T> bool String<T>::StartsWith(const Char *str) const {
        Size len1 = this->Length();
        Size len2 = T::SafeStringLength(str);

        if ((str != NULL) && (len2 <= len1)) {
            for (Size i = 0; i < len2; i++) {
                if (this->data[i] != str[i]) {
                    return false;
                }
            }
            /* No difference found. */

            return true;

        } else {
            /* Cannot start with 'str', if shorter or 'str' invalid. */
            return false;
        }
    }


    /*
     * String<T>::Substring
     */
    template<class T> String<T> String<T>::Substring(const Size begin) const {
        Size len = this->Length();

        if (begin >= len) {
            return String();
        } else {
            return String(this->data + begin);
        }
    }


    /*
     * String<T>::Substring
     */
    template<class T> String<T> String<T>::Substring(const Size begin, 
            const Size length) const {
        Size len = this->Length();

        if (begin >= len) {
            return String();
        } else {
            return String(this->data + begin, length);
        }
    }


    /*
     * vislib::String<T>::TrimBegin
     */
    template<class T> void String<T>::TrimBegin(const Char *chars) {
        const Char *c = NULL;
        Char *s = NULL;
        Size len = this->Length();
        Size i = 0;

        for (i = 0; i < len; i++) {

            /* Check for any character in 'chars' to match. */
            c = chars;
            while (*c && (this->data[i] != *c)) {
                c++;
            }

            if (*c == 0) {
                /* First character not matching any in 'chars' found. */
                break;
            }
        }

        if (i > 0) {
            /* Remove characters at begin. */
            len -= i - 1;
            s = new Char[len];

            ::memcpy(s, this->data + i, len * sizeof(Char));

            delete[] this->data;
            this->data = s;
        }
    }


    /*
     * vislib::String<T>::TrimEnd
     */
    template<class T> void String<T>::TrimEnd(const Char *chars) {
        const Char *c = NULL;
        Char *s = NULL;
        Size len = this->Length();
        Size i = 0;

        for (i = len - 1; i >= 0; i--) {

            /* Check for any character in 'chars' to match. */
            c = chars;
            while (*c && (this->data[i] != *c)) {
                c++;
            }

            if (*c == 0) {
                /* First character not matching any in 'chars' found. */
                break;
            }
        }

        if (i < len - 1) {
            /* Remove characters at end. */
            len -= len - i - 1;
            s = new Char[len + 1];

            ::memcpy(s, this->data, len * sizeof(Char));
            s[len] = 0;

            delete[] this->data;
            this->data = s;
        }
    }

    
    /*
     * String<T>::TrimSpacesBegin
     */
    template<class T> void String<T>::TrimSpacesBegin(void) {
        if ((this->data != NULL) && T::IsSpace(this->data[0])) {
            Char *begin = NULL;
            unsigned int len = 0;
            Char *ptr;

            // calculate string length without the leading whitespaces
            for (ptr = this->data; *ptr != 0; ptr++) {
                if (begin == NULL) {
                    if (!T::IsSpace(*ptr)) { // first non-whitespace character
                        begin = ptr;
                        len = 1;
                    }

                } else {
                    len++;

                }
            }

            ptr = new Char[len + 1]; 
            memcpy(ptr, begin, len * sizeof(Char));
            ptr[len] = 0;

            delete[] this->data;
            this->data = ptr;
        }
    }


    /*
     * String<T>::TrimSpacesEnd
     */
    template<class T> void String<T>::TrimSpacesEnd(void) {
        unsigned int len = this->Length();
        while (len > 0) {
            len--;
            if (!T::IsSpace(this->data[len])) {
                len++;
                break;
            }
        }

        if (len > 0) {
            Char *nb = new Char[len + 1];
            memcpy(nb, this->data, len * sizeof(Char));
            nb[len] = 0;

            delete[] this->data;
            this->data = nb;
        }
    }


    /*
     * String<T>::Truncate
     */
    template<class T> void String<T>::Truncate(const Size size) {
        Size len = this->Length();
        Char *str = NULL;

        if (size < len) {
            str = new Char[size + 1];
            ::memcpy(str, this->data, size * sizeof(Char));
            str[size] = 0;

            delete[] this->data;
            this->data = str;
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

            if (rhs != NULL) {
                ::memcpy(this->data, rhs, newLen * T::CharSize());

            } else {
                this->data[0] = 0;

            }
        }

        return *this;
    }


    /*
     * String<T>::operator =
     */
    template<class T> template<class U> 
    String<T>& String<T>::operator =(const String<U>& rhs) {
        if (static_cast<void *>(this) != static_cast<const void *>(&rhs)) {
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
    template<class T> bool String<T>::operator ==(const Char *rhs) const {
        const Char *str = this->data;

        if (rhs == NULL) {
            /* 
             * NULL pointer can never be equal as we have at least the terminating 
             * zero in our strings. 
             */
            return false;

        } else {
            while ((*str == *rhs) && (*str != 0)) {
                str++;
                rhs++;
            }

            return (*str == *rhs);
        }
    }


    /*
     * String<T>::operator +
     */
    template<class T> String<T> String<T>::operator +(const Char rhs) const {
        Size len = this->Length();

        String retval;
        delete[] retval.data;   // tricky!
        retval.data = new Char[len + 2];

        ::memcpy(retval.data, this->data, len * T::CharSize());
        retval.data[len] = rhs;
        retval.data[len + 1] = 0;

        return retval;
    }


    /*
     * String<T>::operator +
     */
    template<class T> String<T> String<T>::operator +(const Char *rhs) const {
        if (rhs != NULL) {
            Size len1 = this->Length();
            Size len2 = T::StringLength(rhs);

            String retval;
            delete[] retval.data;   // tricky!
            retval.data = new Char[len1 + len2 + 1];

            ::memcpy(retval.data, this->data, len1 * T::CharSize());
            ::memcpy(retval.data + len1, rhs, (len2 + 1) * T::CharSize());

            return retval;

        } else {
            return String(this->data);
        }
    }


    /*
     * String<T>::operator +=
     */
    template<class T> void String<T>::operator +=(const Char rhs) {
        Size len = this->Length();
        Char *str = new Char[len + 2];

        ::memcpy(str, this->data, len * T::CharSize());
        str[len] = rhs;
        str[len + 1] = 0;

        delete[] this->data;
        this->data = str;
    }


    /*
     * String<T>::operator +=
     */
    template<class T> void String<T>::operator +=(const Char *rhs) {
        if (rhs != NULL) {
            Size len1 = this->Length();
            Size len2 = T::StringLength(rhs);
            Char *str = new Char[len1 + len2 + 1];

            ::memcpy(str, this->data, len1 * T::CharSize());
            ::memcpy(str + len1, rhs, (len2 + 1) * T::CharSize());

            delete[] this->data;
            this->data = str;
        }
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

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_STRING_H_INCLUDED */
