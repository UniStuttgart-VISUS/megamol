/*
 * String.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 - 2009 by Christoph Mueller. All rights reserved.
 */

#ifndef VISLIB_STRING_H_INCLUDED
#define VISLIB_STRING_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/CharTraits.h"
#include "vislib/deprecated.h"
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

        /** An empty string instance. */
        static const String EMPTY;

        /** Index of not found substrings. */
        static const Size INVALID_POS;

        /**
         * This value for replace limit guarantees that all occurrences are 
         * replaced. 
         */
        static const Size NO_LIMIT;

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
         * @param data A string of at least 'cnt' characters.
         * @param cnt  The number of characters to read.
         */
        String(const Char *data, const Size& cnt);

        /**
         * Create a string with the initial data 'data'. This constructor 
         * performs the necessary conversions. It will only be available, if the
         * appropriate U::Convert() method is implemented in the U character traits
         * class.
         *
         * @param data A string.
         */
        template<class U> String(const String<U>& data);

        /**
         * Create a string with the initial data 'data'. This constructor performs
         * the necessary conversions. It is only available if CharTraits<U> exists
         * and has the appropriate Convert() method.
         *
         * Note: This ctor cannot be used for implict string conversion as it 
         * would introduce a bunch of ambiguities.
         *
         * @param data A zero-terminated string used to initialise this object.
         */
        template<class U> explicit String(const U *data);

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
         * Append 'cnt' characters starting at 'rhs' to the string. This string 
         * will hold the result.
         *
         * @param rhs The string to be appended.
         * @param cnt The number of characters to be appended.
         */
        void Append(const Char *rhs, const Size cnt);

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
         * Compare this string and 'rhs'.
         *
         * @param rhs             The right hand side operand.
         * @param isCaseSensitive If true, the string will be compared 
         *                        case-sensitive, otherwise, the comparison
         *                        will be case-insensitive.
         *
         * @return -1 if (this < 'rhs'), 0 if (this == 'rhs'), 
         *         +1 if (this > 'rhs')
         */
        Size Compare(const Char *rhs, const bool isCaseSensitive) const;

        /**
         * Compare this string and 'rhs'.
         *
         * @param rhs A string.
         * @param isCaseSensitive If true, the string will be compared 
         *                        case-sensitive, otherwise, the comparison
         *                        will be case-insensitive.
         *
         * @return -1 if (this < 'rhs'), 0 if (this == 'rhs'), 
         *         +1 if (this > 'rhs')
         */
        inline Size Compare(const String& rhs, 
                const bool isCaseSensitive) const {
            return this->Compare(rhs.data, isCaseSensitive);
        }

        /**
         * Answer whether both strings have the same value.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if both strings are equal, false otherwise.
         */
        VLDEPRECATED inline bool Compare(const Char *rhs) const {
        // TODO: The semantics of this method is unintuitive and should be 
        // changed to be similar to strcmp.
        // TODO: Merge Compare and CompareInsensitive to one method and
        // specify case-sensitivity via parameter.
            return this->Equals(rhs, true);
        }

        /**
         * Answer whether both strings have the same value.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if both strings are equal, false otherwise.
         */
        VLDEPRECATED inline bool Compare(const String& rhs) const {
        // TODO: The semantics of this method is unintuitive and should be 
        // changed to be similar to strcmp.
        // TODO: Merge Compare and CompareInsensitive to one method and
        // specify case-sensitivity via parameter.
            return this->Equals(rhs, true);
        }

        /**
         * Answer whether both strings have the same value, comparing them
         * case insensitive.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if both strings are equal, false otherwise.
         */
        VLDEPRECATED inline bool CompareInsensitive(const Char *rhs) const {
            return this->Equals(rhs, false);
        }

        /**
         * Answer whether both strings have the same value, comparing them
         * case insensitive.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if both strings are equal, false otherwise.
         */
        VLDEPRECATED inline bool CompareInsensitive(const String& rhs) const {
            return this->Equals(rhs, false);
        }

        /**
         * Answer whether the string contains 'c'. The search begins with the 
         * 'beginningAt'th character.
         *
         * @param c		      The character to be searched.
         * @param beginningAt The index of the first character to be tested.
         *
         * @return true if 'c' is found at 'beginningAt' or after that.
         */
        inline bool Contains(const Char c, const Size beginningAt = 0) const {
            return (this->Find(c, beginningAt) != INVALID_POS);
        }

        /**
         * Answer whether the string contains 'str'. The search begins with the 
         * 'beginningAt'th character.
         *
         * @param str         The string to be searched.
         * @param beginningAt The index of the first character to be tested.
         *
         * @return true if 'str' is found at 'beginningAt' or after that.
         */
        inline bool Contains(const Char *str, 
                const Size beginningAt = 0) const {
            return (this->Find(str, beginningAt) != INVALID_POS);
        }

        /**
         * Answer whether the string contains 'str'. The search begins with the 
         * 'beginningAt'th character.
         *
         * @param str         The string to be searched.
         * @param beginningAt The index of the first character to be tested.
         *
         * @return true if 'str' is found at 'beginningAt' or after that.
         */
        inline bool Contains(const String& str, 
                const Size beginningAt = 0) const {
            return (this->Find(str.data, beginningAt) != INVALID_POS);
        }

        /**
         * Count the occurences of 'c' at or after 'beginningAt'.
         *
         * @param c		      The character to be counted.
         * @param beginningAt The index of the first character to be tested.
         *
         * @return The number of occurrences of 'c' in the substring starting
         *         at 'beginningAt'.
         */
        SIZE_T Count(const Char c, const Size beginningAt = 0) const;

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
         * Answer whether this string and 'str' are equal.
         *
         * @param rhs A string.
         * @param isCaseSensitive If true, the string will be compared 
         *                        case-sensitive, otherwise, the comparison
         *                        will be case-insensitive.
         */
        bool Equals(const Char *rhs, const bool isCaseSensitive = true) const;

        /**
         * Answer whether this string and 'str' are equal.
         *
         * @param rhs A string.
         * @param isCaseSensitive If true, the string will be compared 
         *                        case-sensitive, otherwise, the comparison
         *                        will be case-insensitive.
         */
        inline bool Equals(const String& rhs, 
                const bool isCaseSensitive = true) const {
            return this->Equals(rhs.data, isCaseSensitive);
        }

        /**
         * Escapes a set of characters within the string with a given escape
         * character.
         *
         * Example:
         *  EscapeCharacters('\\', "\n\r\t", "nrt");
         *
         * @param ec The escape character. Must not be present in the 
         *           parameters 'normChars' and 'escpdChars'. Must not be zero.
         * @param normChars This string represents the list of characters to
         *                  be escaped. The character 'ec' must not be
         *                  included, since it is implicitly included. Each
         *                  character must only be present once.
         * @param escpdChars This string represents the list of escaped 
         *                   characters. The string must be of same length
         *                   as 'normChars', must not contain any character
         *                   more than once, and must not contain 'ec'.
         *
         * @return 'true' if the string was successfully escaped.
         *         'false' if there was an error.
         */
        inline bool EscapeCharacters(const Char ec, const String& normChars, 
                const String& escpdChars) {
            return this->EscapeCharacters(ec, normChars.PeekBuffer(),
                escpdChars.PeekBuffer());
        }

        /**
         * Escapes a set of characters within the string with a given escape
         * character.
         *
         * Example:
         *  EscapeCharacters('\\', "\n\r\t", "nrt");
         *
         * @param ec The escape character. Must not be present in the 
         *           parameters 'normChars' and 'escpdChars'. Must not be zero.
         * @param normChars This string represents the list of characters to
         *                  be escaped. The character 'ec' must not be
         *                  included, since it is implicitly included. Each
         *                  character must only be present once.
         * @param escpdChars This string represents the list of escaped 
         *                   characters. The string must be of same length
         *                   as 'normChars', must not contain any character
         *                   more than once, and must not contain 'ec'.
         *
         * @return 'true' if the string was successfully escaped.
         *         'false' if there was an error.
         */
        bool EscapeCharacters(const Char ec, const Char *normChars, 
            const Char *escpdChars);

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
         * Prints into this string like sprintf.
         *
         * @param fmt The format string.
         * @param ... Optional parameters.
         */
        void Format(const String *format, ...);

        /**
         * Prints into this string like vsprintf.
         *
         * @param fmt The format string.
         * @param ... Optional parameters.
         */
        void FormatVa(const Char *fmt, va_list argptr);

        /**
         * Prints into this string like vsprintf.
         *
         * @param fmt The format string.
         * @param ... Optional parameters.
         */
        inline void FormatVa(const String& fmt, va_list argptr) {
            this->FormatVa(fmt.PeekBuffer(), argptr);
        }

        /**
         * Answer a hash code of the string.
         *
         * @return A hash code of the string.
         */
        UINT32 HashCode(void) const;

        /**
         * Insert 'c' at position 'pos' in the string. 'pos' must be a valid 
         * index in the string or the length of the string. In the latter case, 
         * the method behaves like Append().
         *
         * @param pos The position to insert the character at.
         * @param c   The character to add.
         *
         * @throws OutOfRangeException If 'pos' is not within 
         *                             [0, this->Length()].
         */
        void Insert(const Size pos, const Char c);

        /**
         * Insert 'str' at position 'pos' in the string. 'pos' must be a valid 
         * index in the string or the length of the string. In the latter case, 
         * the method behaves like Append().
         *
         * @param pos The position to insert the string at.
         * @param c   The string to add. It is safe to pass a NULL pointer.
         *
         * @throws OutOfRangeException If 'pos' is not within 
         *                             [0, this->Length()].
         */
        void Insert(const Size pos, const Char *str);

        /**
         * Insert 'str' at position 'pos' in the string. 'pos' must be a valid 
         * index in the string or the length of the string. In the latter case, 
         * the method behaves like Append().
         *
         * @param pos The position to insert the string at.
         * @param c   The string to add.
         *
         * @throws OutOfRangeException If 'pos' is not within 
         *                             [0, this->Length()].
         */
        inline void Insert(const Size pos, const String& str) {
            this->Insert(pos, str.data);
        }

        /**
         * Answer whether the string is empty.
         *
         * @return true, if this string is empty, false otherwise.
         */
        inline bool IsEmpty(void) const {
            return (this->data[0] == static_cast<Char>(0));
        }

        /**
         * Compute the Levenshtein 
         * (http://www.keldysh.ru/departments/dpt_10/lev.html) distance between
         * this string and 'rhs'. Note that this string is considered the 
         * expected text and 'rhs' the actually found one.
         *
         * This operation has a memory consumption of 
         * (this->Length() + 1) * (ths.Length() + 1) * sizeof(Size).
         *
         * @rhs              The string to be compared.
         * @param costAdd    The cost for an insertion (default 1).
         * @param costDelete The cost for a deletion (default 1).
         * @param costChange The cost for a substitution (default 1).
         *
         * @return The Levenshtein distance using the given weights.
         *
         * @throws std::bad_alloc if the temporary buffer for computing the 
         *                        result cannot be allocated.
         */
        Size LevenshteinDistance(const Char *rhs, const Size costAdd = 1,
            const Size costDelete = 1, const Size costChange = 1) const;

        /**
         * Compute the Levenshtein 
         * (http://www.keldysh.ru/departments/dpt_10/lev.html) distance between
         * this string and 'rhs'. Note that this string is considered the 
         * expected text and 'rhs' the actually found one.
         *
         * This operation has a memory consumption of 
         * (this->Length() + 1) * (ths.Length() + 1) * sizeof(Size).
         *
         * @rhs              The string to be compared.
         * @param costAdd    The cost for an insertion (default 1).
         * @param costDelete The cost for a deletion (default 1).
         * @param costChange The cost for a substitution (default 1).
         *
         * @return The Levenshtein distance using the given weights.
         *
         * @throws std::bad_alloc if the temporary buffer for computing the 
         *                        result cannot be allocated.
         */
        inline Size LevenshteinDistance(const String& rhs, 
                const Size costAdd = 1, const Size costDelete = 1, 
                const Size costChange = 1) const {
            return this->LevenshteinDistance(rhs.data, costAdd, 
                costDelete, costChange);
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
         * @param limit   If this is a positive integer, at most 'limit' 
         *                replacements are made. Use NO_LIMIT for replacing all
         *                occurrences.
         *
         * @return The number of characters that have been replaced.
         */
        Size Replace(const Char oldChar, const Char newChar, 
            const Size limit = NO_LIMIT);

        /**
         * Replace all occurrences of 'oldStr' in the string with 'newStr'.
         *
         * @param oldStr The string to be replaced.
         * @param newStr The replacement sequence.
         * @param limit  If this is a positive integer, at most 'limit' 
         *               replacements are made. Use NO_LIMIT for replacing all
         *               occurrences.
         *
         * @return The number of replacements made.
         */
        Size Replace(const Char *oldStr, const Char *newStr,
            const Size limit = NO_LIMIT);

        /**
         * Replace all occurrences of 'oldStr' in the string with 'newStr'.
         *
         * @param oldStr The string to be replaced.
         * @param newStr The replacement sequence.
         * @param limit  If this is a positive integer, at most 'limit' 
         *               replacements are made. Use NO_LIMIT for replacing all
         *               occurrences.
         *
         * @return The number of replacements made.
         */
        inline Size Replace(const String& oldStr, const String& newStr,
                const Size limit = NO_LIMIT) {
            return this->Replace(oldStr.data, newStr.data, limit);
        }

        /**
         * Replace all occurrences of 'oldChar' in the string with 'newStr'.
         *
         * @param oldChar The character to be replaced.
         * @param newStr  The replacement sequence.
         * @param limit   If this is a positive integer, at most 'limit' 
         *                replacements are made. Use NO_LIMIT for replacing all
         *                occurrences.
         *
         * @return The number of replacements made.
         */
        inline Size Replace(const Char oldChar, const Char *newStr,
                const Size limit = NO_LIMIT) {
            Char tmp[2] = { oldChar, static_cast<Char>(0) };
            return this->Replace(tmp, newStr, limit);
        }

        /**
         * Replace all occurrences of 'oldChar' in the string with 'newStr'.
         *
         * @param oldChar The character to be replaced.
         * @param newStr  The replacement sequence.
         * @param limit   If this is a positive integer, at most 'limit' 
         *                replacements are made. Use NO_LIMIT for replacing all
         *                occurrences.
         *
         * @return The number of replacements made.
         */
        inline Size Replace(const Char oldChar, const String& newStr,
                const Size limit = NO_LIMIT) {
            return this->Replace(oldChar, newStr.data, limit);
        }

        /**
         * Replace all occurrences of 'oldStr' in the character with 'newChar'.
         *
         * @param oldStr  The string to be replaced.
         * @param newChar The replacement character.
         * @param limit   If this is a positive integer, at most 'limit' 
         *                replacements are made. Use NO_LIMIT for replacing all
         *                occurrences.
         *
         * @return The number of replacements made.
         */
        inline Size Replace(const Char *oldStr, const Char newChar,
                const Size limit = NO_LIMIT) {
            Char tmp[2] = { newChar, static_cast<Char>(0) };
            return this->Replace(oldStr, tmp, limit);
        }

        /**
         * Replace all occurrences of 'oldStr' in the character with 'newChar'.
         *
         * @param oldStr  The string to be replaced.
         * @param newChar The replacement character.
         * @param limit   If this is a positive integer, at most 'limit' 
         *                replacements are made. Use NO_LIMIT for replacing all
         *                occurrences.
         *
         * @return The number of replacements made.
         */
        inline Size Replace(const String& oldStr, const Char newChar,
                const Size limit = NO_LIMIT) {
            return this->Replace(oldStr.data, newChar, limit);
        }

        /**
         * Obfuscate the string by applying Rot48. This works only on
         * characters between 32 and 127.
         */
        void Rot48(void);

        /**
         * Answer whether this string starts with the character 'c'.
         *
         * @param c               The character to be searched at the begin.
         * @param isCaseSensitive If true, the string will be compared 
         *                        case-sensitive, otherwise, the comparison
         *                        will be case-insensitive.
         *
         * @return true, if this string starts with 'c', false otherwise.
         */
        bool StartsWith(const Char c, const bool isCaseSensitive = true) const;

        /**
         * Answer whether this string starts with the string 'str'.
         *
         * Note that for 'str' being a NULL pointer, the result is always false.
         *
         * @param str             The string to be searched at the begin.
         * @param isCaseSensitive If true, the string will be compared 
         *                        case-sensitive, otherwise, the comparison
         *                        will be case-insensitive.
         *
         * @return true, if this string starts with 'str', false otherwise.
         */
        bool StartsWith(const Char *str, 
            const bool isCaseSensitive = true) const;

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
         * Answer whether this string starts with the character 'c'. The 
         * comparation of the strings will be performed case insensitive.
         *
         * @param c The character to be searched at the begin.
         *
         * @return true, if this string starts with 'c', false otherwise.
         */
        inline bool StartsWithInsensitive(const Char c) const {
            return this->StartsWith(c, false);
        }

        /**
         * Answer whether this string starts with the string 'str'. The 
         * comparation of the strings will be performed case insensitive.
         *
         * Note that for 'str' being a NULL pointer, the result is always false.
         *
         * @param str The string to be searched at the begin.
         *
         * @return true, if this string starts with 'str', false otherwise.
         */
        inline bool StartsWithInsensitive(const Char *str) const {
            return this->StartsWith(str, false);
        }

        /**
         * Answer whether this string starts with the string 'str'. The 
         * comparation of the strings will be performed case insensitive.
         *
         * @param str The string to be searched at the begin.
         *
         * @return true, if this string starts with 'str', false otherwise.
         */
        inline bool StartsWithInsensitive(const String& str) const {
            return this->StartsWithInsensitive(str.data);
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

        /**
         * Convert all characters to lower case.
         */
        inline void ToLowerCase(void) {
            // TODO: This is still a hack.
            T::ToLower(this->data, this->Length() + 1, this->data);
        }

        /**
         * Convert all characters to upper case.
         */
        inline void ToUpperCase(void) {
            // TODO: This is still a hack.
            T::ToUpper(this->data, this->Length() + 1, this->data);
        }

        /**
         * Unescapes a set of characters within the escaped string with a 
         * given escape character.
         *
         * Example:
         *  UnescapeCharacters('\\', "\n\r\t", "nrt");
         *
         * @param ec The escape character. Must not be present in the 
         *           parameters 'normChars' and 'escpdChars'. Must not be zero.
         * @param normChars This string represents the list of characters to
         *                  be restored by unescaping. The character 'ec' must
         *                  not be included, since it is implicitly included.
         *                  Each character must only be present once.
         * @param escpdChars This string represents the list of escaped 
         *                   characters. The string must be of same length
         *                   as 'normChars', must not contain any character
         *                   more than once, and must not contain 'ec'.
         *
         * @return 'true' if the string was successfully unescaped.
         *         'false' if there was an error.
         */
        inline bool UnescapeCharacters(const Char ec, const String& normChars, 
                const String& escpdChars) {
            return this->UnescapeCharacters(ec, normChars.PeekBuffer(),
                escpdChars.PeekBuffer());
        }

        /**
         * Unescapes a set of characters within the escaped string with a 
         * given escape character.
         *
         * Example:
         *  UnescapeCharacters('\\', "\n\r\t", "nrt");
         *
         * @param ec The escape character. Must not be present in the 
         *           parameters 'normChars' and 'escpdChars'. Must not be zero.
         * @param normChars This string represents the list of characters to
         *                  be restored by unescaping. The character 'ec' must
         *                  not be included, since it is implicitly included.
         *                  Each character must only be present once.
         * @param escpdChars This string represents the list of escaped 
         *                   characters. The string must be of same length
         *                   as 'normChars', must not contain any character
         *                   more than once, and must not contain 'ec'.
         *
         * @return 'true' if the string was successfully unescaped.
         *         'false' if there was an error.
         */
        bool UnescapeCharacters(const Char ec, const Char *normChars, 
            const Char *escpdChars);

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
         * Conversion assignment.
         * It is only available if CharTraits<U> exists and has the appropriate 
         * Convert() method.
         *
         * @param rhs The right hand side operand.
         *
         * @return This string.
         */
        template<class U> String& operator =(const U *rhs);

        /**
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if 'rhs' and this string are equal, false otherwise.
         */
        inline bool operator ==(const Char *rhs) const {
            return this->Equals(rhs, true);
        }

        /**
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if 'rhs' and this string are equal, false otherwise.
         */
        inline bool operator ==(const String& rhs) const {
            return this->Equals(rhs.data, true);
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
     * vislib::String<T>::EMPTY
     */
    template<class T>
    const String<T> String<T>::EMPTY; //(CharTraits<T>::EMPTY_STRING);


    /*
     * vislib::String<T>::INVALID_POS
     */
    template<class T> 
    const typename String<T>::Size String<T>::INVALID_POS = -1;


    /*
     * vislib::String<T>::NO_LIMIT
     */
    template<class T> 
    const typename String<T>::Size String<T>::NO_LIMIT = -1;


    /*
     * vislib::String<T>::String
     */
    template<class T> String<T>::String(void) : data(NULL) {
        this->data = new Char[1];
        this->data[0] = 0;
    }


    /*
     * vislib::String<T>::String
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
     * vislib::String<T>::String
     */
    template<class T> String<T>::String(const Char *data, const Size& cnt) 
            : data(NULL) {
        Size newLen = T::SafeStringLength(data);

        if (cnt < newLen) {
            newLen = cnt;
        }
            
        this->data = new Char[newLen + 1];
        if (data != NULL) {
            ::memcpy(this->data, data, newLen * T::CharSize());
            this->data[newLen] = 0;
        } else {
            this->data[0] = 0;
        }
    }


    /*
     * vislib::String<T>::String
     */
    template<class T> 
    template<class U> String<T>::String(const String<U>& data) {
        Size newLen = data.Length() + 1;
        this->data = new Char[newLen];
        U::Convert(this->data, newLen, 
            static_cast<const typename U::Char *>(data));
    }


    /*
     * vislib::String<T>::String
     */
    template<class T> 
    template<class U> String<T>::String(const U *data) {
        Size newLen = CharTraits<U>::SafeStringLength(data) + 1;
        this->data = new Char[newLen];
        if (newLen > 1) {
            CharTraits<U>::Convert(this->data, newLen, data);
        } else {
            /* 'data' must be a NULL pointer. */
            this->data[0] = 0;
        }
    }


    /*
     * vislib::String<T>::String
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
     * vislib::String<T>::String
     */
    template<class T> String<T>::String(const String& rhs) : data(NULL) {
        Size newLen = rhs.Length() + 1;
        this->data = new Char[newLen];
        ::memcpy(this->data, rhs.data, newLen * T::CharSize());
    }


    /*
     * vislib::String<T>::~String
     */
    template<class T> String<T>::~String(void) {
        ARY_SAFE_DELETE(this->data);
    }


    /*
     * vislib::String<T>::AllocateBuffer
     */
    template<class T>
    typename String<T>::Char *String<T>::AllocateBuffer(const Size newLen) {
        ARY_SAFE_DELETE(this->data);
        this->data = new Char[newLen + 1];
        this->data[newLen] = 0;
        return this->data;
    }


    /*
     * vislib::String<T>::Append
     */
    template<class T> void String<T>::Append(const Char *rhs, const Size cnt) {
        Size len = this->Length();
        Char *str = new Char[len + cnt + 1];

        ::memcpy(str, this->data, len * T::CharSize());
        ::memcpy(str + len, rhs, cnt * T::CharSize());
        str[len + cnt] = 0;

        delete[] this->data;
        this->data = str;
    }


    /*
     * vislib::String<T>::Clear
     */
    template<class T> void String<T>::Clear(void) {
        delete[] this->data;
        this->data = new Char[1];
        *this->data = 0;
    }


    /*
     * vislib::String<T>::Compare
     */
    template<class T>
    typename String<T>::Size String<T>::Compare(const Char *rhs, 
            const bool isCaseSensitive) const {
        const Char *lhs = this->data;
        Size retval = 0;

        if (rhs == NULL) {
            /* Assume NULL being less than everything else. */
            return 1;
        }

        if (isCaseSensitive) {
            while (((retval = *lhs - *rhs) == 0) && (*lhs != 0)) {
                lhs++;
                rhs++;
            }
        } else {
            while (((retval = T::ToLower(*lhs) - T::ToLower(*rhs)) == 0) 
                    && (*lhs != 0)) {
                lhs++;
                rhs++;
            }
        }

        if (retval < 0) {
            retval = -1;
        } else if (retval > 0) {
            retval = 1;
        }

        return retval;
    }


    /*
     * vislib::String<T>::Count
     */
    template<class T> 
    SIZE_T String<T>::Count(const Char c, const Size beginningAt) const {
        SIZE_T retval = 0;
        Size offset = beginningAt;

        while ((offset = this->Find(c, offset)) != INVALID_POS) {
            retval++;
            offset++;
        }

        return retval;
    }


    /*
     * vislib::String<T>::EndsWith
     */
    template<class T> bool String<T>::EndsWith(const Char c) const {
        Size len = this->Length();
        return (len > 0) ? (this->data[len - 1] == c) : false;
    }


    /*
     * vislib::String<T>::EndsWith
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
     * vislib::String<T>::Equals
     */
    template<class T>
    bool String<T>::Equals(const Char *rhs, const bool isCaseSensitive) const {
        const Char *lhs = this->data;

        if (rhs == NULL) {
            /* 
             * NULL pointer can never be equal as we have at least the terminating 
             * zero in our strings. 
             */
            return false;

        } else if (isCaseSensitive) {
            while ((*lhs == *rhs) && (*lhs != 0)) {
                lhs++;
                rhs++;
            }

            return (*lhs == *rhs);
        } else {
            while ((T::ToLower(*lhs) == T::ToLower(*rhs)) && (*lhs != 0)) {
                lhs++;
                rhs++;
            }

            return (T::ToLower(*lhs) == T::ToLower(*rhs));
        }
        ASSERT(false);
    }


    /*
     * vislib::String<T>::EscapeCharacters
     */
    template<class T> bool String<T>::EscapeCharacters(const Char ec, 
            const Char *normChars, const Char *escpdChars) {
        Size ncLen = T::SafeStringLength(normChars);

        // checking preconditions
        ASSERT(T::SafeStringLength(escpdChars) == ncLen);
#if (defined(DEBUG) || defined(_DEBUG))
        for (Size i = 0; i < ncLen; i++) {
            ASSERT(normChars[i] != ec);
            ASSERT(escpdChars[i] != ec);
            for (Size j = i + 1; j < ncLen; j++) {
                ASSERT(normChars[i] != normChars[j]);
                ASSERT(escpdChars[i] != escpdChars[j]);
            }
        }
#endif /* (defined(DEBUG) || defined(_DEBUG)) */

        // counting number of characters to be escaped
        unsigned int cntC2E = 0;
        Size dataLen = this->Length();
        for (Size i = 0; i < dataLen; i++) {
            for (Size j = 0; j < ncLen; j++) {
                if ((this->data[i] == normChars[j]) || (this->data[i] == ec)) {
                    cntC2E++;
                    break;
                }
            }
        }

        // escaping the characters
        Char *newData = new Char[dataLen + cntC2E + 1];
        cntC2E = 0;
        for (Size i = 0; i < dataLen; i++, cntC2E++) {
            newData[cntC2E] = this->data[i];
            if (this->data[i] == ec) {
                newData[++cntC2E] = ec;
            } else {
                for (Size j = 0; j < ncLen; j++) {
                    if (this->data[i] == normChars[j]) {
                        newData[cntC2E++] = ec;
                        newData[cntC2E] = escpdChars[j];
                        break;
                    }
                }
            }
        }
        newData[cntC2E] = 0;

        delete[] this->data;
        this->data = newData;

        return true;
    }


    /*
     * vislib::String<T>::Find
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
     * vislib::String<T>::Find
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
     * vislib::String<T>::FindLast
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
     * vislib::String<T>::FindLast
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
     * vislib::String<T>::Format
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
     * vislib::String<T>::Format
     */
    template<class T> void String<T>::Format(const String *fmt, ...) {
        // Implementation note: Using a reference as input to va_start is 
        // illegal according to the C++ standard. Therefore, we must use a
        // pointer here.
        if (fmt != NULL) {
            va_list argptr;
            va_start(argptr, fmt);
            this->FormatVa(fmt->PeekBuffer(), argptr);        
            va_end(argptr);
        } else {
            this->Clear();
        }
    }

    /*
     * vislib::String<T>::FormatVa
     */
    template<class T> void String<T>::FormatVa(const Char *fmt, 
            va_list argptr) {
        va_list argptrtmp;
        Size size = 0;

        /* Determine required buffer size. */
#undef ___local_arg_list_copyied
#ifndef _WIN32
#if defined(va_copy)
        va_copy(argptrtmp, argptr);
#define ___local_arg_list_copyied 1
#elif defined(__va_copy) /* va_copy */
        __va_copy(argptrtmp, argptr);
#define ___local_arg_list_copyied 1
#endif /* va_copy */
#endif /* !_WIN32 */
#ifndef ___local_arg_list_copyied
        argptrtmp = argptr;
#endif /* !___local_arg_list_copyied */
        size = T::Format(NULL, 0, fmt, argptrtmp);
#ifdef ___local_arg_list_copyied
        va_end(argptrtmp);
#endif /* ___local_arg_list_copyied */
#undef ___local_arg_list_copyied

        /* Allocate memory. */
        ASSERT(size >= 0);
        this->AllocateBuffer(size);
        
        /* Write the actual output. */
        T::Format(this->data, size + 1, fmt, argptr);
    }


    /*
     * vislib::String<T>::HashCode
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


    /*
     * vislib::String<T>::Insert
     */
    template<class T> void String<T>::Insert(const Size pos, const Char c) {
        Size len = this->Length();

        if ((pos >= 0) && (pos <= len)) {
            Char *str = new Char[len + 2];
            ::memcpy(str, this->data, pos * T::CharSize());
            str[pos] = c;
            ::memcpy(str + pos + 1, this->data + pos, (len - pos + 1) 
                * T::CharSize());

            delete[] this->data;
            this->data = str;
        } else {
            throw OutOfRangeException(pos, 0, static_cast<int>(len), __FILE__, 
                __LINE__);
        }
    }


    /*
     * vislib::String<T>::Insert
     */
    template<class T> void String<T>::Insert(const Size pos, const Char *str) {
        Size len = this->Length();
        Size strLen = 0;

        if ((pos >= 0) && (pos <= len)) {
            if ((strLen = T::SafeStringLength(str)) > 0) {
                Char *newData = new Char[len + strLen + 1];
                ::memcpy(newData, this->data, pos * T::CharSize());
                ::memcpy(newData + pos, str, strLen * T::CharSize());
                ::memcpy(newData + pos + strLen, this->data + pos, 
                    (len - pos + 1) * T::CharSize());

                delete[] this->data;
                this->data = newData;
            }
        } else {
            throw OutOfRangeException(pos, 0, static_cast<int>(len), __FILE__, 
                __LINE__);
        }
    }


    /*
     * vislib::String<T>::LevenshteinDistance
     */
    template<class T>
    typename String<T>::Size String<T>::LevenshteinDistance(const Char *rhs,
            const Size costAdd, const Size costDelete, 
            const Size costChange) const {
        // mueller: Adapted from wbcore. I am not completely sure whether this 
        // works ...
#define VL_LVS_ARY_IDX(i, j) ((i) * (len2 + 1) + (j))
#define VL_LVS_MIN3(a, b, c) (((a) < (b))\
    ? (((c) < (a)) ? (c) : (a))\
    : (((c) < (b)) ? (c) : (b)))

        ASSERT(rhs != NULL);

        Size len1 = this->Length();
        Size len2 = T::SafeStringLength(rhs);
        Size retval = static_cast<Size>(0);
        Size *dist = new Size[(len1 + 1) * (len2 + 1)];

        dist[0] = static_cast<Size>(0);
        
        for (Size i = 1; i <= len2; i++) {
            dist[i] = i * costAdd;
        }
   
        for (Size i = 1; i <= len1; i++) {
            dist[VL_LVS_ARY_IDX(i, 0)] = i * costDelete;
        }

        for (Size i = 1; i <= len1; i++) {
            for (Size j = 1; j <= len2; j++) {
                dist[VL_LVS_ARY_IDX(i, j)] = VL_LVS_MIN3(
                    dist[VL_LVS_ARY_IDX(i - 1, j - 1)] 
                    + ((this->data[i - 1] == rhs[j - 1]) ? 0 : costChange),
                    dist[VL_LVS_ARY_IDX(i, j - 1)] + costAdd,
                    dist[VL_LVS_ARY_IDX(i - 1, j)] + costDelete);
            }
        }

        retval = dist[VL_LVS_ARY_IDX(len1, len2)];
        ARY_SAFE_DELETE(dist);
        return retval;

#undef VL_LVS_MIN3
#undef VL_LVS_ARY_IDX
    }


#ifdef _WIN32
    /*
     * vislib::String<T>::Load
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
     * vislib::String<T>::Prepend
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
     * vislib::String<T>::Prepend
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
     * vislib::String<T>::Remove
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
     * vislib::String<T>::Replace
     */
    template<class T> typename String<T>::Size String<T>::Replace(
            const Char oldChar, const Char newChar, const Size limit) {
        Size retval = 0;
        Size len = this->Length();
        
        for (Size i = 0; (i < len) && ((limit < 0) || (retval < limit)); i++) {
            if (this->data[i] == oldChar) {
                this->data[i] = newChar;
                retval++;
            }
        }

        return retval;
    }


    /*
     * vislib::String<T>::Replace
     */
    template<class T> typename String<T>::Size String<T>::Replace(
            const Char *oldStr, const Char *newStr, const Size limit) {
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
        while (((nextStart = this->Find(oldStr, nextStart)) != INVALID_POS)
                && ((limit < 0) || (retval < limit))) {
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
        retval = 0;
        src = this->data;
        dst = str;
        while (((nextStart = this->Find(oldStr, nextStart)) != INVALID_POS)
                && ((limit < 0) || (retval < limit))) {
            while (src < this->data + nextStart) {
                *dst++ = *src++;
            }
            ::memcpy(dst, newStr, newLen * sizeof(Char));
            dst += newLen;
            src += oldLen;
            retval++;
            nextStart += oldLen;
        }
        while ((*dst++ = *src++) != 0);

        delete[] this->data;
        this->data = str;
        return retval;
    }


    /*
     * vislib::String<T>::Rot48
     */
    template<class T> void String<T>::Rot48(void) {
        for (Char *c = this->data; *c != 0; c++) {
            if ((*c > 31) && (*c < 128)) {
                *c = static_cast<Char>(((*c - 32 + 48) % 96) + 32);
            }
        }
    }


    /*
     * vislib::String<T>::StartsWith
     */
    template<class T> bool String<T>::StartsWith(const Char c,
            const bool isCaseSensitive) const {
        return isCaseSensitive 
            ? (this->data[0] == c) 
            : (T::ToLower(this->data[0]) == T::ToLower(c));
    }


    /*
     * vislib::String<T>::StartsWith
     */
    template<class T> bool String<T>::StartsWith(const Char *str,
            const bool isCaseSensitive) const {
        if (str != NULL) {
            const Char *s = str;
            const Char *d = this->data;

            if (isCaseSensitive) {
                while ((*s != 0) && (*s == *d)) {
                    s++;
                    d++;
                }
            } else {
                while ((*s != 0) && (T::ToLower(*s) == T::ToLower(*d))) {
                    s++;
                    d++;
                }
            }
            return (*s == 0);   // 's' must have been consumed completely.

        } else {
            /* Cannot start with 'str', if 'str' invalid. */
            return false;
        }
    }

    /*
     * vislib::String<T>::Substring
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
     * vislib::String<T>::Substring
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
     * vislib::String<T>::TrimSpacesBegin
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
     * vislib::String<T>::TrimSpacesEnd
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
     * vislib::String<T>::Truncate
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
     * vislib::String<T>::UnescapeCharacters
     */
    template<class T> bool String<T>::UnescapeCharacters(const Char ec, 
            const Char *normChars, const Char *escpdChars) {
        Size ncLen = T::SafeStringLength(normChars);

        // checking preconditions
        ASSERT(T::SafeStringLength(escpdChars) == ncLen);
#if (defined(DEBUG) || defined(_DEBUG))
        for (Size i = 0; i < ncLen; i++) {
            ASSERT(normChars[i] != ec);
            ASSERT(escpdChars[i] != ec);
            for (Size j = i + 1; j < ncLen; j++) {
                ASSERT(normChars[i] != normChars[j]);
                ASSERT(escpdChars[i] != escpdChars[j]);
            }
        }
#endif /* (defined(DEBUG) || defined(_DEBUG)) */

        // Checking for unrecognised escape sequences.
        Size dataLen = this->Length();
        bool ok;
        for (Size i = 0; i < dataLen; i++) {
            if (this->data[i] == ec) {
                i++;
                if (this->data[i] != ec) {
                    ok = false;
                    for (Size j = 0; j < ncLen; j++) {
                        if (this->data[i] == escpdChars[j]) {
                            ok = true;
                            break;
                        }
                    }
                    if (!ok) {
                        return false; // unrecognised escape sequence.
                    }
                }
            }
        }

        // unescaping the characters
        Size k = 0;
        for (Size i = 0; i < dataLen; i++, k++) {
            if (this->data[i] == ec) {
                i++;
                if (this->data[i] == ec) {
                    this->data[k] = ec;
                } else {
                    for (Size j = 0; j < ncLen; j++) {
                        if (this->data[i] == escpdChars[j]) {
                            this->data[k] = normChars[j];
                            break;
                        }
                    }
                }
            } else {
                this->data[k] = this->data[i];
            }
        }
        this->data[k] = 0;

        return true;
    }


    /*
     * vislib::String<T>::operator =
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
     * vislib::String<T>::operator =
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
     * vislib::String<T>::operator =
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
     * vislib::String<T>::operator =
     */
    template<class T> template<class U> 
    String<T>& String<T>::operator =(const U *rhs) {
        if (static_cast<void *>(this->data) != static_cast<const void *>(rhs)) {
            delete[] this->data;

            Size newLen = CharTraits<U>::SafeStringLength(rhs) + 1;
            this->data = new Char[newLen];

            CharTraits<U>::Convert(this->data, newLen, rhs);
        }

        return *this;
    }


    /*
     * vislib::String<T>::operator +
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
     * vislib::String<T>::operator +
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
     * vislib::String<T>::operator +=
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
     * vislib::String<T>::operator +=
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
     * vislib::String<T>::operator []
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
     * vislib::String<T>::operator []
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
