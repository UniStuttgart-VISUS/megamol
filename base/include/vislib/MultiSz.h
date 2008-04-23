/*
 * MultiSz.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_MULTISZ_H_INCLUDED
#define VISLIB_MULTISZ_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include <cstdarg>

#include "vislib/String.h"


namespace vislib {


    /**
     * Represents a set of zero-terminated strings terminated by an additional 
     * zero. The class can be instantiated for vislib::CharTraits subclasses 
     * that implement all the required operations.
     *
     * Note that the set cannot contain empty strings as this would corrupt the
     * data structure.
     */
    template<class T> class MultiSz {

    public:

        /** Define a local name for the character type. */
        typedef typename T::Char Char;

        /** Define a local name for the string size. */
        typedef typename T::Size Size;

        /**
         * Answer the number of entries in the set of zero-terminated strings
         * 'multiSz'. Note that is is indispensable that 'multiSz' is terminated
         * using TWO zeros. It is safe to pass a NULL pointer.
         *
         * @param multiSz A set of zero-terminated strings, terminated with TWO
         *                zeros at the end of the set. The caller remains owner
         *                of the memory.
         *
         * @return The number of strings in the set.
         */
        static SIZE_T Count(const Char *multiSz);

        /**
         * Answer the length of the set of zero-terminated strings 'multiSz' 
         * including all terminating zeros. Note that is is indispensable that 
         * 'multiSz' is terminated using TWO zeros. It is safe to pass a NULL 
         * pointer.
         *
         * @param multiSz A set of zero-terminated strings, terminated with TWO
         *                zeros at the end of the set. The caller remains owner
         *                of the memory.
         * 
         * @return The number of characters that are required to store the
         *         'multiSz'.
         */
        static SIZE_T Length(const Char *multiSz);

        /**
         * Answer the 'idx' string in the set of zero-terminated string 
         * 'multiSz' or NULL if 'idx' is out of range.  Note that is is 
         * indispensable that 'multiSz' is terminated using TWO zeros. It 
         * is safe to pass a NULL pointer.
         *
         * @param idx     The index of the string to retrieve.
         * @param multiSz A set of zero-terminated strings to get the 'idx'th
         *                element from. The caller remains owner of the memory.
         *
         * @return A pointer to the begin of the 'idx'th string or NULL, if
         *         'idx' is out of range.
         */
        static const Char *PeekAt(const SIZE_T idx, const Char *multiSz);

        /**
         * Create a new object using the the set of zero-terminated strings
         * 'multiSz' for initialisation. Note that is is indispensable that 
         * 'multiSz' is terminated using TWO zeros. It is safe to pass a NULL 
         * pointer.
         *
         * @param multiSz The set of zero-terminated strings used for 
         *                initialisation. If this is a NULL pointer, an empty
         *                MultiSz is created. The caller remains owner of the
         *                memory.
         *
         * @throws std::bad_alloc If the memory required for the MultiSz could
         *                        not be allocated.
         */
        explicit MultiSz(const Char *multiSz = NULL);

        /**
         * Create a new object from an array of strings.
         *
         * @param string     An array of strings. It is safe to pass a NULL
         *                   pointer. The caller remains owner of the memory.
         * @param cntStrings The number of elements in 'strings'.
         *
         * @throws std::bad_alloc If the memory required for the MultiSz could
         *                        not be allocated.
         */
        MultiSz(const Char **strings, const SIZE_T cntStrings);

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         *
         * @throws std::bad_alloc If the memory required for the MultiSz could
         *                        not be allocated.
         */
        MultiSz(const MultiSz& rhs);

        /** Dtor. */
        ~MultiSz(void);

        /**
         * Append 'str' at the end of the MultiSz.
         *
         * @param str The string to be appended. It is safe to pass a NULL
         *            pointer (nothing will happen in this case). Nothing will
         *            happen if 'str' is an empty string. The caller remains 
         *            owner of the memory.
         *
         * @throws std::bad_alloc If the memory required for the MultiSz could
         *                        not be allocated.
         */
        void Append(const Char *str);

        /**
         * Append 'str' at the end of the MultiSz.
         *
         * @param str The string to be appended. If the string is empty, nothing
         *            will happen.
         *
         * @throws std::bad_alloc If the memory required for the MultiSz could
         *                        not be allocated.
         */
        inline void Append(const String<T>& str) {
            this->Append(str.PeekBuffer());
        }

        /**
         * Deletes the current content of the MultiSz, allocates memory for 
         * 'cnt' characters, and returns a pointer to this buffer. The buffer
         * will not be initialised.
         *
         * Note: The behaviour of this method is different from the behaviour
         * of the vislib::String! You must specify the memory for the 
         * terminating zeros yourself!
         *
         * @param cnt The new size of the buffer. This will be the exact new
         *            size, no space for zeros will be added!
         *
         * @return The pointer to the new internal buffer.
         *
         * @throws std::bad_alloc If the memory requested amount of memory could
         *                        not be allocated.
         */
        Char *AllocateBuffer(const SIZE_T cnt);

        /**
         * Clear all elements in the MultiSz.
         */
        inline void Clear(void) {
            ARY_SAFE_DELETE(this->data);
            ASSERT(this->data == NULL);
        }

        /**
         * Answer the number of strings in the MultiSz.
         *
         * @return The number of strings in the MultiSz.
         */
        inline SIZE_T Count(void) const {
            return MultiSz::Count(this->data);
        }

        /**
         * Answer the string at the 'idx'th position in the MultiSz.
         *
         * @param idx The index of the string to retrieve.
         *
         * @return The string at the requested position.
         *
         * @throws OutOfRangeException If 'idx' is not within [0, Count()[.
         */
        String<T> GetAt(const SIZE_T idx) const {
            return (*this)[idx];
        }

        /**
         * Insert 'str' at the 'idx'th position. All elements behind
         * 'idx' are shifted one element right. 'idx' must be a valid index in 
         * the MultiSz or the index directly following the end, i. e. Count(). 
         * In the latter case, the method behaves like Append().
         *
         * @param idx The position to insert the string at.
         * @param str The string to add. The caller remains owner of the memory.
         *            It is safe to pass a NULL pointer. Nothing will happen
         *            in the latter case. Nothing will happen, too, if the string 
         *            is empty.
         *
         * @throws OutOfRangeException If 'idx' is not within 
         *                             [0, this->Count()].
         * @throws std::bad_alloc If the memory required for the MultiSz could
         *                        not be allocated.
         */
        void Insert(const SIZE_T idx, const Char *str);

        /**
         * Insert 'str' at the 'idx'th position. All elements behind
         * 'idx' are shifted one element right. 'idx' must be a valid index in 
         * the MultiSz or the index directly following the end, i. e. Count(). 
         * In the latter case, the method behaves like Append().
         *
         * @param idx The position to insert the string at.
         * @param str The string to add. Nothing will happen if the string is 
         *            empty.
         *
         * @throws OutOfRangeException If 'idx' is not within 
         *                             [0, this->Count()].
         * @throws std::bad_alloc If the memory required for the MultiSz could
         *                        not be allocated.
         */
        inline void Insert(const SIZE_T idx, const String<T>& str) {
            this->Insert(idx, str.PeekBuffer());
        }

        /**
         * Answer whether there is no string at all in the MultiSz.
         *
         * @return true if the set is empty, false otherwise.
         */
        inline bool IsEmpty(void) const {
            return (this->data == NULL);
        }

        /**
         * Answer the length of the MultiSz including all terminating zeros.
         * 
         * @return The number of characters that are required to store the
         *         MultiSz.
         */
        inline SIZE_T Length(void) const {
            return MultiSz::Length(this->data);
        }

        /**
         * Answer the raw data of the 'idx'th string in the MultiSz or NULL
         * if no such element exists.
         *
         * @param idx The index of the string to get.
         *
         * @return A pointer to the begin of the 'idx'th string or NULL, if the
         *         index is out of range.
         */
        inline const Char *PeekAt(const SIZE_T idx) const {
            return MultiSz::PeekAt(idx, this->data);
        }

        /**
         * Answer the raw data pointer. Note, that is unsafe to operate with 
         * this pointer and that the pointer might become invalid as the
         * MultiSz is manipulated using other methods.
         *
         * @return The raw data pointer.
         */
        inline const Char *PeekBuffer(void) const {
            return this->data;
        }

        /**
         * Remove all occurrences of 'str' in the set.
         *
         * @param str The string to be removed. It is safe to pass a NULL
         *            pointer or an empty string. The caller remains owner
         *            of the memory.
         */
        void Remove(const Char *str);

        /**
         * Remove all occurrences of 'str' in the set.
         *
         * @param str The string to be removed. It is safe to pass an empty 
         *            string. 
         */
        inline void Remove(const String<T>& str) {
            this->Remove(str.PeekBuffer());
        }

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        MultiSz& operator =(const MultiSz& rhs);

        /**
         * Assignment from a raw set of zero-terminated strings 'rhs'. Note 
         * that is is indispensable that 'rhs' is terminated using TWO zeros. 
         * It is safe to pass a NULL pointer.
         *
         * @param rhs The right hand side operand, which must be a set of 
         *            zero-terminated strings, terminated with TWO zeros at 
         *            the end of the set. The caller remains owner of the 
         *            memory.
         *
         * @return *this.
         */
        MultiSz& operator =(const Char *rhs);

        /**
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if 'rhs' and this object are equal, false otherwise.
         */
        bool operator ==(const MultiSz& rhs) const;

        /**
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if 'rhs' and this object are  not equal, 
         *         false otherwise.
         */
        inline bool operator !=(const MultiSz& rhs) const {
            return !(*this == rhs);
        }

        /**
         * Answer the string at the 'idx'th position in the MultiSz.
         *
         * @param idx The index of the string to retrieve.
         *
         * @return The string at the requested position.
         *
         * @throws OutOfRangeException If 'idx' is not within [0, Count()[.
         */
        String<T> operator [](const SIZE_T idx) const;

    private:

        /** The raw data. */
        Char *data;

    };


    /*
     * MultiSz<T>::Count
     */
    template<class T> SIZE_T MultiSz<T>::Count(const Char *multiSz) {
        SIZE_T retval = 0;
        const Char *cursor = multiSz;

        if (cursor != NULL) {
            while (*cursor != 0) {
                while (*cursor++ != 0);
                retval++;
            }
        }

        return retval;
    }


    /*
     * MultiSz<T>::Length
     */
    template<class T> SIZE_T MultiSz<T>::Length(const Char *multiSz) {
        const Char *cursor = multiSz;

        if (cursor != NULL) {
            while ((*(++cursor - 1) != 0) || (*cursor != 0));
            return (cursor - multiSz + 1);
        } else {
            return 0;
        }
    }


    /* 
     * MultiSz<T>::PeekAt
     */
    template<class T> 
    const typename MultiSz<T>::Char *MultiSz<T>::PeekAt(const SIZE_T idx, 
            const Char *multiSz) {
        const Char *cursor = multiSz;

        if (cursor != NULL) {
            for (SIZE_T i = 0; (i < idx) && (*cursor != 0); i++) {
                while (*cursor++ != 0);
            }

            if (*cursor == 0) {
                cursor = NULL;
            }
        }

        return cursor;
    }


    /*
     * MultiSz<T>::MultiSz
     */
    template<class T> 
    MultiSz<T>::MultiSz(const Char *multiSz) : data(NULL) {
        *this = multiSz;
    }


    /*
     * MultiSz<T>::MultiSz
     */
    template<class T> 
    MultiSz<T>::MultiSz(const Char **strings, const SIZE_T cntStrings) 
            : data(NULL) {
        if ((strings != NULL) && (cntStrings > 0)) {
            SIZE_T cnt = 0;
            Char *dstPos = NULL;
            const Char *srcPos = NULL;

            for (SIZE_T i = 0; i < cntStrings; i++) {
                cnt += T::SafeStringLength(strings[i]) + 1;
            }

            this->data = dstPos = new Char[cnt + 1];
            
            for (SIZE_T i = 0; i < cntStrings; i++) {
                if ((strings[i] != NULL) && (*(strings[i]) != 0)) {
                    srcPos = strings[i];
                    while ((*dstPos++ = *srcPos++) != 0);
                }
            }

            /* Add double-zero at end. */
            *dstPos = 0;
        } /* end if ((strings != NULL) && (cntStrings > 0)) */
    }


    /*
     * vislib::MultiSz<T>::MultiSz
     */
    template<class T> MultiSz<T>::MultiSz(const MultiSz& rhs) : data(NULL) {
        *this = rhs;
    }


    /*
     * vislib::MultiSz<T>::~MultiSz
     */
    template<class T> MultiSz<T>::~MultiSz(void) {
        this->Clear();
    }


    /*
     * vislib::MultiSz<T>::Append
     */
    template<class T> void MultiSz<T>::Append(const Char *str) {
        if ((str != NULL) && (*str != 0)) {
            SIZE_T oldLen = MultiSz::Length(this->data);
            SIZE_T strLen = T::SafeStringLength(str) + 1;
            Char *newData = NULL;
            ASSERT(strLen > 0);

            if (oldLen > 0) {
                newData = new Char[oldLen + strLen];
                ::memcpy(newData, this->data, oldLen * sizeof(Char));
                ::memcpy(newData + oldLen - 1, str, strLen * sizeof(Char));
                newData[oldLen + strLen - 1] = 0;
            } else {
                newData = new Char[strLen + 1];
                ::memcpy(newData, str, strLen * sizeof(Char));
                newData[strLen] = 0;
            } 
            
            ARY_SAFE_DELETE(this->data);
            this->data = newData;
        }
    }


    /*
     * vislib::MultiSz<T>::AllocateBuffer
     */
    template<class T> 
    typename MultiSz<T>::Char *MultiSz<T>::AllocateBuffer(const SIZE_T cnt) {
        ARY_SAFE_DELETE(this->data);
        if (cnt > 0) {
            this->data = new Char[cnt];
        }
        return this->data;
    }


    /*
     * vislib::MultiSz<T>::Insert
     */
    template<class T> 
    void MultiSz<T>::Insert(const SIZE_T idx, const Char *str) {
        SIZE_T oldCnt = MultiSz::Count(this->data);

        if (idx == oldCnt) {
            this->Append(str);

        } else if (idx < oldCnt) {
            if ((str != NULL) && (*str != 0)) {
                SIZE_T oldLen = MultiSz::Length(this->data);
                SIZE_T strLen = T::SafeStringLength(str) + 1;
                SIZE_T offset = this->PeekAt(idx) - this->data;
                Char *newData = NULL;
                ASSERT(oldLen > 0);
                ASSERT(strLen > 0);
                
                newData = new Char[oldLen + strLen];
                ::memcpy(newData, this->data, offset * sizeof(Char));
                ::memcpy(newData + offset, str, strLen * sizeof(Char));
                ::memcpy(newData + offset + strLen, this->data + offset, 
                    (oldLen - offset) * sizeof(Char));
                ASSERT(newData[oldLen + strLen - 1] == 0);
                ASSERT(newData[oldLen + strLen - 2] == 0);

                ARY_SAFE_DELETE(this->data);
                this->data = newData;
            }
        } else {
            throw OutOfRangeException(static_cast<int>(idx), 0, 
                static_cast<int>(oldCnt), __FILE__, __LINE__);
        }
    }


    /**
     *vislib::MultiSz<T>::Remove
     */
    template<class T> void MultiSz<T>::Remove(const Char *str) {
        if ((str != NULL) && (*str != 0) && (this->data != NULL)) {
            SIZE_T cnt = 0;
            SIZE_T newLen = 0;
            SIZE_T oldLen = MultiSz::Length(this->data);
            SIZE_T strLen = T::SafeStringLength(str) + 1;
            const Char *cursor = this->data;
            const Char *s = NULL;
            const Char *strStart = NULL;
            Char *newData = NULL;
            Char *insPos = NULL;
            
            

            /* Count occurrences of string to be removed. */
            while (*cursor != 0) {
                s = str;
                while ((*s == *cursor) && (*cursor != 0)) {
                    s++;
                    cursor++;
                }
                if (*s == *cursor) {
                    cursor++;
                    cnt++;
                } else {
                    /* Consume the rest of the string. */
                    while (*cursor++ != 0);
                }
            }

            /* Reallocate and copy non-removed elements. */
            if ((newLen = oldLen - cnt * strLen) > 2) {
                newData = insPos = new Char[newLen];
                cursor = this->data;

                while (*cursor != 0) {
                    // TODO: Das könnte man mit einer one-pass-Lösung noch
                    // chefmäßiger machen.
                    s = str;
                    strStart = cursor;
                    while ((*s == *cursor) && (*cursor != 0)) {
                        s++;
                        cursor++;
                    }
                    if (*s != *cursor) {
                        /* No match, copy to new list. */
                        cursor = strStart;
                        while ((*insPos++ = *cursor++) != 0);
                    } else {
                        /* Skip trailing zero of active string. */
                        cursor++;
                    }
                }

                newData[newLen - 1] = 0;    // Enfore double-zero.
                ASSERT(newData[newLen - 1] == 0);
                ASSERT(newData[newLen - 2] == 0);

                ARY_SAFE_DELETE(this->data);
                this->data = newData;
            } else {
                /* Remove everything. */
                ARY_SAFE_DELETE(this->data);
                ASSERT(this->data == NULL);
            } /* end if ((newLen = oldLen - cnt * strLen) > 2) */
        } /* end if ((str != NULL) && (*str != 0) && (this->data != NULL)) */
    }


    /*
     * vislib::MultiSz<T>::operator =
     */
    template<class T> MultiSz<T>& MultiSz<T>::operator =(const MultiSz& rhs) {
        if (this != &rhs) {
            this->Clear();
            SIZE_T cnt = rhs.Length();
            if (cnt > 0) {
                this->data = new Char[cnt];
                ::memcpy(this->data, rhs.data, cnt * sizeof(Char));
            }
        }

        return *this;
    }


    /*
     * vislib::MultiSz<T>::operator =
     */
    template<class T> MultiSz<T>& MultiSz<T>::operator =(const Char *rhs) {
        this->Clear();

        if (rhs != NULL) {
            SIZE_T cnt = MultiSz::Length(rhs);
            ASSERT(cnt > 0);
            this->data = new Char[cnt];
            ::memcpy(this->data, rhs, cnt * sizeof(Char));
        }

        return *this;
    }


    /*
     * vislib::MultiSz<T>::operator ==
     */
    template<class T> bool MultiSz<T>::operator ==(const MultiSz& rhs) const {
        const Char *l = this->data;
        const Char *r = rhs.data;

        if ((l != NULL) && (r != NULL)) {
            while ((*l != 0) && (*r != 0)) {
                while (*l++ == *r++);
            }
    
            return (*l == *r);
        } else {
            /* 'l' and 'r' must both be NULL for being equal. */
            return (l == r);
        }
    }


    /*
     * vislib::MultiSz<T>::operator []
     */
    template<class T> 
    String<T> MultiSz<T>::operator [](const SIZE_T idx) const {
        const Char *tmp = MultiSz::PeekAt(idx, this->data);
        if (tmp != NULL) {
            return String<T>(tmp);
        } else {
            throw OutOfRangeException(static_cast<int>(idx), 0, 
                static_cast<int>(this->Count()) - 1, __FILE__, __LINE__);
        }
    }


    /** Template instantiation for ANSI strings. */
    typedef MultiSz<CharTraitsA> MultiSzA;

    /** Template instantiation for wide strings. */
    typedef MultiSz<CharTraitsW> MultiSzW;

    /** Template instantiation for TCHARs. */
    typedef MultiSz<TCharTraits> TMultiSz;
    
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_MULTISZ_H_INCLUDED */
