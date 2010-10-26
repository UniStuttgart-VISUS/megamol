/*
 * DirectoryIterator.h
 *
 * Copyright (C) 2007 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 */

#ifndef VISLIB_DIRECTORYITERATOR_H_INCLUDED
#define VISLIB_DIRECTORYITERATOR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/CharTraits.h"
#include "vislib/DirectoryEntry.h"
#include "vislib/String.h"
#include "vislib/Iterator.h"
#include "vislib/IOException.h"
#include "vislib/NoSuchElementException.h"
#include "vislib/SystemException.h"
#include "vislib/Path.h"
#include "vislib/UnsupportedOperationException.h"

#ifdef _WIN32
#include <Windows.h>
#else /* _WIN32 */
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <iostream>
#include "vislib/File.h"
#include "vislib/StringConverter.h"
#include "vislib/sysfunctions.h"
#endif /* _WIN32 */


namespace vislib {
namespace sys {


    /**
     * Instances of this class let the user enumerate all files/subdirectories in
     * a certain path. In contrast to the OS-dependent implementations, "." and ".."
     * are omitted.
     * Template type T is the CharTraits type
     */
    template<class T> class DirectoryIterator
            : public vislib::Iterator<DirectoryEntry<T> > {
    public:

        /** Alias for correct char type */
        typedef typename T::Char Char;

        /** Alias for correct entry type */
        typedef DirectoryEntry<T> Entry;

        /**
         * Ctor
         *
         * @param path The path which this Iterator will enumerate
         * @param isPattern If false 'path' specifies the path to search in;
         *                  If true 'path' specifies a file globbing pattern
         * @param showDirs If false no directories will be iterated, only
         *                 files; If true files and directories will be
         *                 iterated except for the two directories '.' and
         *                 '..'
         */
        DirectoryIterator(const Char* path, bool isPattern = false,
            bool showDirs = true);

        /** Dtor */
        virtual ~DirectoryIterator(void);

        /** Behaves like Iterator<T>::HasNext */
        virtual bool HasNext(void) const;

        /** 
         * Behaves like Iterator<T>::Next 
         *
         * @throws NoSuchElementException if there is no next element
         */
        virtual Entry& Next(void);

    private:

        /**
         * Forbidden copy-ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        inline DirectoryIterator(const DirectoryIterator& rhs) {
            throw UnsupportedOperationException(
                "vislib::sys::DirectoryIterator::DirectoryIterator",
                __FILE__, __LINE__);
        }

        /**
         * Forbidden assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         *
         * @throws IllegalParamException If &'rhs' != this.
         */
        inline DirectoryIterator& operator=(const DirectoryIterator& rhs) {
            if (this != &rhs) {
                throw IllegalParamException("rhs", __FILE__, __LINE__);
            }
        }

        /**
         * Fetches the next item in the iteration
         */
        void fetchNextItem(void);

        /** The next element */
        Entry nextItem;

        /** The current element */
        Entry currentItem;

        /** Flag to omit folders in enumeration */
        bool omitFolders;

#ifdef _WIN32

        /** Handle to the file find search */
        HANDLE findHandle;

#else /* _WIN32 */

        /** The directory stream of the directory to iterate */
        DIR *dirStream;

        /** The base path */
        StringA basePath;

        /** The file globbing pattern */
        StringA pattern;

#endif /* _WIN32 */

    };/* end class Iterator */


    /*
     * DirectoryIterator<T>::DirectoryIterator
     */
    template<class T> DirectoryIterator<T>::DirectoryIterator(
            const Char* path, bool isPattern, bool showDirs) : nextItem(),
            currentItem(), omitFolders(!showDirs) {
        // We won't find anything for this type!
        throw UnsupportedOperationException(
             "DirectoryIterator<T>::DirectoryIterator", __FILE__, __LINE__);
    }


    /*
     * DirectoryIterator<CharTraitsA>::DirectoryIterator
     */
    template<> DirectoryIterator<CharTraitsA>::DirectoryIterator(
            const Char* path, bool isPattern, bool showDirs);


    /*
     * DirectoryIterator<CharTraitsW>::DirectoryIterator
     */
    template<> DirectoryIterator<CharTraitsW>::DirectoryIterator(
            const Char* path, bool isPattern, bool showDirs);


    /*
     * DirectoryIterator<T>::~DirectoryIterator
     */
    template<class T> DirectoryIterator<T>::~DirectoryIterator(void) {
#ifdef _WIN32
        if (this->findHandle != INVALID_HANDLE_VALUE) {
            FindClose(this->findHandle);
        }
#else /* _WIN32 */
        if (this->dirStream != NULL) {
            closedir(dirStream);
        }
#endif /* _WIN32 */
    }


    /*
     * DirectoryIterator<T>::HasNext
     */
    template<class T> bool DirectoryIterator<T>::HasNext(void) const {
        return !this->nextItem.Path.IsEmpty();
    }


    /*
     * DirectoryIterator<T>::Next
     */
    template<class T>
    typename DirectoryIterator<T>::Entry& DirectoryIterator<T>::Next(void) {
        this->currentItem = this->nextItem;
        this->fetchNextItem();
        if (this->currentItem.Path.IsEmpty()) {
            throw NoSuchElementException("No next element.", __FILE__, __LINE__);
        }
        return this->currentItem;

    }


    /*
     * DirectoryIterator<T>::fetchNextItem
     */
    template<class T> void DirectoryIterator<T>::fetchNextItem(void) {
        // We won't find anything for this type!
        throw UnsupportedOperationException(
            "DirectoryIterator<T>::fetchNextItem", __FILE__, __LINE__);
    }


    /*
     * DirectoryIterator<CharTraitsA>::fetchNextItem
     */
    template<> void DirectoryIterator<CharTraitsA>::fetchNextItem(void);


    /*
     * DirectoryIterator<CharTraitsW>::fetchNextItem
     */
    template<> void DirectoryIterator<CharTraitsW>::fetchNextItem(void);


    /** Template instantiation for ANSI char DirectoryIterator. */
    typedef DirectoryIterator<CharTraitsA> DirectoryIteratorA;

    /** Template instantiation for wide char DirectoryIterator. */
    typedef DirectoryIterator<CharTraitsW> DirectoryIteratorW;

    /** Template instantiation for TCHAR DirectoryIterator. */
    typedef DirectoryIterator<TCharTraits> TDirectoryIterator;


} /* end namespace sys */
} /* end namespace vislib */


//#include "vislib/DirectoryIterator.inl"

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_DIRECTORYITERATOR_H_INCLUDED */
