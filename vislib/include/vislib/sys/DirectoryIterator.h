/*
 * DirectoryIterator.h
 *
 * Copyright (C) 2007 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 */

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/CharTraits.h"
#include "vislib/Iterator.h"
#include "vislib/NoSuchElementException.h"
#include "vislib/String.h"
#include "vislib/UnsupportedOperationException.h"
#include "vislib/sys/DirectoryEntry.h"
#include "vislib/sys/IOException.h"
#include "vislib/sys/Path.h"
#include "vislib/sys/SystemException.h"

#ifdef _WIN32
#include <Windows.h>
#undef min
#undef max
#else /* _WIN32 */
#include "vislib/StringConverter.h"
#include "vislib/sys/File.h"
#include "vislib/sys/sysfunctions.h"
#include <dirent.h>
#include <errno.h>
#include <iostream>
#include <sys/types.h>
#endif /* _WIN32 */


namespace vislib::sys {


/**
 * Instances of this class let the user enumerate all files/subdirectories in
 * a certain path. In contrast to the OS-dependent implementations, "." and ".."
 * are omitted.
 * Template type T is the CharTraits type
 */
template<class T>
class DirectoryIterator : public vislib::Iterator<DirectoryEntry<T>> {
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
    DirectoryIterator(const Char* path, bool isPattern = false, bool showDirs = true);

    /** Dtor */
    ~DirectoryIterator() override;

    /** Behaves like Iterator<T>::HasNext */
    bool HasNext() const override;

    /**
     * Behaves like Iterator<T>::Next
     *
     * @throws NoSuchElementException if there is no next element
     */
    Entry& Next() override;

private:
    /**
     * Forbidden copy-ctor.
     *
     * @param rhs The object to be cloned.
     *
     * @throws UnsupportedOperationException Unconditionally.
     */
    inline DirectoryIterator(const DirectoryIterator& rhs) {
        throw UnsupportedOperationException("vislib::sys::DirectoryIterator::DirectoryIterator", __FILE__, __LINE__);
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
    void fetchNextItem();

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
    DIR* dirStream;

    /** The base path */
    StringA basePath;

    /** The file globbing pattern */
    StringA pattern;

#endif /* _WIN32 */

}; /* end class Iterator */


/*
 * DirectoryIterator<T>::DirectoryIterator
 */
template<class T>
DirectoryIterator<T>::DirectoryIterator(const Char* path, bool isPattern, bool showDirs)
        : nextItem()
        , currentItem()
        , omitFolders(!showDirs) {
    // We won't find anything for this type!
    throw UnsupportedOperationException("DirectoryIterator<T>::DirectoryIterator", __FILE__, __LINE__);
}


/*
 * DirectoryIterator<CharTraitsA>::DirectoryIterator
 */
template<>
DirectoryIterator<CharTraitsA>::DirectoryIterator(const Char* path, bool isPattern, bool showDirs);


/*
 * DirectoryIterator<CharTraitsW>::DirectoryIterator
 */
template<>
DirectoryIterator<CharTraitsW>::DirectoryIterator(const Char* path, bool isPattern, bool showDirs);


/*
 * DirectoryIterator<T>::~DirectoryIterator
 */
template<class T>
DirectoryIterator<T>::~DirectoryIterator() {
#ifdef _WIN32
    if (this->findHandle != INVALID_HANDLE_VALUE) {
        FindClose(this->findHandle);
    }
#else  /* _WIN32 */
    if (this->dirStream != NULL) {
        closedir(dirStream);
    }
#endif /* _WIN32 */
}


/*
 * DirectoryIterator<T>::HasNext
 */
template<class T>
bool DirectoryIterator<T>::HasNext() const {
    return !this->nextItem.Path.IsEmpty();
}


/*
 * DirectoryIterator<T>::Next
 */
template<class T>
typename DirectoryIterator<T>::Entry& DirectoryIterator<T>::Next() {
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
template<class T>
void DirectoryIterator<T>::fetchNextItem() {
    // We won't find anything for this type!
    throw UnsupportedOperationException("DirectoryIterator<T>::fetchNextItem", __FILE__, __LINE__);
}


/*
 * DirectoryIterator<CharTraitsA>::fetchNextItem
 */
template<>
void DirectoryIterator<CharTraitsA>::fetchNextItem();


/*
 * DirectoryIterator<CharTraitsW>::fetchNextItem
 */
template<>
void DirectoryIterator<CharTraitsW>::fetchNextItem();


/** Template instantiation for ANSI char DirectoryIterator. */
typedef DirectoryIterator<CharTraitsA> DirectoryIteratorA;

/** Template instantiation for wide char DirectoryIterator. */
typedef DirectoryIterator<CharTraitsW> DirectoryIteratorW;

/** Template instantiation for TCHAR DirectoryIterator. */
typedef DirectoryIterator<TCharTraits> TDirectoryIterator;


} // namespace vislib::sys


//#include "vislib/DirectoryIterator.inl"

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
