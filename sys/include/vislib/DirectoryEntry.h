/*
 * DirectoryEntry.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_DIRECTORYENTRY_H_INCLUDED
#define VISLIB_DIRECTORYENTRY_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/CharTraits.h"
#include "vislib/String.h"


namespace vislib {
namespace sys {


    /**
     * Utility class used by DirectoryIterator to store a single entry
     * Template type T is the CharTraits type
     */
    template<class T> class DirectoryEntry {
    public:

        /** Enum for the currently supported types of DirectoryEntry */
        enum EntryType {
            DIRECTORY,
            FILE
        };

        /** Ctor. */
        DirectoryEntry(void);

        /**
         * Ctor.
         *
         * @param path The name of the entry
         * @param type The type of the entry
         */
        DirectoryEntry(const vislib::String<T>& path, EntryType& type);

        /**
         * Copy ctor.
         *
         * @param src The object to clone from
         */
        DirectoryEntry(const DirectoryEntry& src);

        /** Dtor. */
        ~DirectoryEntry(void);

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         */
        DirectoryEntry& operator=(const DirectoryEntry& rhs);

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         *
         * @return True if this and rhs are equal
         */
        bool operator==(const DirectoryEntry& rhs) const;

        /** Name of the entry */
        String<T> Path;

        /** Type of the entry */
        EntryType Type;

    };


    /*
     * DirectoryEntry<T>::DirectoryEntry
     */
    template<class T> DirectoryEntry<T>::DirectoryEntry(void)
            : Path(), Type(DirectoryEntry::FILE) {
        // intentionally empty
    }


    /*
     * DirectoryEntry<T>::DirectoryEntry
     */
    template<class T> DirectoryEntry<T>::DirectoryEntry(
            const vislib::String<T>& path, EntryType& type)
            : Path(path), Type(type) {
        // intentionally empty
    }


    /*
     * DirectoryEntry<T>::DirectoryEntry
     */
    template<class T> DirectoryEntry<T>::DirectoryEntry(
            const DirectoryEntry& src) : Path(src.Path), Type(src.Type) {
        // intentionally empty
    }


    /*
     * DirectoryEntry<T>::~DirectoryEntry
     */
    template<class T> DirectoryEntry<T>::~DirectoryEntry(void) {
        // intentionally empty
    }


    /*
     * DirectoryEntry<T>::operator=
     */
    template<class T> DirectoryEntry<T>& DirectoryEntry<T>::operator=(
            const DirectoryEntry<T>& rhs) {
        this->Path = rhs.Path;
        this->Type = rhs.Type;
        return *this;
    }


    /*
     * DirectoryEntry<T>::operator==
     */
    template<class T> bool DirectoryEntry<T>::operator==(
            const DirectoryEntry<T>& rhs) const {
        return (this->Path == rhs.Path)
            && (this->Type == rhs.Type);
    }


    /** Template instantiation for ANSI char DirectoryEntry. */
    typedef DirectoryEntry<CharTraitsA> DirectoryEntryA;

    /** Template instantiation for wide char DirectoryEntry. */
    typedef DirectoryEntry<CharTraitsW> DirectoryEntryW;

    /** Template instantiation for TCHAR DirectoryEntrys. */
    typedef DirectoryEntry<TCharTraits> TDirectoryEntry;

} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_DIRECTORYENTRY_H_INCLUDED */

