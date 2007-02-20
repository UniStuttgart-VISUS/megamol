/*
 * DirectoryIterator.h
 *
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_DIRECTORYITERATOR_H_INCLUDED
#define VISLIB_DIRECTORYITERATOR_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/CharTraits.h"
#include "vislib/String.h"
#include "vislib/Iterator.h"
#include "vislib/IOException.h"
#include "vislib/NoSuchElementException.h"
#include "vislib/SystemException.h"
#include "vislib/Path.h"

#ifndef _WIN32
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <iostream>

#include "vislib/File.h"
#include "vislib/StringConverter.h"
#endif /* _WIN32 */


namespace vislib {
namespace sys {

	template<class T> class DirectoryEntry {
		typedef typename T::Char Char;
	public:
		enum EntryType {
			DIRECTORY,
			FILE
		};
		String<T> Path;
		EntryType Type;
	};
 

    /** Template instantiation for ANSI char DirectoryEntry. */
    typedef DirectoryEntry<CharTraitsA> DirectoryEntryA;

    /** Template instantiation for wide char DirectoryEntry. */
    typedef DirectoryEntry<CharTraitsW> DirectoryEntryW;

    /** Template instantiation for TCHAR DirectoryEntrys. */
    typedef DirectoryEntry<TCharTraits> TDirectoryEntry;


    /**
     * Instances of this class let the user enumerate all files/subdirectories in
	 * a certain path. In contrast to the OS-dependent implementations, "." and ".."
	 * are omitted.
     */
	template<class T> class DirectoryIterator
            : public vislib::Iterator<DirectoryEntry<T> > {
	};/* end class Iterator */


    /**
     * Instances of this class let the user enumerate all files/subdirectories in
	 * a certain path. In contrast to the OS-dependent implementations, "." and ".."
	 * are omitted. For Unicode support, use DirectoryIterator<CharTraitsW>.
     */
	template<> class DirectoryIterator<CharTraitsA>
            : public vislib::Iterator<DirectoryEntry<CharTraitsA> > {
        typedef CharTraitsA::Char Char;
		typedef DirectoryEntry<CharTraitsA> Item;
	public:

		/** Ctor.
		 *
		 * @param path the path which this Iterator will enumerate.
		 *
		 * @throws SystemException if iterating the directory fails
		 */
		DirectoryIterator(const CharTraitsA::Char *path);

		/** Dtor. */
		virtual ~DirectoryIterator(void);

		/** Behaves like Iterator<T>::HasNext */
		virtual bool HasNext(void) const;

		/** 
		 * Behaves like Iterator<T>::Next 
		 *
		 * @throws NoSuchElementException if there is no next element
		 */
		virtual Item& Next();


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
		inline DirectoryIterator& operator =(const DirectoryIterator& rhs) {
		    if (this != &rhs) {
				throw IllegalParamException("rhs", __FILE__, __LINE__);
			}
		}

		/** pointer to the next element store */
		Item nextItem;
		
		/** pointer to the current element */
		Item currentItem;

		/** handle for iterating the directory contents */
#ifdef _WIN32
		HANDLE findHandle;
#else /* _WIN32 */
		DIR *dirStream;
#endif /* _WIN32 */

		/** 
		 * since (on linux) we need to explicitly find out if an entry is a directory,
		 * we also need to remember where we are iterating.
		 */
#ifdef _WIN32
#else /* _WIN32 */
		StringA basePath;
#endif /* _WIN32 */

	}; /* end class DirectoryIterator<CharTraitsA> */


    /**
     * Instances of this class let the user enumerate all files/subdirectories in
	 * a certain path. In contrast to the OS-dependent implementations, "." and ".."
	 * are omitted.
     */
	template<> class DirectoryIterator<CharTraitsW> 
            : public vislib::Iterator<DirectoryEntry<CharTraitsW> > {
        typedef CharTraitsW::Char Char;
		typedef DirectoryEntry<CharTraitsW> Item;
	public:

		/** Ctor.
		 *
		 * @param path the path which this Iterator will enumerate.
		 *
		 * @throws SystemException if iterating the directory fails
		 */
		DirectoryIterator(const CharTraitsW::Char *path);

		/** Dtor. */
		virtual ~DirectoryIterator(void);

		/** Behaves like Iterator<T>::HasNext */
		virtual bool HasNext(void) const;

		/** 
		 * Behaves like Iterator<T>::Next 
		 *
		 * @throws NoSuchElementException if there is no next element
		 */
		virtual Item& Next();


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
		inline DirectoryIterator& operator =(const DirectoryIterator& rhs) {
		    if (this != &rhs) {
				throw IllegalParamException("rhs", __FILE__, __LINE__);
			}
		}

		/** pointer to the next element store */
		Item nextItem;
		
		/** pointer to the current element */
		Item currentItem;

		/** handle for iterating the directory contents */
#ifdef _WIN32
		HANDLE findHandle;
#else /* _WIN32 */
		DIR *dirStream;
#endif /* _WIN32 */

#ifdef _WIN32
#else /* _WIN32 */
		/** 
		 * since (on linux) we need to explicitly find out if an entry is a directory,
		 * we also need to remember where we are iterating.
		 */
		StringA basePath;

		/**
		 * unicode on linux is... suboptimal.
		 */
		DirectoryIterator<CharTraitsA> *DICA;
#endif /* _WIN32 */

	}; /* end class DirectoryIterator<CharTraitsA> */
 

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
