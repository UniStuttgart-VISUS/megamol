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

#include "vislib/CharTraits.h"
#include "vislib/String.h"
#include "vislib/Iterator.h"
#include "vislib/IOException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/SystemException.h"
#include "vislib/Path.h"

#ifndef _WIN32
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <iostream>

#include "vislib/File.h"
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

    /**
     * Instances of this class let the user enumerate all files/subdirectories in
	 * a certain path. In contrast to the OS-dependent implementations, "." and ".."
	 * are omitted.
     */
	template<class T> class DirectoryIterator: public vislib::Iterator<DirectoryEntry<T> > {
	};/* end class Iterator */


    /**
     * Instances of this class let the user enumerate all files/subdirectories in
	 * a certain path. In contrast to the OS-dependent implementations, "." and ".."
	 * are omitted. For Unicode support, use DirectoryIterator<CharTraitsW>.
     */
	template<> class DirectoryIterator<CharTraitsA>: public vislib::Iterator<DirectoryEntry<CharTraitsA> > {
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
		 * @throws IllegalStateException if there is no next element
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


	/*
	 * DirectoryIterator<CharTraitsA>::DirectoryIterator
	 */
	DirectoryIterator<CharTraitsA>::DirectoryIterator(const CharTraitsA::Char *path) {
#ifdef _WIN32
		WIN32_FIND_DATAA fd;

		StringA p = path;
		if (!p.EndsWith(vislib::sys::Path::SEPARATOR_A)) {
			if (!p.EndsWith(":")) {
				p += vislib::sys::Path::SEPARATOR_A;
			}
		}

		this->findHandle = FindFirstFileA(p + "*.*", &fd);
		if (this->findHandle == INVALID_HANDLE_VALUE) {
			throw SystemException(__FILE__, __LINE__);
		}

		while ((strcmp(fd.cFileName, ".") == 0) || (strcmp(fd.cFileName, "..") == 0)) {
			if (FindNextFileA(this->findHandle, &fd) == 0) {
				DWORD le;
				if ((le = GetLastError()) == ERROR_NO_MORE_FILES) {
					this->nextItem.Path.Clear();
				} else {
					throw SystemException(le, __FILE__, __LINE__);
				}
			}
		}

		if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
			this->nextItem.Type = Item::DIRECTORY;
		} else {
			this->nextItem.Type = Item::FILE;
		}
		this->nextItem.Path = fd.cFileName;
#else /* _WIN32 */
		struct dirent *de;

		if ((this->dirStream = opendir(path)) == NULL) {
			throw SystemException(__FILE__, __LINE__);
		}
		this->basePath = path;
		// BUG: Linux documentation is all lies. the errno stunt does not work at all.
		//errno = 0;
		do {
			if ((de = readdir(this->dirStream)) == NULL) {
				//if (errno == 0) {
				//	this->nextItem.Path.Clear();
				//} else {
				//	throw SystemException(errno, __FILE__, __LINE__);
				//}
			}
		} while (de != NULL && ((strcmp(de->d_name, ".") == 0) || (strcmp(de->d_name, "..") == 0)));
		if (de == NULL) {
			this->nextItem.Path.Clear();
		}	else {
			if (vislib::sys::File::IsDirectory(this->basePath + Path::SEPARATOR_A + de->d_name)) {
				this->nextItem.Type = Item::DIRECTORY;
			} else {
				this->nextItem.Type = Item::FILE;
			}
			this->nextItem.Path = de->d_name;
		}
#endif /* _WIN32 */
	}


	/*
	 * DirectoryIterator<CharTraitsA>::~DirectoryIterator
	 */
	DirectoryIterator<CharTraitsA>::~DirectoryIterator() {
#ifdef _WIN32
		if (this->findHandle != INVALID_HANDLE_VALUE) {
			FindClose(this->findHandle);
		}
#else /* _WIN32 */
		if (this->dirStream != NULL) {
			closedir(dirStream);
		}
#endif /* WIN32 */
	}


	/*
	 * DirectoryIterator<CharTraitsA>::HasNext
	 */
	bool DirectoryIterator<CharTraitsA>::HasNext() const {
		return !this->nextItem.Path.IsEmpty();
	}


	/*
	 * DirectoryIterator<CharTraitsA>::Next
	 */
	DirectoryEntry<CharTraitsA>& DirectoryIterator<CharTraitsA>::Next() {
		this->currentItem = this->nextItem;
#ifdef _WIN32
		if (this->currentItem.Path.IsEmpty()) {
			throw IllegalStateException("No next element.", __FILE__, __LINE__);
		} else {
			WIN32_FIND_DATAA fd;
			DWORD le;

			do {
				if (FindNextFileA(this->findHandle, &fd) == 0) {
					if ((le = GetLastError()) == ERROR_NO_MORE_FILES) {
						this->nextItem.Path.Clear();
					} else {
						throw IOException(le, __FILE__, __LINE__);
					}
				} else {
					if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
						this->nextItem.Type = Item::DIRECTORY;
					} else {
						this->nextItem.Type = Item::FILE;
					}
					this->nextItem.Path = fd.cFileName;
				}
			} while ((strcmp(fd.cFileName, ".") == 0) || (strcmp(fd.cFileName, "..") == 0));
			return this->currentItem;
		}
#else /* _WIN32 */
		if (this->currentItem.Path.IsEmpty()) {
			throw IllegalStateException("No next element.", __FILE__, __LINE__);
		} else {
			struct dirent *de;
			do {
				// BUG: Linux documentation is all lies. the errno stunt does not work at all.
				//errno = 0;
				if ((de = readdir(this->dirStream)) == NULL) {
					//if (errno == 0) {
					//	this->nextItem.Path.Clear();
					//} else {
					//	throw SystemException(errno, __FILE__, __LINE__);
					//}
				} else {
					if (vislib::sys::File::IsDirectory(this->basePath + Path::SEPARATOR_A + de->d_name)) {
						this->nextItem.Type = Item::DIRECTORY;
					} else {
						this->nextItem.Type = Item::FILE;
					}
				}
			} while (de != NULL && ((strcmp(de->d_name, ".") == 0) || (strcmp(de->d_name, "..") == 0)));
			if (de == NULL) {
				this->nextItem.Path.Clear();
			}
			else {
				this->nextItem.Path = de->d_name;
			}
			return this->currentItem;
		}
#endif /* _WIN32 */
	}


    /**
     * Instances of this class let the user enumerate all files/subdirectories in
	 * a certain path. In contrast to the OS-dependent implementations, "." and ".."
	 * are omitted.
     */
	template<> class DirectoryIterator<CharTraitsW>: public vislib::Iterator<DirectoryEntry<CharTraitsW> > {
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
		 * @throws IllegalStateException if there is no next element
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


	/*
	 * DirectoryIterator<CharTraitsW>::DirectoryIterator
	 */
	DirectoryIterator<CharTraitsW>::DirectoryIterator(const CharTraitsW::Char *path) {
#ifdef _WIN32
		WIN32_FIND_DATAW fd;

		StringW p = path;
		if (!p.EndsWith(vislib::sys::Path::SEPARATOR_W)) {
			if (!p.EndsWith(L":")) {
				p += vislib::sys::Path::SEPARATOR_W;
			}
		}

		this->findHandle = FindFirstFileW(p + L"*.*", &fd);
		if (this->findHandle == INVALID_HANDLE_VALUE) {
			throw SystemException(__FILE__, __LINE__);
		}

		while ((wcscmp(fd.cFileName, L".") == 0) || (wcscmp(fd.cFileName, L"..") == 0)) {
			if (FindNextFileW(this->findHandle, &fd) == 0) {
				DWORD le;
				if ((le = GetLastError()) == ERROR_NO_MORE_FILES) {
					this->nextItem.Path.Clear();
				} else {
					throw SystemException(le, __FILE__, __LINE__);
				}
			}
		}

		if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
			this->nextItem.Type = Item::DIRECTORY;
		} else {
			this->nextItem.Type = Item::FILE;
		}
		this->nextItem.Path = fd.cFileName;
#else /* _WIN32 */
#if 0
		struct dirent *de;

		if ((this->dirStream = opendir(path)) == NULL) {
			throw SystemException(__FILE__, __LINE__);
		}
		this->basePath = path;
		// BUG: Linux documentation is all lies. the errno stunt does not work at all.
		//errno = 0;
		do {
			if ((de = readdir(this->dirStream)) == NULL) {
				//if (errno == 0) {
				//	this->nextItem.Path.Clear();
				//} else {
				//	throw SystemException(errno, __FILE__, __LINE__);
				//}
			}
		} while (de != NULL && ((strcmp(de->d_name, ".") == 0) || (strcmp(de->d_name, "..") == 0)));
		if (de == NULL) {
			this->nextItem.Path.Clear();
		}	else {
			if (vislib::sys::File::IsDirectory(this->basePath + Path::SEPARATORSTR_A + de->d_name)) {
				this->nextItem.Type = Item::DIRECTORY;
			} else {
				this->nextItem.Type = Item::FILE;
			}
			this->nextItem.Path = de->d_name;
		}
#endif
#endif /* _WIN32 */
	}


	/*
	 * DirectoryIterator<CharTraitsW>::~DirectoryIterator
	 */
	DirectoryIterator<CharTraitsW>::~DirectoryIterator() {
#ifdef _WIN32
		if (this->findHandle != INVALID_HANDLE_VALUE) {
			FindClose(this->findHandle);
		}
#else /* _WIN32 */
		if (this->dirStream != NULL) {
			closedir(dirStream);
		}
#endif /* WIN32 */
	}


	/*
	 * DirectoryIterator<CharTraitsW>::HasNext
	 */
	bool DirectoryIterator<CharTraitsW>::HasNext() const {
		return !this->nextItem.Path.IsEmpty();
	}


	/*
	 * DirectoryIterator<CharTraitsW>::Next
	 */
	DirectoryEntry<CharTraitsW>& DirectoryIterator<CharTraitsW>::Next() {
		this->currentItem = this->nextItem;
#ifdef _WIN32
		if (this->currentItem.Path.IsEmpty()) {
			throw IllegalStateException("No next element.", __FILE__, __LINE__);
		} else {
			WIN32_FIND_DATAW fd;
			DWORD le;

			do {
				if (FindNextFileW(this->findHandle, &fd) == 0) {
					if ((le = GetLastError()) == ERROR_NO_MORE_FILES) {
						this->nextItem.Path.Clear();
					} else {
						throw SystemException(le, __FILE__, __LINE__);
					}
				} else {
					if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
						this->nextItem.Type = Item::DIRECTORY;
					} else {
						this->nextItem.Type = Item::FILE;
					}
					this->nextItem.Path = fd.cFileName;
				}
			} while ((wcscmp(fd.cFileName, L".") == 0) || (wcscmp(fd.cFileName, L"..") == 0));
			return this->currentItem;
		}
#else /* _WIN32 */
#if 0
		if (this->currentItem.Path.IsEmpty()) {
			throw IllegalStateException("No next element.", __FILE__, __LINE__);
		} else {
			struct dirent *de;
			do {
				// BUG: Linux documentation is all lies. the errno stunt does not work at all.
				//errno = 0;
				if ((de = readdir(this->dirStream)) == NULL) {
					//if (errno == 0) {
					//	this->nextItem.Path.Clear();
					//} else {
					//	throw SystemException(errno, __FILE__, __LINE__);
					//}
				} else {
					if (vislib::sys::File::IsDirectory(this->basePath + Path::SEPARATORSTR_A + de->d_name)) {
						this->nextItem.Type = Item::DIRECTORY;
					} else {
						this->nextItem.Type = Item::FILE;
					}
				}
			} while (de != NULL && ((strcmp(de->d_name, ".") == 0) || (strcmp(de->d_name, "..") == 0)));
			if (de == NULL) {
				this->nextItem.Path.Clear();
			}
			else {
				this->nextItem.Path = de->d_name;
			}
			return this->currentItem;
		}
#endif
		return this->currentItem;
#endif /* _WIN32 */
	} /* end class DirectoryIterator<CharTraitsW> */

} /* end namespace sys */
} /* end namespace vislib */

#endif /* VISLIB_DIRECTORYITERATOR_H_INCLUDED */
