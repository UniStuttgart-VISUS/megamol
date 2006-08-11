/*
 * File.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifdef _WIN32
#include <Shlwapi.h>
#else /* _WIN32 */
/* tell linux runtime to do 64bit seek/tell */
#define _FILE_OFFSET_BITS 64
#define _LARGEFILE64_SOURCE  

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>
#include <stdio.h> 
#endif /* _WIN32 */

#include "vislib/File.h"
#include "vislib/error.h"
#include "vislib/assert.h"
#include "vislib/IOException.h"
#include "vislib/Trace.h"
#include "vislib/UnsupportedOperationException.h"
#include "vislib/IllegalParamException.h"


/*
 * vislib::sys::File::Delete
 */
bool vislib::sys::File::Delete(const TCHAR *filename) {
#ifdef _WIN32
	return (::DeleteFile(filename) == TRUE); 

#else /* _WIN32 */
#if defined(UNICODE) || defined(_UNICODE)
#error "Unicode implementation for File::Delete missing"
#else /* defined(UNICODE) || defined(_UNICODE) */
	return (::remove(filename) == 0);

#endif /* defined(UNICODE) || defined(_UNICODE) */

#endif /* _WIN32 */
}


/*
 * vislib::sys::File::Exists
 */
bool vislib::sys::File::Exists(const TCHAR *filename) {
#ifdef _WIN32
	return (PathFileExists(filename) == 1); // GetLastError() holds more information in case of problem. who cares

#else /* _WIN32 */
#if defined(UNICODE) || defined(_UNICODE)
#error "Unicode implementation for File::Exists missing"
#else /* defined(UNICODE) || defined(_UNICODE) */
	struct stat buf;
	int i = stat(filename, &buf); // errno holds additional information (ENOENT and EBADF etc.). who cares
	return (i == 0);

#endif /* defined(UNICODE) || defined(_UNICODE) */

#endif /* _WIN32 */
}


/*
 * vislib::sys::File::Rename
 */
bool vislib::sys::File::Rename(const TCHAR *oldName, const TCHAR *newName) {
#ifdef _WIN32
    return (::MoveFile(oldName, newName) == TRUE);
#else /* _WIN32 */
#if defined(UNICODE) || defined(_UNICODE)
#error "Unicode implementation for File::Rename missing"
#endif 
    return ::rename(oldName, newName);
#endif /* _WIN32 */
}


/*
 * vislib::sys::File::File
 */
vislib::sys::File::File(void) {
#ifdef _WIN32
    this->handle = INVALID_HANDLE_VALUE;

#else /* _WIN32 */
	this->handle = -1;

#endif /* _WIN32 */
}


/*
 * vislib::sys::File::~File
 */
vislib::sys::File::~File(void) {
	this->Close();
}


/*
 * vislib::sys::File::Close
 */
void vislib::sys::File::Close(void) {
	if (this->IsOpen()) {
#ifdef _WIN32 
		::CloseHandle(this->handle);
		this->handle = INVALID_HANDLE_VALUE;

#else /* _WIN32 */
		::close(this->handle);
		this->handle = -1;

#endif /* _WIN32 */
	}
}


/*
 * vislib::sys::File::Flush
 */
void vislib::sys::File::Flush(void) {
#ifdef _WIN32
	if (!::FlushFileBuffers(this->handle)) {
#else /* _WIN32 */
    if (::fsync(this->handle) != 0) {
#endif /* _WIN32 */
		throw IOException(::GetLastError(), __FILE__, __LINE__);
	}
}


/*
 * vislib::sys::File::GetSize
 */
vislib::sys::File::FileSize vislib::sys::File::GetSize(void) {
#ifdef _WIN32
	LARGE_INTEGER size;

	if (::GetFileSizeEx(this->handle, &size)) {
		return size.QuadPart;

#else /* _WIN32 */
	struct stat64 stat;

	if (::fstat64(this->handle, &stat) == 0) {
		return stat.st_size;

#endif /* _WIN32 */
	} else {
		throw IOException(::GetLastError(), __FILE__, __LINE__);
	}
}


/*
 * vislib::sys::File::IsEOF
 */
bool vislib::sys::File::IsEOF(void) {
    FileSize here = this->Seek(0, CURRENT);
    FileSize end = this->Seek(0, END);

    if (here == end) {
        return true;
    } else {
        this->Seek(here, BEGIN);
        return false;
    }
}


/*
 * vislib::sys::File::IsOpen
 */
bool vislib::sys::File::IsOpen(void) const {
#ifdef _WIN32
	return (this->handle != INVALID_HANDLE_VALUE);
#else /* _WIN32 */
	return (this->handle != -1);
#endif /* _WIN32 */
}


/*
 * vislib::sys::File::Open
 */
bool vislib::sys::File::Open(const TCHAR *filename, const AccessMode accessMode, 
		const ShareMode shareMode, const CreationMode creationMode) {
	this->Close();
#ifdef _WIN32
	DWORD access;
	DWORD share;
	DWORD create;

	switch (accessMode) {
		case READ_WRITE: access = GENERIC_READ | GENERIC_WRITE; break;
		case READ_ONLY: access = GENERIC_READ; break;
		case WRITE_ONLY: access = GENERIC_WRITE; break;
		default: throw IllegalParamException(_T("accessMode"), __FILE__, __LINE__);
	}

	switch (shareMode) {
		case SHARE_READ: share = FILE_SHARE_READ; break;
		case SHARE_WRITE: share = FILE_SHARE_WRITE; break;
		case SHARE_READWRITE: share = FILE_SHARE_READ | FILE_SHARE_WRITE; break;
		default: throw IllegalParamException(_T("shareMode"), __FILE__, __LINE__);
	}

	switch (creationMode) {
		case CREATE_ONLY: create = CREATE_NEW; break;
		case CREATE_OVERWRITE: create = CREATE_ALWAYS; break;
		case OPEN_ONLY: create = OPEN_EXISTING; break;
		case OPEN_CREATE: create = OPEN_ALWAYS; break;
		default: throw IllegalParamException(_T("creationMode"), __FILE__, __LINE__);
	}

	this->handle = ::CreateFile(filename, access, share, NULL, create, FILE_ATTRIBUTE_NORMAL, NULL);
	return (this->handle != INVALID_HANDLE_VALUE);
#else /* _WIN32 */
	int oflag = O_LARGEFILE | O_SYNC;
	bool fileExists = vislib::sys::File::Exists(filename);
	// ToDo: consider O_DIRECT for disabling OS-Caching

	switch (accessMode) {
		case READ_WRITE: oflag |= O_RDWR; break;
		case READ_ONLY: oflag |= O_RDONLY; break;
		case WRITE_ONLY: oflag |= O_WRONLY; break;
		default: throw IllegalParamException(_T("accessMode"), __FILE__, __LINE__);
	}

	switch (creationMode) {
		case CREATE_ONLY: 
			if (fileExists) return false;
			oflag |= O_CREAT;
			break;
		case CREATE_OVERWRITE: 
			oflag |= (fileExists) ? O_TRUNC : O_CREAT;
			break;
		case OPEN_ONLY: 
			if (!fileExists) return false;
			break;
		case OPEN_CREATE: 
			if (!fileExists) oflag |= O_CREAT;
			break;
		default: throw IllegalParamException(_T("creationMode"), __FILE__, __LINE__);
	}

#if defined(UNICODE) || defined(_UNICODE)
	assert(false); // TODO: Working with unicode under linux
#else /* defined(UNICODE) || defined(_UNICODE) */
	this->handle = ::open(filename, oflag);
#endif /* defined(UNICODE) || defined(_UNICODE) */

	return (this->handle != -1);
#endif /* _WIN32 */
}


/*
 * vislib::sys::File::Read
 */
vislib::sys::File::FileSize vislib::sys::File::Read(void *outBuf, 
                                                    const FileSize bufSize) {
#ifdef _WIN32
	DWORD readBytes;
	if (::ReadFile(this->handle, outBuf, static_cast<DWORD>(bufSize), 
			&readBytes, NULL)) {
		return readBytes;

#else /* _WIN32 */
	ASSERT(bufSize < INT_MAX);

	int readBytes = ::read(this->handle, outBuf, bufSize);
	if (readBytes != -1) {
		return readBytes;

#endif /* _WIN32 */
	} else {
		throw IOException(::GetLastError(), __FILE__, __LINE__);
	}
}


/*
 * vislib::sys::File::Seek
 */
vislib::sys::File::FileSize vislib::sys::File::Seek(const FileOffset offset, 
													const SeekStartPoint from) {
#ifdef _WIN32
	LARGE_INTEGER o;
	LARGE_INTEGER n;
	o.QuadPart = offset; 

	if (::SetFilePointerEx(this->handle, o, &n, static_cast<DWORD>(from))) {
		return n.QuadPart;		

#else /* _WIN32 */
	off64_t n = ::lseek64(this->handle, offset, from);

	if (n != static_cast<off64_t>(-1)) {
		return n;

#endif /* _WIN32 */
	} else {
		throw IOException(::GetLastError(), __FILE__, __LINE__);
	}
}


/*
 * vislib::sys::File::Write
 */
vislib::sys::File::FileSize vislib::sys::File::Write(const void *buf, 
													 const FileSize bufSize) {
#ifdef _WIN32
	DWORD writtenBytes;
	if (::WriteFile(this->handle, buf, static_cast<DWORD>(bufSize), 
			&writtenBytes, NULL)) {
		return writtenBytes;

#else /* _WIN32 */
	ASSERT(bufSize < INT_MAX);

	int writtenBytes = ::write(this->handle, buf, bufSize);
	if (writtenBytes != -1) {
		return writtenBytes;

#endif /* _WIN32 */
	} else {
		throw IOException(::GetLastError(), __FILE__, __LINE__);
	}
}


/*
 * vislib::sys::File::File
 */
vislib::sys::File::File(const File& rhs) {
    throw UnsupportedOperationException(_T("vislib::sys::File::File"), 
		__FILE__, __LINE__);
}


/*
 * vislib::sys::File::operator =
 */
vislib::sys::File& vislib::sys::File::operator =(const File& rhs) {
    if (this != &rhs) {
        throw IllegalParamException(_T("rhs"), __FILE__, __LINE__);
    }

    return *this;
}
