/*
 * File.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#endif /* _WIN32 */

#include "vislib/File.h"
#include "vislib/IOException.h"
#include "vislib/Trace.h"


/*
 * vislib::sys::File::Delete
 */
bool vislib::sys::File::Delete(const char *filename) {
#ifdef _WIN32
	return (::DeleteFileA(filename) == TRUE); 
#else /* _WIN32 */
	return (::remove(filename) == 0);
#endif /* _WIN32 */
}


/*
 * vislib::sys::File::Exists
 */
bool vislib::sys::File::Exists(const char *filename) {
#ifdef _WIN32
    WIN32_FIND_DATAA fd;
    HANDLE fh = INVALID_HANDLE_VALUE;

    if ((fh = ::FindFirstFileA(filename, &fd)) != INVALID_HANDLE_VALUE) {
        ::CloseHandle(fh);
        return true;

    } else {
        return false;
    }

#else /* _WIN32 */
    int fh = ::open(filename, O_RDONLY, S_IREAD);
    
    if (fh != -1) {
        ::close(fh);
        return true;

    } else {
        return false;
    }
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
 * vislib::sys::File::File
 */
vislib::sys::File::File(const File& rhs) {
	// TODO: Consider forbidden copy ctor and assignment instead.
#ifdef _WIN32
	this->handle = ::CreateFile;
	if (::DuplicateHandle(::GetCurrentProcess(), rhs.handle, 
			::GetCurrentProcess(), &this->handle, 0, TRUE, 
			DUPLICATE_SAME_ACCESS) == FALSE) {
		TRACE(_T("DuplicateHandle failed in File copy ctor.\n"));
		this->handle = NULL;
	}

#else /* _WIN32 */
	this->handle = ::dup(rhs.handle);

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
bool vislib::sys::File::Open(const char *filename, const DWORD flags) {
	TRACE(_T("TODO: Specify mode for File::Open in a system idependent manner.\n"));
#ifdef _WIN32 // TODO
	DWORD desiredAccess = GENERIC_READ | GENERIC_WRITE;
	DWORD shareMode = FILE_SHARE_READ;
	DWORD creationDisposition = OPEN_ALWAYS;
	DWORD flagsAndAttributes = FILE_ATTRIBUTE_NORMAL;

	return ((this->handle = ::CreateFileA(filename, desiredAccess, shareMode,
		NULL, creationDisposition, flagsAndAttributes, NULL)) != NULL);

#else /* _WIN32 */
	int oflag = O_RDWR;	// TODO
	int pmode = S_IREAD | S_IWRITE;

	return ((this->handle = ::open(filename, oflag, pmode)) != -1);

#endif /* _WIN32 */
}


/*
 * vislib::sys::File::Read
 */
vislib::sys::File::FileSize vislib::sys::File::Read(void *outBuf, const FileSize bufSize) {
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

	if (n != ((off64_t) - 1)) {
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
