/*
 * MemmappedFile.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "vislib/MemmappedFile.h"

#include "vislib/assert.h"
#include "vislib/error.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/IOException.h"
#include "vislib/mathfunctions.h"
#include "vislib/SystemInformation.h"
#include "vislib/memutils.h"
#include "vislib/UnsupportedOperationException.h"

#ifndef _WIN32
#include <sys/stat.h>
#include <iostream>
#endif

/*
 * vislib::sys::MemmappedFile::MemmappedFile
 */
vislib::sys::MemmappedFile::MemmappedFile(void) 
#ifdef _WIN32
		: File(), viewStart(0), viewSize(SystemInformation::AllocationGranularity()), mappedData(NULL), viewDirty(false), mapping(INVALID_HANDLE_VALUE) {
#else /* _WIN32 */
        : File(), viewStart(0), viewSize(SystemInformation::AllocationGranularity()), mappedData(NULL), viewDirty(false) {
#endif /* _WIN32 */
}

/*
 * vislib::sys::MemmappedFile::~MemmappedFile
 */
vislib::sys::MemmappedFile::~MemmappedFile(void) {
    this->Close(); // Close file here (vtable issue)
}


/*
 * vislib::sys::MemmappedFile::AdjustedViewSize
 */
inline vislib::sys::File::FileSize vislib::sys::MemmappedFile::AdjustedViewSize(File::FileSize pos) {
	File::FileSize vs = this->viewSize;
	File::FileSize start = AlignPosition(pos);
	if (this->access == READ_ONLY || this->access == READ_WRITE) {
		if (this->endPos < start + this->viewSize) {
			vs = this->endPos - start;
		}
	}
	return vs;
}


/*
 * vislib::sys::MemmappedFile::SafeUnmapView
 */
inline void vislib::sys::MemmappedFile::SafeUnmapView() {
	if (this->mappedData != NULL) {
		if (this->access != READ_ONLY) {
			this->Flush();
		}
#ifdef _WIN32
		if (!UnmapViewOfFile(this->mappedData)) {
#else /* _WIN32 */
		size_t vs = static_cast <size_t>(this->AdjustedViewSize(this->viewStart));
		if (munmap(this->mappedData, vs) == -1) {
#endif /* _WIN32 */
			throw IOException(::GetLastError(), __FILE__, __LINE__);
		}
		this->mappedData = NULL;
	}
}


/*
 * vislib::sys::MemmappedFile::SafeMapView
 */
inline char* vislib::sys::MemmappedFile::SafeMapView() {
	char *ret;

#ifdef _WIN32
	DWORD da;
	ULARGE_INTEGER fp;
	fp.QuadPart = this->AlignPosition(this->filePos);
	if (this->mapping == NULL || this->mapping == INVALID_HANDLE_VALUE) {
		throw IllegalStateException("SafeMapView while mapping invalid", __FILE__, __LINE__);
	}
	SIZE_T vs = static_cast <SIZE_T>(this->AdjustedViewSize(this->filePos));
	switch(this->access) {
		case READ_ONLY:
			da = FILE_MAP_READ;
			break;
		case READ_WRITE:
			da = FILE_MAP_WRITE;
			break;
		case WRITE_ONLY:
			da = FILE_MAP_WRITE;
			break;
		default:
			throw IllegalStateException("SafeMapView", __FILE__, __LINE__);
			break;
	}
	ret = static_cast <char*> (MapViewOfFile(this->mapping, da, fp.HighPart,
		fp.LowPart, vs));
	if (ret == NULL) {
		throw IOException(::GetLastError(), __FILE__, __LINE__);
	}
	this->viewStart = fp.QuadPart;
#else /* _WIN32 */
	off_t fp = this->AlignPosition(this->filePos);
	size_t vs = static_cast <size_t>(this->AdjustedViewSize(this->filePos));

	if (this->endPos < fp + vs) {
		/* go to the location corresponding to the last byte */
		if (lseek (this->handle, fp + vs - 1, SEEK_SET) == -1) {
			throw IOException(GetLastError(), __FILE__ , __LINE__);
		}
		/* write a dummy byte at the last location */
		if (write (this->handle, "", 1) != 1) {
			throw IOException(GetLastError(), __FILE__ , __LINE__);
		}
	}

	ret = static_cast <char*> (mmap(0, vs, this->protect, MAP_SHARED, this->handle, fp));
	if (ret == MAP_FAILED) {
		throw IOException(::GetLastError(), __FILE__, __LINE__);
	}
	this->viewStart = fp;
#endif /* _WIN32 */
	return ret;
}


#ifdef _WIN32
/*
 * vislib::sys::MemmappedFile::SafeCreateMapping
 */
inline void vislib::sys::MemmappedFile::SafeCreateMapping(File::FileSize mappingsize) {
	ULARGE_INTEGER oldSize;
	oldSize.QuadPart = mappingsize;

	this->mapping = CreateFileMapping(this->handle, NULL, this->protect, oldSize.HighPart, oldSize.LowPart, NULL);
	if (GetLastError() == ERROR_ALREADY_EXISTS) {
		LARGE_INTEGER s;
		GetFileSizeEx(this->handle, &s);
		this->referenceSize = s.QuadPart;
	}
	if (this->mapping == NULL || this->mapping == INVALID_HANDLE_VALUE) {
		throw IOException(::GetLastError(), __FILE__, __LINE__);
	}
}
#endif /* _WIN32 */


#ifdef _WIN32
/*
 * vislib::sys::MemmappedFile::SafeCloseMapping
 */
inline void vislib::sys::MemmappedFile::SafeCloseMapping() {
	if (this->mapping != INVALID_HANDLE_VALUE && this->mapping != NULL) {
		if (CloseHandle(this->mapping) == 0) {
			throw IOException(::GetLastError(), __FILE__, __LINE__);
		}
		this->mapping = INVALID_HANDLE_VALUE;
	}
}
#endif /* _WIN32 */


/*
 * vislib::sys::MemmappedFile::AlignPosition
 */
inline vislib::sys::File::FileSize vislib::sys::MemmappedFile::AlignPosition(vislib::sys::File::FileSize position) {
	static DWORD granularity = SystemInformation::AllocationGranularity();
	if (position % granularity != 0) {
		position -= position % granularity;
	}
	return position;
}

/*
 * vislib::sys::MemmappedFile::Close
 */
void vislib::sys::MemmappedFile::Close(void) {
    this->Flush();
	this->SafeUnmapView();
#ifdef _WIN32
	this->SafeCloseMapping();
#endif /* _WIN32 */

	if (this->access == READ_WRITE || this->access == WRITE_ONLY) {
#ifdef _WIN32
		LARGE_INTEGER dist;
		dist.QuadPart = this->endPos;
		SetFilePointerEx(this->handle, dist, NULL, FILE_BEGIN);
		SetEndOfFile(this->handle);
#else /* _WIN32 */
		ftruncate(this->handle, this->endPos);
#endif /* _WIN32 */
	}

#ifdef _WIN32
	this->mapping = INVALID_HANDLE_VALUE;
#endif /* _WIN32 */
	
	File::Close();
    this->viewStart = 0;
	this->viewDirty = false;
}


/*
 * vislib::sys::MemmappedFile::Flush
 */
void vislib::sys::MemmappedFile::Flush(void) {
	if (this->mappedData != NULL && this->viewDirty) {
#ifdef _WIN32
		if (FlushViewOfFile(this->mappedData, 0) == 0) {
			throw IOException(ERROR_WRITE_FAULT, __FILE__, __LINE__);
		}
#else /* _WIN32 */
		size_t vs = static_cast<size_t>(this->AdjustedViewSize(this->viewStart));
		if (msync(this->mappedData, vs, MS_SYNC) == -1) {
			throw IOException(GetLastError(), __FILE__, __LINE__);
		}
#endif /* _WIN32 */
		this->viewDirty = false;
		// BUG: (?) actually, can't we keep the mapped data? it's just not dirty anymore...
		//this->mappedData = NULL;
	}
}


/*
 * vislib::sys::MemmappedFile::GetSize
 */
vislib::sys::File::FileSize vislib::sys::MemmappedFile::GetSize(void) const {
#ifdef _WIN32
	if (this->handle != INVALID_HANDLE_VALUE) {
#else /* _WIN32 */
	if (this->handle != -1) {
#endif /* _WIN32 */
		return this->endPos;
	} else {
		throw IllegalStateException("GetSize while file not open", __FILE__, __LINE__);
	}
}


/*
 * vislib::sys::MemmappedFile::SetViewSize
 */
vislib::sys::File::FileSize vislib::sys::MemmappedFile::SetViewSize(File::FileSize newSize) {
	File::FileSize ns = this->AlignPosition(newSize);
	if (ns == 0) {
		ns = SystemInformation::AllocationGranularity();
	}
	if (ns != this->viewSize) {
		this->SafeUnmapView();
	}
	this->viewSize = ns;
	return this->viewSize;
}


/*
 * vislib::sys::MemmappedFile::Open
 */
bool vislib::sys::MemmappedFile::Open(const char *filename, const vislib::sys::File::AccessMode accessMode, 
        const vislib::sys::File::ShareMode shareMode, const vislib::sys::File::CreationMode creationMode) {
	// file already open?
	if (this->mappedData != NULL) {
		this->Close();
	}
	vislib::sys::File::AccessMode am = accessMode;
	if (am == WRITE_ONLY) {
		am = READ_WRITE;
	}
	if (File::Open(filename, am, shareMode, creationMode)) {
		this->access = accessMode;
		return this->CommonOpen(am);
    } else {
        return false;
    }
}


/*
 * vislib::sys::MemmappedFile::Open
 */
bool vislib::sys::MemmappedFile::Open(const wchar_t *filename, const vislib::sys::File::AccessMode accessMode, 
        const vislib::sys::File::ShareMode shareMode, const vislib::sys::File::CreationMode creationMode) {
	// file already open?
	if (this->mappedData != NULL) {
		this->Close();
	}
	vislib::sys::File::AccessMode am = accessMode;
	// write-only does not exist
	if (am == WRITE_ONLY) {
		am = READ_WRITE;
	}
    if (File::Open(filename, am, shareMode, creationMode)) {
		this->access = accessMode;
		return this->CommonOpen(am);
    } else {
        return false;
    }
}

/*
 * vislib::sys::MemmappedFile::commonOpen
 */
bool vislib::sys::MemmappedFile::CommonOpen(const File::AccessMode accessMode) {
#ifdef _WIN32
	this->viewStart = 0;
	this->viewDirty = false;
	LARGE_INTEGER oldSize;
	switch(accessMode) {
		case READ_ONLY:
			this->protect = PAGE_READONLY;
			if (!GetFileSizeEx(this->handle, &oldSize)) {
				throw IOException(::GetLastError(), __FILE__, __LINE__);
			} else {
				this->endPos = oldSize.QuadPart;
				this->referenceSize = oldSize.QuadPart;
				oldSize.QuadPart = 0;
			}
			break;
		case READ_WRITE:
		case WRITE_ONLY:
			this->protect = PAGE_READWRITE;
			if (!GetFileSizeEx(this->handle, &oldSize)) {
				this->endPos = 0;
				this->referenceSize = 0;
			} else {
				this->endPos = oldSize.QuadPart;
			}
			if (oldSize.QuadPart == 0) {
				oldSize.QuadPart = SystemInformation::AllocationGranularity();
				this->referenceSize = oldSize.QuadPart;
			}
			break;
		default:
			throw IllegalParamException("accessMode", __FILE__, __LINE__);
			break;
	}
	this->SafeCreateMapping(oldSize.QuadPart);
#else /* _WIN32 */
	this->viewStart = 0;
	this->viewDirty = false;
    struct stat64 stat;

	this->protect = 0;
	switch(accessMode) {
		case READ_ONLY:
			this->protect = PROT_READ;
			if (::fstat64(this->handle, &stat) == 0) {
				this->endPos = stat.st_size;
				this->referenceSize = stat.st_size;
			} else {
				throw IOException(::GetLastError(), __FILE__, __LINE__);
			}
			break;
		case READ_WRITE:
			this->protect |= PROT_READ;
			// falls through!
		case WRITE_ONLY:
			this->protect |= PROT_WRITE;
			if (::fstat64(this->handle, &stat) == 0) {
				this->endPos = stat.st_size;
			} else {
				this->endPos = 0;
				this->referenceSize = 0;			
			}
			if (stat.st_size == 0) {
				this->referenceSize = SystemInformation::AllocationGranularity();
			}
			break;
		default:
			throw IllegalParamException("accessMode", __FILE__, __LINE__);
			break;
	}
#endif /* _WIN32 */
	this->filePos = 0;
    return true;
}


/*
 * vislib::sys::MemmappedFile::Read
 */
vislib::sys::File::FileSize vislib::sys::MemmappedFile::Read(void *outBuf, 
        const vislib::sys::File::FileSize bufSize) {
	if (bufSize < 0) {
		// hate!
		throw IllegalParamException("bufSize", __FILE__, __LINE__);
	}
	if (bufSize == 0) {
		// null op
		return 0;
	}
	File::FileSize bufS = bufSize;
	if (this->filePos + bufSize > this->endPos) {
		bufS = this->endPos - this->filePos;
	}

	if (this->access == WRITE_ONLY) {
#ifdef _WIN32
		throw IOException(ERROR_ACCESS_DENIED, __FILE__, __LINE__);
#else /* _WIN32 */
		throw IOException(EACCES, __FILE__, __LINE__);
#endif /* _WIN32 */
	}

#ifdef _WIN32
	if (this->mapping == INVALID_HANDLE_VALUE || this->mapping == NULL) {
		// something went wrong when opening the file. I do not think retrying will make it better, will it?
		throw IllegalStateException("Read", __FILE__, __LINE__);
	}
#endif /* _WIN32 */

	// can we just slurp everything in?
	if ((this->mappedData != NULL) && (filePos >= viewStart) && (filePos + bufSize <= viewStart + viewSize)) {
		// BUG: is there something like memcpy64?
		memcpy(outBuf, static_cast <void*>(mappedData + filePos - viewStart), static_cast <size_t>(bufS));
		this->filePos += bufS;
	} else {
		File::FileSize dataLeft, readSize;
		char *bufferPos;

		// no view defined or outside view
		if (this->mappedData == NULL || this->filePos < this->viewStart || this->filePos >= this->viewStart + this->viewSize) {
			this->SafeUnmapView();
			this->mappedData = this->SafeMapView();
		}
		dataLeft = bufS;
		bufferPos = (char*)outBuf;
		while (dataLeft > 0) {
			// BUG: what happens if reading more data than available and the view
			// has been resized to smaller than viewSize because the file is shorter?
			// should be fixed at begin of this method tho.
			readSize = vislib::math::Min(dataLeft, this->viewStart + this->viewSize - filePos);
			memcpy(bufferPos, static_cast<void*>(mappedData + filePos - viewStart), static_cast<size_t>(readSize));
			bufferPos += readSize;
			this->filePos += readSize;
			dataLeft -= readSize;
			if (dataLeft > 0) {
				this->SafeUnmapView();
				this->mappedData = this->SafeMapView();
			}
		}
	}
	return bufS;
}


/*
 * vislib::sys::MemmappedFile::Seek
 */
vislib::sys::File::FileSize vislib::sys::MemmappedFile::Seek(const vislib::sys::File::FileOffset offset, 
        const vislib::sys::File::SeekStartPoint from) {

    vislib::sys::File::FileSize destination;

	switch (from) {
		case BEGIN:
			destination = offset;
			break;
		case CURRENT:
			destination = this->filePos + offset;
			break;
		case END:
			destination = this->endPos + offset;
			break;
		default:
			// bug: now what?
			destination = 0;
			throw IllegalParamException("from", __FILE__, __LINE__);
			break;
	}
	// I cannot be arsed...
	destination = vislib::math::Max(destination, static_cast <File::FileSize>(0));
	destination = vislib::math::Min(destination, this->endPos);

	// does NOT flush, since unmapview does that if necessary
	this->filePos = destination;

    return destination;
}


/*
 * vislib::sys::MemmappedFile::Tell
 */
vislib::sys::File::FileSize vislib::sys::MemmappedFile::Tell(void) const {
#ifdef _WIN32
	if (this->handle != INVALID_HANDLE_VALUE) {
#else /* _WIN32 */
	if (this->handle != -1) {
#endif /* _WIN32 */
		return (this->filePos);
	} else {
		throw IllegalStateException("Tell while file not open", __FILE__, __LINE__);
	}    
}


/*
 * vislib::sys::MemmappedFile::Write
 */
vislib::sys::File::FileSize vislib::sys::MemmappedFile::Write(const void *buf, const File::FileSize bufSize) {
	if (bufSize < 0) {
		// hate!
		throw IllegalParamException("bufSize", __FILE__, __LINE__);
	}

	if (this->access == READ_ONLY) {
#ifdef _WIN32
		throw IOException(ERROR_ACCESS_DENIED, __FILE__, __LINE__);
#else /* _WIN32 */
		throw IOException(EACCES, __FILE__, __LINE__);
#endif /* _WIN32 */
	}

	if (bufSize == 0) {
		// NOP
		return 0;
	}

	const char *bufferPos = static_cast <const char*>(buf);
	File::FileSize dataLeft = bufSize, writeSize;
#ifdef _WIN32
	if (this->mapping == INVALID_HANDLE_VALUE || this->mapping == NULL) {
		// something went wrong when opening the file. I do not think retrying will make it better, will it?
		throw IllegalStateException("Write", __FILE__, __LINE__);
	}
#endif /* _WIN32 */
	if (this->filePos > this->endPos) {
		// what the fuck?
#ifdef _WIN32
		throw IOException(ERROR_WRITE_FAULT, __FILE__, __LINE__);
#else /* _WIN32 */
		throw IOException(EFBIG, __FILE__, __LINE__);
#endif /* _WIN32 */
	}
	while (dataLeft > 0) {
		if (this->filePos == this->referenceSize) {
			// I dub thee 'append'
			this->SafeUnmapView();
#ifdef _WIN32
			this->SafeCloseMapping();
			//this->SafeCreateMapping(this->referenceSize + dataLeft);
			this->SafeCreateMapping(this->referenceSize + this->viewSize);
#endif /* _WIN32 */
			//this->referenceSize = this->referenceSize + dataLeft;
			this->referenceSize = this->referenceSize + this->viewSize;
			this->mappedData = this->SafeMapView();
		} else {
			if (this->filePos < this->viewStart || this->filePos >= this->viewStart + this->viewSize
					|| this->mappedData == NULL) {
				this->SafeUnmapView();
				this->mappedData = this->SafeMapView();
			}
		}
		writeSize = vislib::math::Min(dataLeft, this->viewStart + this->viewSize - filePos);
		memcpy(static_cast <void*> (this->mappedData + this->filePos - this->viewStart), bufferPos, static_cast <size_t>(writeSize));
		bufferPos += writeSize;
		this->filePos += writeSize;
		dataLeft -= writeSize;
		this->viewDirty = true;
	}
	this->endPos = vislib::math::Max(this->filePos, this->endPos);
	return bufSize - dataLeft;
}

/*
 * vislib::sys::MemmappedFile::MemmappedFile copy ctor
 */
vislib::sys::MemmappedFile::MemmappedFile(const vislib::sys::MemmappedFile& rhs) {
    throw UnsupportedOperationException("vislib::sys::MemmappedFile::MemmappedFile", __FILE__, __LINE__);
}


/*
 * vislib::sys::MemmappedFile::operator =
 */
vislib::sys::MemmappedFile& vislib::sys::MemmappedFile::operator =(const vislib::sys::MemmappedFile& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }
    return *this;
}
