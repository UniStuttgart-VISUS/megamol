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

/*
 * vislib::sys::MemmappedFile::MemmappedFile
 */
vislib::sys::MemmappedFile::MemmappedFile(void) 
        : File(), mappedData(NULL), viewStart(0), viewSize(0), viewDirty(false), mapping(INVALID_HANDLE_VALUE) {
#ifdef _WIN32
#else /* _WIN32 */
	assert(false);
#endif /* _WIN32 */
}

/*
 * vislib::sys::MemmappedFile::~MemmappedFile
 */
vislib::sys::MemmappedFile::~MemmappedFile(void) {
	// BUG: delete stuff
	// relies on File::~File closing the file
}


/*
 * vislib::sys::MemmappedFile::SafeUnmapView
 */
inline void vislib::sys::MemmappedFile::SafeUnmapView() {
#ifdef _WIN32
	if (this->mappedData != NULL) {
		if (this->access != READ_ONLY) {
			this->Flush();
		}
		if (!UnmapViewOfFile(this->mappedData)) {
			throw IOException(::GetLastError(), __FILE__, __LINE__);
		}
		this->mappedData = NULL;
	}
#else /* _WIN32 */
#endif /* _WIN32 */
}


/*
 * vislib::sys::MemmappedFile::SafeMapView
 */
inline char* vislib::sys::MemmappedFile::SafeMapView(DWORD desiredAccess, ULARGE_INTEGER filePos) {
	char *ret;
#ifdef _WIN32
	if (this->mapping == NULL || this->mapping == INVALID_HANDLE_VALUE) {
		throw IllegalStateException("SafeMapView while mapping invalid", __FILE__, __LINE__);
	}
	SIZE_T vs = static_cast <SIZE_T>(this->viewSize);
	if (this->access == READ_ONLY || this->access == READ_WRITE) {
		if (this->endPos < this->viewSize) {
			vs = static_cast <SIZE_T>(this->endPos);
		}
	}
	ret = static_cast <char*> (MapViewOfFile(this->mapping, desiredAccess, filePos.HighPart,
		filePos.LowPart, vs));
	if (ret == NULL) {
		throw IOException(::GetLastError(), __FILE__, __LINE__);
	}
	return ret;
#else /* _WIN32 */
#endif /* _WIN32 */
}


/*
 * vislib::sys::MemmappedFile::SafeCloseMapping
 */
inline void vislib::sys::MemmappedFile::SafeCloseMapping() {
#ifdef _WIN32
	if (this->mapping != INVALID_HANDLE_VALUE && this->mapping != NULL) {
		if (CloseHandle(this->mapping) == 0) {
			throw IOException(::GetLastError(), __FILE__, __LINE__);
		}
		this->mapping = INVALID_HANDLE_VALUE;
	}
#else /* _WIN32 */
#endif /* _WIN32 */
}


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
	this->SafeCloseMapping();

#ifdef _WIN32
	if (this->access == READ_WRITE || this->access == WRITE_ONLY) {
		//Seek(0, END);
		LARGE_INTEGER dist;
		dist.QuadPart = this->endPos;
		SetFilePointerEx(this->handle, dist, NULL, FILE_BEGIN);
		SetEndOfFile(this->handle);
	}
#else /* _WIN32 */
#endif /* _WIN32 */
	
	File::Close();
    this->viewStart = 0;
	this->viewSize = 0;
	this->viewDirty = false;
	this->mapping = INVALID_HANDLE_VALUE;
}


/*
 * vislib::sys::MemmappedFile::Flush
 */
void vislib::sys::MemmappedFile::Flush(void) {
#ifdef _WIN32
	if (this->mappedData != NULL && this->viewDirty) {
		if (FlushViewOfFile(this->mappedData, 0) == 0) {
			throw IOException(ERROR_WRITE_FAULT, __FILE__, __LINE__);
		}
		this->viewDirty = false;
		// BUG: (?) actually, can't we keep the mapped data? it's just not dirty anymore...
		//this->mappedData = NULL;
	}
#else /* _WIN32 */
#endif /* _WIN32 */
}


/*
 * vislib::sys::MemmappedFile::GetSize
 */
vislib::sys::File::FileSize vislib::sys::MemmappedFile::GetSize(void) const {
	if (this->handle != INVALID_HANDLE_VALUE) {
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
	this->viewSize = SystemInformation::AllocationGranularity();
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
	this->mapping = CreateFileMapping(this->handle, NULL, this->protect, oldSize.HighPart, oldSize.LowPart, NULL);
	if (GetLastError() == ERROR_ALREADY_EXISTS) {
		LARGE_INTEGER s;
		GetFileSizeEx(this->handle, &s);
		this->referenceSize = s.QuadPart;
	}
	if (this->mapping == NULL || this->mapping == INVALID_HANDLE_VALUE) {
		throw IOException(::GetLastError(), __FILE__, __LINE__);
		File::Close();
		return false;
	}
	this->filePos = 0;
    return true;
#else /* _WIN32 */
	return false;
#endif /* _WIN32 */
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
#ifdef _WIN32
	File::FileSize bufS = bufSize;
	if (this->filePos + bufSize > this->endPos) {
		bufS = this->endPos - this->filePos;
	}
	if (this->access == WRITE_ONLY) {
		throw IOException(ERROR_ACCESS_DENIED, __FILE__, __LINE__);
	}
	if (this->mapping == INVALID_HANDLE_VALUE || this->mapping == NULL) {
		// something went wrong when opening the file. I do not think retrying will make it better, will it?
		throw IllegalStateException("Read", __FILE__, __LINE__);
	}

	// can we just slurp everything in?
	if ((this->mappedData != NULL) && (filePos >= viewStart) && (filePos + bufSize <= viewStart + viewSize)) {
		// BUG: is there something like memcpy64?
		memcpy(outBuf, static_cast <void*>(mappedData + filePos - viewStart), static_cast <size_t>(bufS));
		this->filePos += bufS;
	} else {
		DWORD da;
		File::FileSize dataLeft, readSize;
		char *bufferPos;
		ULARGE_INTEGER fp;
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
				throw IllegalStateException("Read", __FILE__, __LINE__);
				break;
		}

		// no view defined or outside view
		fp.QuadPart = this->AlignPosition(this->filePos);
		if (this->mappedData == NULL || this->filePos < this->viewStart || this->filePos > this->viewStart + this->viewSize) {
			this->SafeUnmapView();
			this->mappedData = this->SafeMapView(da, fp);
			viewStart = fp.QuadPart;
		}
		dataLeft = bufS;
		bufferPos = (char*)outBuf;
		while (dataLeft > 0) {
			// BUG: what happens if reading more data than available and the view
			// has been resized to smaller than viewSize because the file is shorter?
			// should be fixed at begin of this method tho.
			readSize = min(dataLeft, this->viewStart + this->viewSize - filePos);
			memcpy(bufferPos, static_cast<void*>(mappedData + filePos - viewStart), static_cast<size_t>(readSize));
			bufferPos += readSize;
			this->filePos += readSize;
			dataLeft -= readSize;
			if (dataLeft > 0) {
				fp.QuadPart = this->AlignPosition(this->filePos);
				this->SafeUnmapView();
				this->mappedData = this->SafeMapView(da, fp);
				viewStart = fp.QuadPart;
			}
		}
	}
	return bufS;
#else /* _WIN32 */
	// BUG: implement?
	return 0;
#endif /* _WIN32 */
}


/*
 * vislib::sys::MemmappedFile::Seek
 */
vislib::sys::File::FileSize vislib::sys::MemmappedFile::Seek(const vislib::sys::File::FileOffset offset, 
        const vislib::sys::File::SeekStartPoint from) {

    vislib::sys::File::FileSize destination;
	bool isLeavingView = false;

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
			break;
	}
	// I cannot be arsed...
	destination = vislib::math::Max(destination, static_cast <File::FileSize>(0));
	destination = vislib::math::Min(destination, this->endPos);

	// does NOT flush, since unmapview does that if necessary
	//if (destination < this->viewStart || destination > this->viewStart + this->viewSize) {
	//	isLeavingView = true;
	//}

	//if ((this->access == READ_WRITE || this->access == WRITE_ONLY) && this->viewDirty && isLeavingView) {
	//	this->Flush();
	//}
	this->filePos = destination;

    return destination;
}


/*
 * vislib::sys::MemmappedFile::Tell
 */
vislib::sys::File::FileSize vislib::sys::MemmappedFile::Tell(void) const {
	if (this->handle != INVALID_HANDLE_VALUE) {
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
	if (bufSize == 0) {
		// null op
		return 0;
	}
#ifdef _WIN32
	ULARGE_INTEGER fp;
	DWORD da;
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
			throw IllegalStateException("Read", __FILE__, __LINE__);
			break;
	}

	const char *bufferPos = static_cast <const char*>(buf);
	File::FileSize dataLeft = bufSize, writeSize;
	if (this->mapping == INVALID_HANDLE_VALUE || this->mapping == NULL) {
		// something went wrong when opening the file. I do not think retrying will make it better, will it?
		throw IllegalStateException("Write", __FILE__, __LINE__);
	}
	//if (this->filePos + bufSize > this->referenceSize) {
		if (this->filePos > this->endPos) {
			// what the fuck?
			throw IOException(ERROR_WRITE_FAULT, __FILE__, __LINE__);
		}
		if (this->filePos == this->referenceSize) {
			// I dub thee 'append'
			//if (CloseHandle(this->mapping) == 0) {
			//	throw IOException(::GetLastError(), __FILE__, __LINE__);
			//}
			this->Flush();
			LARGE_INTEGER s;
			s.QuadPart = this->referenceSize + bufSize;
			DWORD lastError = GetLastError();
			this->mapping = CreateFileMapping(this->handle, NULL, this->protect, s.HighPart, s.LowPart, NULL);
			if ((GetLastError() != ERROR_ALREADY_EXISTS && GetLastError() != lastError) || this->mapping == NULL
					|| this->mapping == INVALID_HANDLE_VALUE) {
				// something is wrong
				throw IOException(::GetLastError(), __FILE__, __LINE__);
			}
			this->SafeUnmapView();
			this->referenceSize = s.QuadPart;
			fp.QuadPart = this->AlignPosition(filePos);
			this->mappedData = this->SafeMapView(da, fp);
			this->viewStart = fp.QuadPart;
		}
		if (this->filePos < viewStart || this->filePos > viewStart + viewSize || this->mappedData == NULL) {
			fp.QuadPart = this->AlignPosition(filePos);
			this->mappedData = this->SafeMapView(da, fp);
			this->viewStart = fp.QuadPart;		
		}
		while (dataLeft > 0) {
			writeSize = min(dataLeft, this->viewStart + this->viewSize - filePos);
			memcpy(static_cast <void*> (mappedData + filePos - viewStart), bufferPos, static_cast <size_t>(writeSize));
			bufferPos += writeSize;
			this->filePos += writeSize;
			dataLeft -= writeSize;
			this->viewDirty = true;
			if (dataLeft > 0) {
				this->Flush();
				fp.QuadPart = this->AlignPosition(this->filePos);
				this->SafeUnmapView();
				this->mappedData = SafeMapView(da, fp);
				viewStart = fp.QuadPart;
			}
		}
		this->endPos = max(this->filePos, this->endPos);
		return bufSize - dataLeft;
	//}
#else /* _WIN32 */
	return 0;
#endif /* _WIN32 */
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
