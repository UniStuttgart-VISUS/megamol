/*
 * MemmappedFile.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "vislib/MemmappedFile.h"

#include "vislib/assert.h"
#include "vislib/error.h"
#include "vislib/IllegalParamException.h"
//#include "vislib/IllegalStateException.h"
#include "vislib/IOException.h"
#include "vislib/mathfunctions.h"
#include "vislib/SystemInformation.h"
#include "vislib/memutils.h"
#include "vislib/UnsupportedOperationException.h"

/*
 * vislib::sys::MemmappedFile::MemmappedFile
 */
vislib::sys::MemmappedFile::MemmappedFile(void) 
        : File() {
#ifdef _WIN32
	mappedData = NULL;
    viewStart = 0;
	//viewSize = 0;
	viewSize = sys::SystemInformation::AllocationGranularity();
	viewDirty = false;
	mapping = INVALID_HANDLE_VALUE;
#else /* _WIN32 */
	assert(false);
#endif /* _WIN32 */
}

/*
 * vislib::sys::MemmappedFile::~MemmappedFile
 */
vislib::sys::MemmappedFile::~MemmappedFile(void) {
	// BUG: delete stuff
}


/*
 * vislib::sys::MemmappedFile::safeUnmap
 */
inline void vislib::sys::MemmappedFile::safeUnmap() {
#ifdef _WIN32
	if (this->mappedData != NULL) {
		if (!UnmapViewOfFile(this->mappedData)) {
			throw IOException(::GetLastError(), __FILE__, __LINE__);
		}
		this->mappedData = NULL;
	}
#else /* _WIN32 */
#endif /* _WIN32 */
}

/*
 * vislib::sys::MemmappedFile::safeCloseMapping
 */
inline void vislib::sys::MemmappedFile::safeCloseMapping() {
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
 * vislib::sys::MemmappedFile::Close
 */
void vislib::sys::MemmappedFile::Close(void) {
    this->Flush();
	this->safeUnmap();
	this->safeCloseMapping();

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
	//this->viewSize = 0;
	this->viewDirty = false;
}

/*
 * vislib::sys::MemmappedFile::Flush
 */
void vislib::sys::MemmappedFile::Flush(void) {
#ifdef _WIN32
	if (this->mappedData != NULL && this->viewDirty) {
		if (FlushViewOfFile(this->mappedData, 0) == 0) {
			throw IOException(::GetLastError(), __FILE__, __LINE__);
		}
		this->viewDirty = false;
		// BUG: actually, can't we keep the mapped data? it's just not dirty anymore...
		//this->mappedData = NULL;
	}
#else /* _WIN32 */
#endif
}

/*
 * vislib::sys::MemmappedFile::GetSize
 */
vislib::sys::File::FileSize vislib::sys::MemmappedFile::GetSize(void) const {
	// BUG: steht schon fest. aber wie geht das mit append?
	return this->endPos;
}

/*
 * vislib::sys::MemmappedFile::SetSize
 */
vislib::sys::File::FileSize vislib::sys::MemmappedFile::SetViewSize(File::FileSize newSize) {
	File::FileSize ns = newSize;
	if (ns % SystemInformation::AllocationGranularity() != 0) {
		ns /= SystemInformation::AllocationGranularity();
		ns *= SystemInformation::AllocationGranularity();
	}
	if (ns != this->viewSize) {
		this->Flush();
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
		return commonOpen(accessMode);
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
		return commonOpen(accessMode);
    } else {
        return false;
    }
}

/*
 * vislib::sys::MemmappedFile::commonOpen
 */
bool vislib::sys::MemmappedFile::commonOpen(const File::AccessMode accessMode) {
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
			//if (!GetFileSizeEx(this->handle, &oldSize) || oldSize.QuadPart == 0) {
			if (!GetFileSizeEx(this->handle, &oldSize)) {
				//oldSize.QuadPart = SystemInformation::AllocationGranularity();
				//this->referenceSize = oldSize.QuadPart;
				//if (accessMode == READ_WRITE) {
				//	this->endPos = oldSize.QuadPart;
				//} else {
				//	this->endPos = 0;
				//}
				this->endPos = 0;
			} else {
				this->endPos = oldSize.QuadPart;
			}
			if (oldSize.QuadPart == 0) {
				oldSize.QuadPart = SystemInformation::AllocationGranularity();
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
	if (this->mapping == NULL) {
		throw IOException(::GetLastError(), __FILE__, __LINE__);
		File::Close();
		return false;
	}
	this->access = accessMode;
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
#ifdef _WIN32
	File::FileSize bufS = bufSize;
	if (this->filePos + bufSize > this->endPos) {
		bufS = this->endPos - this->filePos;
	}
	if (this->access == WRITE_ONLY) {
		throw IOException(ERROR_ACCESS_DENIED, __FILE__, __LINE__);
	}

	// can we just slurp everything in?
	if ((this->mappedData != NULL) && (filePos >= viewStart) && (filePos + bufSize <= viewStart + viewSize)) {
		// BUG: is there something like memcpy64?
		memcpy(outBuf, static_cast <void*>(mappedData + filePos - viewStart), bufS);
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
		}

		// no view defined or outside view
		fp.QuadPart = this->filePos;
		if (this->mappedData == NULL || this->filePos < this->viewStart || this->filePos > this->viewStart + this->viewSize) {
			if (this->mappedData != NULL) {
				if (!UnmapViewOfFile(this->mappedData)) {
					throw IOException(::GetLastError(), __FILE__, __LINE__);
				}
			}
			if (fp.QuadPart % SystemInformation::AllocationGranularity() != 0) {
				fp.QuadPart /= SystemInformation::AllocationGranularity();
				fp.QuadPart *= SystemInformation::AllocationGranularity();
			}
			this->mappedData = static_cast <char*> (MapViewOfFile(this->mapping, da, fp.HighPart,
				fp.LowPart, this->viewSize));
			if (this->mappedData == NULL) {
				throw IOException(::GetLastError(), __FILE__, __LINE__);
			}
			viewStart = fp.QuadPart;
		}
		dataLeft = bufS;
		bufferPos = (char*)outBuf;
		while (dataLeft > 0) {
			// what if filepos not at viewStart???
			readSize = min(dataLeft, this->viewStart + this->viewSize - filePos);
			memcpy(bufferPos, static_cast <void*>(mappedData + filePos - viewStart), readSize);
			bufferPos += readSize;
			this->filePos += readSize;
			dataLeft -= readSize;
			if (dataLeft > 0) {
				fp.QuadPart = this->filePos;
				if (!UnmapViewOfFile(this->mappedData)) {
					throw IOException(::GetLastError(), __FILE__, __LINE__);
				}
				if (fp.QuadPart % SystemInformation::AllocationGranularity() != 0) {
					fp.QuadPart /= SystemInformation::AllocationGranularity();
					fp.QuadPart *= SystemInformation::AllocationGranularity();
				}
				this->mappedData = static_cast <char*> (MapViewOfFile(this->mapping, da, fp.HighPart,
					fp.LowPart, this->viewSize));
				if (this->mappedData == NULL) {
					throw IOException(::GetLastError(), __FILE__, __LINE__);
				}
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
	}
	// I cannot be arsed...
	destination = vislib::math::Max (destination, static_cast <File::FileSize>(0));
	destination = vislib::math::Min (destination, this->endPos);
	if (destination < this->viewStart || destination > this->viewStart + this->viewSize) {
		isLeavingView = true;
	}

	if ((this->access == READ_WRITE || this->access == WRITE_ONLY) && this->viewDirty && isLeavingView) {
        this->Flush();
    }
	this->filePos = destination;

    return destination;
}

/*
 * vislib::sys::MemmappedFile::Tell
 */
vislib::sys::File::FileSize vislib::sys::MemmappedFile::Tell(void) const {
    return (this->filePos);
}

/*
 * vislib::sys::MemmappedFile::Write
 */
vislib::sys::File::FileSize vislib::sys::MemmappedFile::Write(const void *buf, const File::FileSize bufSize) {
#ifdef _WIN32
	if (bufSize <= 0) {
		// hate!
		throw IOException(ERROR_WRITE_FAULT, __FILE__, __LINE__);
	}
	LARGE_INTEGER fp;
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
	}

	const char *bufferPos = static_cast <const char*>(buf);
	File::FileSize dataLeft = bufSize, writeSize;
	if (this->mapping == INVALID_HANDLE_VALUE || this->mapping == NULL) {
		// something went wrong when opening the file. I do not think retrying will make it better, will it?
		throw IOException(ERROR_WRITE_FAULT, __FILE__, __LINE__);
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
			if ((GetLastError() != ERROR_ALREADY_EXISTS && GetLastError() != lastError) || this->mapping == NULL) {
				// something is wrong
				throw IOException(::GetLastError(), __FILE__, __LINE__);
			}
			if (this->mappedData != NULL) {
				if (!UnmapViewOfFile(this->mappedData)) {
					throw IOException(::GetLastError(), __FILE__, __LINE__);
				}				
			}
			this->referenceSize = s.QuadPart;
			fp.QuadPart = filePos;
			if (fp.QuadPart % SystemInformation::AllocationGranularity() != 0) {
				fp.QuadPart /= SystemInformation::AllocationGranularity();
				fp.QuadPart *= SystemInformation::AllocationGranularity();
			}
			this->mappedData = static_cast <char*> (MapViewOfFile(this->mapping, da, fp.HighPart,
					fp.LowPart, this->viewSize));
			if (this->mappedData == NULL) {
				throw IOException(::GetLastError(), __FILE__, __LINE__);
			}
			this->viewStart = fp.QuadPart;
		}
		if (this->filePos < viewStart || this->filePos > viewStart + viewSize || this->mappedData == NULL) {
			fp.QuadPart = filePos;
			if (fp.QuadPart % SystemInformation::AllocationGranularity() != 0) {
				fp.QuadPart /= SystemInformation::AllocationGranularity();
				fp.QuadPart *= SystemInformation::AllocationGranularity();
			}
			this->mappedData = static_cast <char*> (MapViewOfFile(this->mapping, da, fp.HighPart,
					fp.LowPart, this->viewSize));
			if (this->mappedData == NULL) {
				throw IOException(::GetLastError(), __FILE__, __LINE__);
			}
			this->viewStart = fp.QuadPart;		
		}
		while (dataLeft > 0) {
			writeSize = min(dataLeft, this->viewStart + this->viewSize - filePos);
			memcpy(static_cast <void*> (mappedData + filePos - viewStart), bufferPos, writeSize);
			bufferPos += writeSize;
			this->filePos += writeSize;
			dataLeft -= writeSize;
			this->viewDirty = true;
			if (dataLeft > 0) {
				this->Flush();
				fp.QuadPart = this->filePos;
				if (!UnmapViewOfFile(this->mappedData)) {
					throw IOException(::GetLastError(), __FILE__, __LINE__);
				}
				if (fp.QuadPart % SystemInformation::AllocationGranularity() != 0) {
					fp.QuadPart /= SystemInformation::AllocationGranularity();
					fp.QuadPart *= SystemInformation::AllocationGranularity();
				}
				this->mappedData = static_cast <char*> (MapViewOfFile(this->mapping, da, fp.HighPart,
					fp.LowPart, this->viewSize));
				if (this->mappedData == NULL) {
					throw IOException(::GetLastError(), __FILE__, __LINE__);
				}
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
