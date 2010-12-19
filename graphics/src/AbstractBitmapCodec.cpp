/*
 * AbstractBitmapCodec.cpp
 *
 * Copyright (C) 2009 - 2010 by Sebastian Grottel.
 * (Copyright (C) 2009 - 2010 by VISUS (Universität Stuttgart))
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractBitmapCodec.h"
#include "vislib/ArrayAllocator.h"
#include "vislib/IllegalStateException.h"
#include "vislib/MemmappedFile.h"
#include "vislib/MemoryFile.h"
#include "vislib/Path.h"
#include "vislib/SmartPtr.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::graphics::AbstractBitmapCodec::AbstractBitmapCodec
 */
vislib::graphics::AbstractBitmapCodec::AbstractBitmapCodec(void) : img(NULL) {
    // Intentionally empty
}


/*
 * vislib::graphics::AbstractBitmapCodec::~AbstractBitmapCodec
 */
vislib::graphics::AbstractBitmapCodec::~AbstractBitmapCodec(void) {
    this->img = NULL; // DO NOT DELETE!
}


/*
 * vislib::graphics::AbstractBitmapCodec::AutoDetect
 */
int vislib::graphics::AbstractBitmapCodec::AutoDetect(const void *mem,
        SIZE_T size) const {
    // does never detect compatibility
    return 0;
}


/*
 * vislib::graphics::AbstractBitmapCodec::CanAutoDetect
 */
bool vislib::graphics::AbstractBitmapCodec::CanAutoDetect(void) const {
    return false; // cannot autodetect
}


/*
 * vislib::graphics::AbstractBitmapCodec::FileNameExtsA
 */
const char* vislib::graphics::AbstractBitmapCodec::FileNameExtsA(void) const {
    return NULL; // no extensions
}


/*
 * vislib::graphics::AbstractBitmapCodec::FileNameExtsW
 */
const wchar_t*
vislib::graphics::AbstractBitmapCodec::FileNameExtsW(void) const {
    return NULL; // no extensions
}


/*
 * vislib::graphics::AbstractBitmapCodec::Load
 */
bool vislib::graphics::AbstractBitmapCodec::Load(const char* filename) {
    this->image();

    if (this->loadFromFileAImplemented()) {
        return this->loadFromFileA(filename);

    } else if (this->loadFromFileWImplemented()) {
        return this->loadFromFileW(vislib::StringW(filename).PeekBuffer());

    } else if (this->loadFromStreamImplemented()) {
        vislib::sys::MemmappedFile file;
        if (file.Open(filename, vislib::sys::File::READ_ONLY,
                vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
            return this->loadFromStream(file);
        }
        return false;

    } else if (this->loadFromMemoryImplemented()) {
        vislib::sys::MemmappedFile file;
        if (file.Open(filename, vislib::sys::File::READ_ONLY,
                vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
            vislib::sys::File::FileSize size = file.GetSize();
            vislib::SmartPtr<char, vislib::ArrayAllocator<char> > mem
                = new char[static_cast<SIZE_T>(size)];
            size = file.Read(mem.operator->(), size);
            return this->loadFromMemory(mem.operator->(),
                static_cast<SIZE_T>(size));
        }
        return false;

    }

    throw vislib::UnsupportedOperationException("Load", __FILE__, __LINE__);
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::Load
 */
bool vislib::graphics::AbstractBitmapCodec::Load(const wchar_t* filename) {
    this->image();

    if (this->loadFromFileWImplemented()) {
        return this->loadFromFileW(filename);

    } else if (this->loadFromFileAImplemented()) {
        return this->loadFromFileA(vislib::StringA(filename).PeekBuffer());

    } else if (this->loadFromStreamImplemented()) {
        vislib::sys::MemmappedFile file;
        if (file.Open(filename, vislib::sys::File::READ_ONLY,
                vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
            return this->loadFromStream(file);
        }
        return false;

    } else if (this->loadFromMemoryImplemented()) {
        vislib::sys::MemmappedFile file;
        if (file.Open(filename, vislib::sys::File::READ_ONLY,
                vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
            vislib::sys::File::FileSize size = file.GetSize();
            vislib::SmartPtr<char, vislib::ArrayAllocator<char> > mem
                = new char[static_cast<SIZE_T>(size)];
            size = file.Read(mem.operator->(), size);
            return this->loadFromMemory(mem.operator->(),
                static_cast<SIZE_T>(size));
        }
        return false;

    }

    throw vislib::UnsupportedOperationException("Load", __FILE__, __LINE__);
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::Load
 */
bool vislib::graphics::AbstractBitmapCodec::Load(vislib::sys::File& file) {
    this->image();

    if (this->loadFromStreamImplemented()) {
        return this->loadFromStream(file);

    } else if (this->loadFromMemoryImplemented()) {
        file.SeekToBegin();
        vislib::sys::File::FileSize size = file.GetSize() - file.Tell();
        vislib::SmartPtr<char, vislib::ArrayAllocator<char> > mem = new char[
            static_cast<SIZE_T>(size)];
        size = file.Read(mem.operator->(), size);
        return this->loadFromMemory(mem.operator->(),
            static_cast<SIZE_T>(size));

    } else if (this->loadFromFileAImplemented()) {
        vislib::StringA filename;
        vislib::sys::File::CreateTempFileName(filename);
        vislib::sys::MemmappedFile tmpfile;
        if (tmpfile.Open(filename, vislib::sys::File::READ_WRITE,
                vislib::sys::File::SHARE_READ,
                vislib::sys::File::CREATE_ONLY)) {

            try {
                SIZE_T size = static_cast<SIZE_T>(
                    file.GetSize() - file.Tell());
                char *buf = new char[size];
                size = static_cast<SIZE_T>(file.Read(buf, size));
                tmpfile.Write(buf, size);
                delete[] buf;
                tmpfile.Flush();
                tmpfile.SeekToBegin();

                this->loadFromFileA(filename);

            } catch(...) {
                tmpfile.Close();
                vislib::sys::File::Delete(filename);
                throw;
            }
            tmpfile.Close();
            vislib::sys::File::Delete(filename);

        }

    } else if (this->loadFromFileWImplemented()) {
        vislib::StringW filename;
        vislib::sys::File::CreateTempFileName(filename);
        vislib::sys::MemmappedFile tmpfile;
        if (tmpfile.Open(filename, vislib::sys::File::READ_WRITE,
                vislib::sys::File::SHARE_READ,
                vislib::sys::File::CREATE_ONLY)) {

            try {
                SIZE_T size = static_cast<SIZE_T>(
                    file.GetSize() - file.Tell());
                char *buf = new char[size];
                size = static_cast<SIZE_T>(file.Read(buf, size));
                tmpfile.Write(buf, size);
                delete[] buf;
                tmpfile.Flush();
                tmpfile.SeekToBegin();

                this->loadFromFileW(filename);

            } catch(...) {
                tmpfile.Close();
                vislib::sys::File::Delete(filename);
                throw;
            }
            tmpfile.Close();
            vislib::sys::File::Delete(filename);

        }

    }

    throw vislib::UnsupportedOperationException("Load", __FILE__, __LINE__);
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::Load
 */
bool vislib::graphics::AbstractBitmapCodec::Load(const void *mem, SIZE_T size) {
    this->image();

    if (this->loadFromMemoryImplemented()) {
        return this->loadFromMemory(mem, size);

    } else if (this->loadFromStreamImplemented()) {
        vislib::sys::MemoryFile stream;
        if (!stream.Open(const_cast<void*>(mem), size,
                vislib::sys::File::READ_ONLY)) {
            return false;
        }
        return this->loadFromStream(stream);

    } else if (this->loadFromFileAImplemented()) {
        vislib::StringA filename;
        vislib::sys::File::CreateTempFileName(filename);
        vislib::sys::MemmappedFile tmpfile;
        if (tmpfile.Open(filename, vislib::sys::File::READ_WRITE,
                vislib::sys::File::SHARE_READ,
                vislib::sys::File::CREATE_ONLY)) {

            try {
                tmpfile.Write(mem, size);
                tmpfile.Flush();
                tmpfile.SeekToBegin();

                this->loadFromFileA(filename);

            } catch(...) {
                tmpfile.Close();
                vislib::sys::File::Delete(filename);
                throw;
            }
            tmpfile.Close();
            vislib::sys::File::Delete(filename);

        }

    } else if (this->loadFromFileWImplemented()) {
        vislib::StringW filename;
        vislib::sys::File::CreateTempFileName(filename);
        vislib::sys::MemmappedFile tmpfile;
        if (tmpfile.Open(filename, vislib::sys::File::READ_WRITE,
                vislib::sys::File::SHARE_READ,
                vislib::sys::File::CREATE_ONLY)) {

            try {
                tmpfile.Write(mem, size);
                tmpfile.Flush();
                tmpfile.SeekToBegin();

                this->loadFromFileW(filename);

            } catch(...) {
                tmpfile.Close();
                vislib::sys::File::Delete(filename);
                throw;
            }
            tmpfile.Close();
            vislib::sys::File::Delete(filename);

        }

    }

    throw vislib::UnsupportedOperationException("Load", __FILE__, __LINE__);
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::Save
 */
bool vislib::graphics::AbstractBitmapCodec::Save(
        const char* filename, bool overwrite) const {
    this->image();

    if (vislib::sys::File::Exists(filename) && !overwrite) {
        return false; // must not overwrite existing file
    }

    if (this->saveToFileAImplemented()) {
        return this->saveToFileA(filename);

    } else if (this->saveToFileWImplemented()) {
        return this->saveToFileW(vislib::StringW(filename));

    } else if (this->saveToStreamImplemented()) {
        vislib::sys::MemmappedFile file;
        if (file.Open(filename, vislib::sys::File::WRITE_ONLY,
                vislib::sys::File::SHARE_READ,
                vislib::sys::File::CREATE_OVERWRITE)) {
            return this->saveToStream(file);
        }
        return false;

    } else if (this->saveToMemoryImplemented()) {
        vislib::RawStorage mem;
        if (this->saveToMemory(mem)) {
        vislib::sys::MemmappedFile file;
            if (file.Open(filename, vislib::sys::File::WRITE_ONLY,
                    vislib::sys::File::SHARE_READ,
                    vislib::sys::File::CREATE_OVERWRITE)) {
                file.Write(mem, mem.GetSize());
                file.Flush();
                file.Close();
                return true;
            }
        }
        return false;

    }

    throw vislib::UnsupportedOperationException("Save", __FILE__, __LINE__);
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::Save
 */
bool vislib::graphics::AbstractBitmapCodec::Save(
        const wchar_t* filename, bool overwrite) const {
    this->image();

    if (vislib::sys::File::Exists(filename) && !overwrite) {
        return false; // must not overwrite existing file
    }

    if (this->saveToFileWImplemented()) {
        return this->saveToFileW(filename);

    } else if (this->saveToFileAImplemented()) {
        return this->saveToFileA(vislib::StringA(filename));

    } else if (this->saveToStreamImplemented()) {
        vislib::sys::MemmappedFile file;
        if (file.Open(filename, vislib::sys::File::WRITE_ONLY,
                vislib::sys::File::SHARE_READ,
                vislib::sys::File::CREATE_OVERWRITE)) {
            return this->saveToStream(file);
        }
        return false;

    } else if (this->saveToMemoryImplemented()) {
        vislib::RawStorage mem;
        if (this->saveToMemory(mem)) {
        vislib::sys::MemmappedFile file;
            if (file.Open(filename, vislib::sys::File::WRITE_ONLY,
                    vislib::sys::File::SHARE_READ,
                    vislib::sys::File::CREATE_OVERWRITE)) {
                file.Write(mem, mem.GetSize());
                file.Flush();
                file.Close();
                return true;
            }
        }
        return false;

    }

    throw vislib::UnsupportedOperationException("Save", __FILE__, __LINE__);
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::Save
 */
bool vislib::graphics::AbstractBitmapCodec::Save(
        vislib::sys::File& file) const {
    this->image();

    if (this->saveToStreamImplemented()) {
        return this->saveToStream(file);

    } else if (this->saveToMemoryImplemented()) {
        vislib::RawStorage mem;
        if (this->saveToMemory(mem)) {
            file.Write(mem, mem.GetSize());
        }
        return false;

    } else if (this->saveToFileAImplemented()) {
        vislib::StringA filename;
        vislib::sys::File::CreateTempFileName(filename);
        vislib::sys::MemmappedFile tmpfile;
        if (tmpfile.Open(filename, vislib::sys::File::READ_WRITE,
                vislib::sys::File::SHARE_READ,
                vislib::sys::File::CREATE_ONLY)) {

            try {
                if (this->saveToFileA(filename)) {
                    tmpfile.Flush();
                    tmpfile.SeekToBegin();
                    SIZE_T size = static_cast<SIZE_T>(tmpfile.GetSize());
                    char *buf = new char[size];
                    size = static_cast<SIZE_T>(tmpfile.Read(buf,
                        static_cast<vislib::sys::File::FileSize>(size)));
                    file.Write(buf,
                        static_cast<vislib::sys::File::FileSize>(size));
                    delete[] buf;
                    return true;

                }

            } catch(...) {
                tmpfile.Close();
                vislib::sys::File::Delete(filename);
                throw;
            }
            tmpfile.Close();
            vislib::sys::File::Delete(filename);

        }
        return false;

    } else if (this->saveToFileWImplemented()) {
        vislib::StringW filename;
        vislib::sys::File::CreateTempFileName(filename);
        vislib::sys::MemmappedFile tmpfile;
        if (tmpfile.Open(filename, vislib::sys::File::READ_WRITE,
                vislib::sys::File::SHARE_READ,
                vislib::sys::File::CREATE_ONLY)) {

            try {
                if (this->saveToFileW(filename)) {
                    tmpfile.Flush();
                    tmpfile.SeekToBegin();
                    SIZE_T size = static_cast<SIZE_T>(tmpfile.GetSize());
                    char *buf = new char[size];
                    size = static_cast<SIZE_T>(tmpfile.Read(buf,
                        static_cast<vislib::sys::File::FileSize>(size)));
                    file.Write(buf,
                        static_cast<vislib::sys::File::FileSize>(size));
                    delete[] buf;
                    return true;

                }

            } catch(...) {
                tmpfile.Close();
                vislib::sys::File::Delete(filename);
                throw;
            }
            tmpfile.Close();
            vislib::sys::File::Delete(filename);

        }
        return false;

    }

    throw vislib::UnsupportedOperationException("Save", __FILE__, __LINE__);
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::Save
 */
bool vislib::graphics::AbstractBitmapCodec::Save(
        vislib::RawStorage& outmem) const {
    this->image();

    if (this->saveToMemoryImplemented()) {
        return this->saveToMemory(outmem);

    } else if (this->saveToStreamImplemented()) {
        vislib::sys::MemoryFile file;
        if (!file.Open(outmem, vislib::sys::File::READ_WRITE)) {
            return false;
        }

        return this->saveToStream(file);

    } else if (this->saveToFileAImplemented()) {
        vislib::StringA filename;
        vislib::sys::File::CreateTempFileName(filename);
        vislib::sys::MemmappedFile tmpfile;
        if (tmpfile.Open(filename, vislib::sys::File::READ_WRITE,
                vislib::sys::File::SHARE_READ,
                vislib::sys::File::CREATE_ONLY)) {

            try {
                if (this->saveToFileA(filename)) {
                    tmpfile.Flush();
                    tmpfile.SeekToBegin();
                    SIZE_T size = static_cast<SIZE_T>(tmpfile.GetSize());
                    outmem.AssertSize(size);
                    size = static_cast<SIZE_T>(tmpfile.Read(outmem,
                        static_cast<vislib::sys::File::FileSize>(size)));
                    outmem.EnforceSize(size, true);
                    return true;

                }

            } catch(...) {
                tmpfile.Close();
                vislib::sys::File::Delete(filename);
                throw;
            }
            tmpfile.Close();
            vislib::sys::File::Delete(filename);

        }
        return false;

    } else if (this->saveToFileWImplemented()) {
        vislib::StringW filename;
        vislib::sys::File::CreateTempFileName(filename);
        vislib::sys::MemmappedFile tmpfile;
        if (tmpfile.Open(filename, vislib::sys::File::READ_WRITE,
                vislib::sys::File::SHARE_READ,
                vislib::sys::File::CREATE_ONLY)) {

            try {
                if (this->saveToFileW(filename)) {
                    tmpfile.Flush();
                    tmpfile.SeekToBegin();
                    SIZE_T size = static_cast<SIZE_T>(tmpfile.GetSize());
                    outmem.AssertSize(size);
                    size = static_cast<SIZE_T>(tmpfile.Read(outmem,
                        static_cast<vislib::sys::File::FileSize>(size)));
                    outmem.EnforceSize(size, true);
                    return true;

                }

            } catch(...) {
                tmpfile.Close();
                vislib::sys::File::Delete(filename);
                throw;
            }
            tmpfile.Close();
            vislib::sys::File::Delete(filename);

        }
        return false;

    }

    throw vislib::UnsupportedOperationException("Save", __FILE__, __LINE__);
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::image
 */
vislib::graphics::BitmapImage&
vislib::graphics::AbstractBitmapCodec::image(void) {
    if (this->img == NULL) {
        throw vislib::IllegalStateException(
            "Must set 'Image' member before calling", __FILE__, __LINE__);
    }
    return *this->img;
}


/*
 * vislib::graphics::AbstractBitmapCodec::image
 */
const vislib::graphics::BitmapImage&
vislib::graphics::AbstractBitmapCodec::image(void) const {
    if (this->img == NULL) {
        throw vislib::IllegalStateException(
            "Must set 'Image' member before calling", __FILE__, __LINE__);
    }
    return *this->img;
}


/*
 * vislib::graphics::AbstractBitmapCodec::loadFromFileA
 */
bool vislib::graphics::AbstractBitmapCodec::loadFromFileA(
        const char *filename) {
    throw vislib::UnsupportedOperationException("loadFromFileA",
        __FILE__, __LINE__);
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::loadFromFileAImplemented
 */
bool
vislib::graphics::AbstractBitmapCodec::loadFromFileAImplemented(void) const {
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::loadFromFileW
 */
bool vislib::graphics::AbstractBitmapCodec::loadFromFileW(
        const wchar_t *filename) {
    throw vislib::UnsupportedOperationException("loadFromFileW",
        __FILE__, __LINE__);
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::loadFromFileWImplemented
 */
bool
vislib::graphics::AbstractBitmapCodec::loadFromFileWImplemented(void) const {
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::loadFromMemory
 */
bool vislib::graphics::AbstractBitmapCodec::loadFromMemory(const void *mem,
        SIZE_T size) {
    throw vislib::UnsupportedOperationException("loadFromMemory",
        __FILE__, __LINE__);
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::loadFromMemoryImplemented
 */
bool
vislib::graphics::AbstractBitmapCodec::loadFromMemoryImplemented(void) const {
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::loadFromStream
 */
bool vislib::graphics::AbstractBitmapCodec::loadFromStream(
        vislib::sys::File& stream) {
    throw vislib::UnsupportedOperationException("loadFromStream",
        __FILE__, __LINE__);
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::loadFromStreamImplemented
 */
bool
vislib::graphics::AbstractBitmapCodec::loadFromStreamImplemented(void) const {
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::saveToFileA
 */
bool vislib::graphics::AbstractBitmapCodec::saveToFileA(
        const char *filename) const {
    throw vislib::UnsupportedOperationException("saveToFileA",
        __FILE__, __LINE__);
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::saveToFileAImplemented
 */
bool
vislib::graphics::AbstractBitmapCodec::saveToFileAImplemented(void) const {
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::saveToFileW
 */
bool vislib::graphics::AbstractBitmapCodec::saveToFileW(
        const wchar_t *filename) const {
    throw vislib::UnsupportedOperationException("saveToFileW",
        __FILE__, __LINE__);
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::saveToFileWImplemented
 */
bool
vislib::graphics::AbstractBitmapCodec::saveToFileWImplemented(void) const {
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::saveToMemory
 */
bool vislib::graphics::AbstractBitmapCodec::saveToMemory(
        vislib::RawStorage &mem) const {
    throw vislib::UnsupportedOperationException("saveToMemory",
        __FILE__, __LINE__);
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::saveToMemoryImplemented
 */
bool
vislib::graphics::AbstractBitmapCodec::saveToMemoryImplemented(void) const {
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::saveToStream
 */
bool vislib::graphics::AbstractBitmapCodec::saveToStream(
        vislib::sys::File& stream) const {
    throw vislib::UnsupportedOperationException("saveToStream",
        __FILE__, __LINE__);
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::saveToStreamImplemented
 */
bool
vislib::graphics::AbstractBitmapCodec::saveToStreamImplemented(void) const {
    return false;
}
