/*
 * AbstractBitmapCodec.cpp
 *
 * Copyright (C) 2009 by Sebastian Grottel.
 * (Copyright (C) 2009 by Visualisierungsinstitut Universitaet Stuttgart.)
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractBitmapCodec.h"
#include "vislib/IllegalStateException.h"
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
 * vislib::graphics::AbstractBitmapCodec::AbstractBitmapCodec
 */
int vislib::graphics::AbstractBitmapCodec::AutoDetect(const void *mem,
        SIZE_T size) const {
    // does never detect compatibility
    return 0;
}


/*
 * vislib::graphics::AbstractBitmapCodec::AbstractBitmapCodec
 */
bool vislib::graphics::AbstractBitmapCodec::CanAutoDetect(void) const {
    return false; // cannot autodetect
}


/*
 * vislib::graphics::AbstractBitmapCodec::AbstractBitmapCodec
 */
bool vislib::graphics::AbstractBitmapCodec::CanLoadFromFile(void) const {
    return this->CanLoadFromMemory() || this->CanLoadFromStream();
}


/*
 * vislib::graphics::AbstractBitmapCodec::AbstractBitmapCodec
 */
bool vislib::graphics::AbstractBitmapCodec::CanLoadFromMemory(void) const {
    return false; // cannot load at all
}


/*
 * vislib::graphics::AbstractBitmapCodec::AbstractBitmapCodec
 */
bool vislib::graphics::AbstractBitmapCodec::CanLoadFromStream(void) const {
    return this->CanLoadFromMemory();
}


/*
 * vislib::graphics::AbstractBitmapCodec::AbstractBitmapCodec
 */
bool vislib::graphics::AbstractBitmapCodec::CanSaveToFile(void) const {
    return this->CanSaveToMemory() || this->CanSaveToStream();
}


/*
 * vislib::graphics::AbstractBitmapCodec::AbstractBitmapCodec
 */
bool vislib::graphics::AbstractBitmapCodec::CanSaveToMemory(void) const {
    return false; // cannot save at all
}


/*
 * vislib::graphics::AbstractBitmapCodec::AbstractBitmapCodec
 */
bool vislib::graphics::AbstractBitmapCodec::CanSaveToStream(void) const {
    return this->CanSaveToMemory();
}


/*
 * vislib::graphics::AbstractBitmapCodec::AbstractBitmapCodec
 */
const char* vislib::graphics::AbstractBitmapCodec::FileNameExtsA(void) const {
    return NULL; // no extensions
}


/*
 * vislib::graphics::AbstractBitmapCodec::AbstractBitmapCodec
 */
const wchar_t*
vislib::graphics::AbstractBitmapCodec::FileNameExtsW(void) const {
    return NULL; // no extensions
}


/*
 * vislib::graphics::AbstractBitmapCodec::AbstractBitmapCodec
 */
bool vislib::graphics::AbstractBitmapCodec::Load(
        const vislib::StringA& filename) {
    bool useStreamLoad;
    this->image();

    if (((useStreamLoad = this->CanLoadFromStream()) == true)
            || (this->CanLoadFromMemory())) {

        vislib::sys::File file;

        if (file.Open(filename, vislib::sys::File::READ_ONLY,
                vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
            bool retval;

            if (useStreamLoad) {
                retval = this->Load(file);

            } else {
                char *mem;
                SIZE_T size = static_cast<SIZE_T>(
                    file.GetSize() - file.Tell());
                mem = new char[size];
                size = static_cast<SIZE_T>(file.Read(mem, size));
                retval = this->Load(mem, size);
                delete[] mem;
            }

            file.Close();

            return retval;
        }

        return false;
    }
    throw vislib::UnsupportedOperationException("Load", __FILE__, __LINE__);
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::AbstractBitmapCodec
 */
bool vislib::graphics::AbstractBitmapCodec::Load(
        const vislib::StringW& filename) {
    bool useStreamLoad;
    this->image();

    if (((useStreamLoad = this->CanLoadFromStream()) == true)
            || (this->CanLoadFromMemory())) {

        vislib::sys::File file;

        if (file.Open(filename, vislib::sys::File::READ_ONLY,
                vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
            bool retval;

            if (useStreamLoad) {
                retval = this->Load(file);

            } else {
                char *mem;
                SIZE_T size = static_cast<SIZE_T>(
                    file.GetSize() - file.Tell());
                mem = new char[size];
                size = static_cast<SIZE_T>(file.Read(mem, size));
                retval = this->Load(mem, size);
                delete[] mem;
            }

            file.Close();

            return retval;
        }

        return false;
    }
    throw vislib::UnsupportedOperationException("Load", __FILE__, __LINE__);
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::AbstractBitmapCodec
 */
bool vislib::graphics::AbstractBitmapCodec::Load(vislib::sys::File& file) {
    this->image();
    if (this->CanLoadFromMemory()) {
        char *mem;
        bool retval;
        SIZE_T size = static_cast<SIZE_T>(file.GetSize() - file.Tell());
        mem = new char[size];
        size = static_cast<SIZE_T>(file.Read(mem, size));
        retval = this->Load(mem, size);
        delete[] mem;
        return retval;
    }
    throw vislib::UnsupportedOperationException("Load", __FILE__, __LINE__);
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::AbstractBitmapCodec
 */
bool vislib::graphics::AbstractBitmapCodec::Load(const void *mem,
        SIZE_T size) {
    this->image();
    throw vislib::UnsupportedOperationException("Load", __FILE__, __LINE__);
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::AbstractBitmapCodec
 */
bool vislib::graphics::AbstractBitmapCodec::Load(
        const vislib::RawStorage& mem) {
    this->image();
    return this->Load(mem, mem.GetSize());
}


/*
 * vislib::graphics::AbstractBitmapCodec::AbstractBitmapCodec
 */
bool vislib::graphics::AbstractBitmapCodec::Save(
        const vislib::StringA& filename, bool overwrite) const {
    bool useStreamSave;
    this->image();

    if (((useStreamSave = this->CanSaveToStream()) == true)
            || (this->CanSaveToMemory())) {

        vislib::sys::File file;

        if (file.Open(filename, vislib::sys::File::WRITE_ONLY,
                vislib::sys::File::SHARE_READ, overwrite
                ? vislib::sys::File::CREATE_OVERWRITE
                : vislib::sys::File::CREATE_ONLY)) {
            bool retval;

            if (useStreamSave) {
                retval = this->Save(file);

            } else {
                vislib::RawStorage mem;
                if (this->Save(mem)) {
                    retval = (file.Write(mem, mem.GetSize()) == mem.GetSize());
                } else {
                    retval = false;
                }
            }

            file.Close();

            return retval;
        }

        return false;
    }
    throw vislib::UnsupportedOperationException("Save", __FILE__, __LINE__);
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::AbstractBitmapCodec
 */
bool vislib::graphics::AbstractBitmapCodec::Save(
        const vislib::StringW& filename, bool overwrite) const {
    bool useStreamSave;
    this->image();

    if (((useStreamSave = this->CanSaveToStream()) == true)
            || (this->CanSaveToMemory())) {

        vislib::sys::File file;

        if (file.Open(filename, vislib::sys::File::WRITE_ONLY,
                vislib::sys::File::SHARE_READ, overwrite
                ? vislib::sys::File::CREATE_OVERWRITE
                : vislib::sys::File::CREATE_ONLY)) {
            bool retval;

            if (useStreamSave) {
                retval = this->Save(file);

            } else {
                vislib::RawStorage mem;
                if (this->Save(mem)) {
                    retval = (file.Write(mem, mem.GetSize()) == mem.GetSize());
                } else {
                    retval = false;
                }
            }

            file.Close();

            return retval;
        }

        return false;
    }
    throw vislib::UnsupportedOperationException("Save", __FILE__, __LINE__);
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::AbstractBitmapCodec
 */
bool vislib::graphics::AbstractBitmapCodec::Save(
        vislib::sys::File& file) const {
    this->image();
    if (this->CanSaveToMemory()) {
        vislib::RawStorage mem;
        if (this->Save(mem)) {
            return file.Write(mem, mem.GetSize()) == mem.GetSize();
        }
        return false;
    }
    throw vislib::UnsupportedOperationException("Save", __FILE__, __LINE__);
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::AbstractBitmapCodec
 */
bool vislib::graphics::AbstractBitmapCodec::Save(
        vislib::RawStorage& outmem) const {
    this->image();
    throw vislib::UnsupportedOperationException("Save", __FILE__, __LINE__);
    return false;
}


/*
 * vislib::graphics::AbstractBitmapCodec::AbstractBitmapCodec
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
 * vislib::graphics::AbstractBitmapCodec::AbstractBitmapCodec
 */
const vislib::graphics::BitmapImage&
vislib::graphics::AbstractBitmapCodec::image(void) const {
    if (this->img == NULL) {
        throw vislib::IllegalStateException(
            "Must set 'Image' member before calling", __FILE__, __LINE__);
    }
    return *this->img;
}
