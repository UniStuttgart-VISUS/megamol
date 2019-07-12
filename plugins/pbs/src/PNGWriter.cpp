#include "PNGWriter.h"
#include "vislib/sys/Path.h"


namespace megamol {
namespace pbs {

    /*
 * CinematicView::render2file_setup
 */
bool PNGWriter::setup(std::string _fullpath) {

    // init png data struct
    this->pngdata.buffer = nullptr;
    this->pngdata.ptr = nullptr;
    this->pngdata.infoptr = nullptr;

    this->pngdata.path = vislib::sys::Path::GetDirectoryName(_fullpath.c_str());
    this->pngdata.filename = _fullpath;

    vislib::sys::Path::MakeDirectory(this->pngdata.path.c_str());

    return true;
}




/*
 * PNGWriter::render2file_write_png
 */
bool PNGWriter::render2file() {

    // open final image file
    if (!this->pngdata.file.Open(this->pngdata.filename.c_str(), vislib::sys::File::WRITE_ONLY,
            vislib::sys::File::SHARE_EXCLUSIVE, vislib::sys::File::CREATE_OVERWRITE)) {
        throw vislib::Exception("[PNGWriter] [startAnimRendering] Cannot open output file", __FILE__, __LINE__);
    }

    // init png lib
    this->pngdata.ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, &this->pngError, &this->pngWarn);
    if (this->pngdata.ptr == nullptr) {
        throw vislib::Exception("[PNGWriter] [startAnimRendering] Cannot create png structure", __FILE__, __LINE__);
    }
    this->pngdata.infoptr = png_create_info_struct(this->pngdata.ptr);
    if (this->pngdata.infoptr == nullptr) {
        throw vislib::Exception("[PNGWriter] [startAnimRendering] Cannot create png info", __FILE__, __LINE__);
    }
    png_set_write_fn(this->pngdata.ptr, static_cast<void*>(&this->pngdata.file), &this->pngWrite, &this->pngFlush);
    png_set_IHDR(this->pngdata.ptr, this->pngdata.infoptr, this->pngdata.width, this->pngdata.height, 8,
        PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    if (this->pngdata.buffer == nullptr) {
        throw vislib::Exception(
            "[PNGWriter] [writeTextureToPng] Failed to create Screenshot: Cannot read image data", __FILE__, __LINE__);
    }

    BYTE** rows = nullptr;
    try {
        rows = new BYTE*[this->pngdata.height];
        for (UINT i = 0; i < this->pngdata.height; i++) {
            rows[this->pngdata.height - (1 + i)] = this->pngdata.buffer + this->pngdata.bpp * i * this->pngdata.width;
        }
        png_set_rows(this->pngdata.ptr, this->pngdata.infoptr, rows);

        png_write_png(this->pngdata.ptr, this->pngdata.infoptr, PNG_TRANSFORM_IDENTITY, nullptr);

        ARY_SAFE_DELETE(rows);
    } catch (...) {
        if (rows != nullptr) {
            ARY_SAFE_DELETE(rows);
        }
        throw;
    }

    if (this->pngdata.ptr != nullptr) {
        if (this->pngdata.infoptr != nullptr) {
            png_destroy_write_struct(&this->pngdata.ptr, &this->pngdata.infoptr);
        } else {
            png_destroy_write_struct(&this->pngdata.ptr, (png_infopp) nullptr);
        }
    }

    try {
        this->pngdata.file.Flush();
    } catch (...) {
    }
    try {
        this->pngdata.file.Close();
    } catch (...) {
    }

    return true;
}


/*
 * CinematicView::render2file_finish
 */
bool PNGWriter::finish() {

    if (this->pngdata.ptr != nullptr) {
        if (this->pngdata.infoptr != nullptr) {
            png_destroy_write_struct(&this->pngdata.ptr, &this->pngdata.infoptr);
        } else {
            png_destroy_write_struct(&this->pngdata.ptr, (png_infopp) nullptr);
        }
    }

    try {
        this->pngdata.file.Flush();
    } catch (...) {
    }
    try {
        this->pngdata.file.Close();
    } catch (...) {
    }

    //ARY_SAFE_DELETE(this->pngdata.buffer);

    vislib::sys::Log::DefaultLog.WriteInfo("[PNGWriter] STOPPED rendering.");
    return true;
}

void PNGWriter::set_buffer(BYTE* _buffer, unsigned _width, unsigned _height, unsigned int _bytesPerPixel = 4) {
    if (_buffer != nullptr) {
        // Create new byte buffer
        this->pngdata.bpp = _bytesPerPixel;
        this->pngdata.width = _width;
        this->pngdata.height = _height;

        //this->pngdata.buffer = new BYTE[this->pngdata.width * this->pngdata.height * this->pngdata.bpp];
        //if (this->pngdata.buffer == nullptr) {
        //    throw vislib::Exception(
        //        "[PNGWriter] [startAnimRendering] Cannot allocate image buffer.", __FILE__, __LINE__);
        //}
        this->pngdata.buffer = _buffer;

    } else {
        this->pngdata.bpp = 0;
        this->pngdata.width = 0;
        this->pngdata.height = 0;
        vislib::sys::Log::DefaultLog.WriteError("[PNGWriter] Input buffer is empty.");
    }
}

} // namespace pbs
} // namespace megamol