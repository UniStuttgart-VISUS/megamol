#pragma once
#include "png.h"
#include "vislib/sys/File.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/FastFile.h"
#include <chrono>
#include <string>

namespace megamol {
namespace pbs {

class PNGWriter {
public:

    /** Render to file functions */
    bool setup(std::string _fullpath);

    /** */
    bool render2file();

    /** */
    bool finish();

    /** */
    void set_buffer(BYTE* _buffer, unsigned int _width, unsigned int _height, unsigned int _bytesPerPixel);

    /**
     * Error handling function for png export
     *
     * @param pngPtr The png structure pointer
     * @param msg The error message
     */
    static void PNGAPI pngError(png_structp pngPtr, png_const_charp msg) {
        throw vislib::Exception(msg, __FILE__, __LINE__);
    }

    /**
     * Error handling function for png export
     *
     * @param pngPtr The png structure pointer
     * @param msg The error message
     */
    static void PNGAPI pngWarn(png_structp pngPtr, png_const_charp msg) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN, "Png-Warning: %s\n", msg);
    }

    /**
     * Write function for png export
     *
     * @param pngPtr The png structure pointer
     * @param buf The pointer to the buffer to be written
     * @param size The number of bytes to be written
     */
    static void PNGAPI pngWrite(png_structp pngPtr, png_bytep buf, png_size_t size) {
        vislib::sys::File* f = static_cast<vislib::sys::File*>(png_get_io_ptr(pngPtr));
        f->Write(buf, size);
    }

    /**
     * Flush function for png export
     *
     * @param pngPtr The png structure pointer
     */
    static void PNGAPI pngFlush(png_structp pngPtr) {
        vislib::sys::File* f = static_cast<vislib::sys::File*>(png_get_io_ptr(pngPtr));
        f->Flush();
    }



    private:

    struct pngData {
        BYTE* buffer = nullptr;
        vislib::sys::FastFile file;
        unsigned int width;
        unsigned int height;
        unsigned int bpp;
        std::string path;
        std::string filename;
        png_structp ptr = nullptr;
        png_infop infoptr = nullptr;
    } pngdata;

};
} // namespace pbs
} // namespace megamol