/*
 * PngBitmapCodec.cpp
 *
 * Copyright (C) 2010 by Sebastian Grottel.
 * (Copyright (C) 2010 by VISUS (Universitaet Stuttgart))
 * Alle Rechte vorbehalten.
 */
#include "vislib/graphics/PngBitmapCodec.h"
#include "png.h"
#include "vislib/SmartPtr.h"
#include "zlib.h"

using namespace sg::graphics;
using namespace vislib::graphics;


namespace sg {
namespace graphics {
namespace pngutil {

/**
 * Utility buffer to read a png from memory
 */
typedef struct __png_readbuffer_t {
    const unsigned char* buffer;
    SIZE_T bufferSize;
    SIZE_T bufferPos;
} pngReadBuffer;


/**
 * Utility function to read a png from memory
 *
 * @param png_ptr The png control struct
 * @param data Pointer to the memory to receive the data
 * @param length The numbers of bytes to read
 */
void pngReadData(png_structp png_ptr, png_bytep data, png_size_t length) {
    pngReadBuffer* rd = static_cast<pngReadBuffer*>(png_get_io_ptr(png_ptr));
    SIZE_T remaining = rd->bufferSize - rd->bufferPos;
    if (length > remaining) {
        png_error(png_ptr, "unexpected end-of-data");
    }
    ::memcpy(data, rd->buffer + rd->bufferPos, length);
    rd->bufferPos += length;
}


/**
 * Utility function to ignore Png warnings
 *
 * @param png_ptr The png control struct pointer
 * @param msg The png warning message
 */
void pngWarningIgnore(png_structp png_ptr, png_const_charp msg) {
    // intentionally empty
}


/**
 * Utility function to throw vislib::Exception on png error
 *
 * @param png_ptr The png control struct pointer
 * @param msg The png error message
 */
void pngErrorThrow(png_structp png_ptr, png_const_charp msg) {
    throw vislib::Exception(msg, __FILE__, __LINE__);
}


/**
 * Writes the data to the vislib stream
 *
 * @param png_ptr The png control struct pointer
 * @param data The data to be written
 * @param length The length of the data in bytes
 */
void pngWriteData(png_structp png_ptr, png_bytep data, png_size_t length) {
    vislib::sys::File** stream = static_cast<vislib::sys::File**>(::png_get_io_ptr(png_ptr));
    if ((*stream)->Write(data, length) != length) {
        ::png_error(png_ptr, "generic write error");
    }
}


/**
 * Flushes the vislib stream
 *
 * @param png_ptr The png control struct pointer
 */
void pngFlushData(png_structp png_ptr) {
    vislib::sys::File** stream = static_cast<vislib::sys::File**>(::png_get_io_ptr(png_ptr));
    (*stream)->Flush();
}


} /* end namespace pngutil */
} /* end namespace graphics */
} /* end namespace sg */


/*
 * PngBitmapCodec::PngBitmapCodec
 */
PngBitmapCodec::PngBitmapCodec() : AbstractBitmapCodec() {
    // intentionally empty
}


/*
 * PngBitmapCodec::~PngBitmapCodec
 */
PngBitmapCodec::~PngBitmapCodec() {
    // intentionally empty
}


/*
 * PngBitmapCodec::AutoDetect
 */
int PngBitmapCodec::AutoDetect(const void* mem, SIZE_T size) const {
    const unsigned char* dat = static_cast<const unsigned char*>(mem);
    if (size < 8)
        return -1; // insufficient data

    if ((dat[0] == 137) && (dat[1] == 80) && (dat[2] == 78) && (dat[3] == 71) && (dat[4] == 13) && (dat[5] == 10) &&
        (dat[6] == 26) && (dat[7] == 10)) {
        return 1; // Png file signature
    }

    return 0;
}


/*
 * PngBitmapCodec::CanAutoDetect
 */
bool PngBitmapCodec::CanAutoDetect() const {
    return true;
}


/*
 * PngBitmapCodec::FileNameExtsA
 */
const char* PngBitmapCodec::FileNameExtsA() const {
    return ".png";
}


/*
 * PngBitmapCodec::FileNameExtsW
 */
const wchar_t* PngBitmapCodec::FileNameExtsW() const {
    return L".png";
}


/*
 * PngBitmapCodec::NameA
 */
const char* PngBitmapCodec::NameA() const {
    return "Portable Network Graphics";
}


/*
 * PngBitmapCodec::NameW
 */
const wchar_t* PngBitmapCodec::NameW() const {
    return L"Portable Network Graphics";
}


/*
 * PngBitmapCodec::loadFromMemory
 */
bool PngBitmapCodec::loadFromMemory(const void* mem, SIZE_T size) {

    png_structp png_ptr =
        png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, pngutil::pngErrorThrow, pngutil::pngWarningIgnore);
    if (!png_ptr)
        return false;

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return false;
    }

    png_infop end_info = png_create_info_struct(png_ptr);
    if (!end_info) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return false;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
        return false;
    }

    pngutil::pngReadBuffer rd;
    rd.buffer = static_cast<const unsigned char*>(mem);
    rd.bufferSize = size;
    rd.bufferPos = 0;

    png_set_read_fn(png_ptr, static_cast<void*>(&rd), &pngutil::pngReadData);

    png_read_png(png_ptr, info_ptr,
        PNG_TRANSFORM_STRIP_16      // 16 bit col => 8 bit col
            | PNG_TRANSFORM_PACKING // (1|2|4) bit col => 8 bit col
            | PNG_TRANSFORM_EXPAND  // Palette => RGB
        ,
        NULL);

    unsigned int bit_depth = png_get_bit_depth(png_ptr, info_ptr);
    if (bit_depth != 8) {
        return false; // currently only 8-bit is supported (input transfors should fix other depths)
    }

    unsigned int width = png_get_image_width(png_ptr, info_ptr);
    unsigned int height = png_get_image_height(png_ptr, info_ptr);
    unsigned int color_type = png_get_color_type(png_ptr, info_ptr);

    if ((color_type & PNG_COLOR_MASK_PALETTE) == PNG_COLOR_TYPE_PALETTE) {
        // palette image
        return false; // not directly supported but since PNG_TRANSFORM_EXPAND it should be ok

    } else {
        // full colour image
        unsigned int chanCnt = (((color_type & PNG_COLOR_MASK_COLOR) == PNG_COLOR_MASK_COLOR) ? 3 : 1) +
                               (((color_type & PNG_COLOR_MASK_ALPHA) == PNG_COLOR_MASK_ALPHA) ? 1 : 0);

        this->image().CreateImage(width, height, chanCnt, BitmapImage::CHANNELTYPE_BYTE);

        if ((color_type & PNG_COLOR_MASK_COLOR) == PNG_COLOR_MASK_COLOR) {
            this->image().SetChannelLabel(0, BitmapImage::CHANNEL_RED);
            this->image().SetChannelLabel(1, BitmapImage::CHANNEL_GREEN);
            this->image().SetChannelLabel(2, BitmapImage::CHANNEL_BLUE);
        } else {
            this->image().SetChannelLabel(0, BitmapImage::CHANNEL_GRAY);
        }

        if ((color_type & PNG_COLOR_MASK_ALPHA) == PNG_COLOR_MASK_ALPHA) {
            this->image().SetChannelLabel(chanCnt - 1, BitmapImage::CHANNEL_ALPHA);
        }

        png_bytepp row_pointers = png_get_rows(png_ptr, info_ptr);
        unsigned char* imgptr = this->image().PeekDataAs<unsigned char>();
        unsigned int lineSize = (width * chanCnt * 1);
        for (unsigned int y = 0; y < height; y++) {
            ::memcpy(imgptr + y * lineSize, row_pointers[y], lineSize);
        }
    }

    png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);

    return true;
}


/*
 * PngBitmapCodec::loadFromMemoryImplemented
 */
bool PngBitmapCodec::loadFromMemoryImplemented() const {
    return true;
}


/*
 * PngBitmapCodec::saveToStream
 */
bool PngBitmapCodec::saveToStream(vislib::sys::File& stream) const {
    using vislib::graphics::BitmapImage;

    png_structp png_ptr =
        ::png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, pngutil::pngErrorThrow, pngutil::pngWarningIgnore);
    if (!png_ptr)
        return false;

    png_infop info_ptr = ::png_create_info_struct(png_ptr);
    if (!info_ptr) {
        ::png_destroy_write_struct(&png_ptr, NULL);
        return false;
    }

    vislib::sys::File* sptr = &stream; // for ultimate paranoia
    ::png_set_write_fn(png_ptr, &sptr, pngutil::pngWriteData, pngutil::pngFlushData);

    if (::setjmp(png_jmpbuf(png_ptr))) {
        ::png_destroy_write_struct(&png_ptr, &info_ptr);
        return false;
    }

    const BitmapImage* src = &(this->image());
    vislib::SmartPtr<BitmapImage> alt;
    bool hasAlpha = this->image().HasAlpha();
    if (!src->EqualChannelLayout(hasAlpha ? BitmapImage::TemplateByteRGBA : BitmapImage::TemplateByteRGB)) {
        alt = new BitmapImage();
        alt->ConvertFrom(*src, hasAlpha ? BitmapImage::TemplateByteRGBA : BitmapImage::TemplateByteRGB);
        src = alt.operator->();
    }

    ::png_set_compression_level(png_ptr, Z_BEST_COMPRESSION);
    ::png_set_IHDR(png_ptr, info_ptr, this->image().Width(), this->image().Height(), 8,
        hasAlpha ? PNG_COLOR_TYPE_RGB_ALPHA : PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT);

    png_byte** rows = new png_byte*[this->image().Height()];
    unsigned int stride = this->image().Width() * (hasAlpha ? 4 : 3);
    png_byte* imgData = const_cast<png_byte*>(this->image().PeekDataAs<png_byte>());
    for (unsigned int y = 0; y < this->image().Height(); y++) {
        rows[y] = imgData;
        imgData += stride;
    }

    ::png_set_rows(png_ptr, info_ptr, rows);

    ::png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

    delete[] rows;

    ::png_destroy_write_struct(&png_ptr, &info_ptr);
    return true;
}


/*
 * PngBitmapCodec::saveToStreamImplemented
 */
bool PngBitmapCodec::saveToStreamImplemented() const {
    return true;
}
