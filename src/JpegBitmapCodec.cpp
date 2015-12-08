/*
 * JpegBitmapCodec.cpp
 *
 * Copyright (C) 2010 by Sebastian Grottel.
 * (Copyright (C) 2010 by VISUS (Universitaet Stuttgart))
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "imageviewer2/JpegBitmapCodec.h"
#include "jpeglib.h"
#include "vislib/SmartPtr.h"
#include "vislib/graphics/BitmapImage.h"
#include "vislib/IllegalStateException.h"
#include "vislib/RawStorage.h"

using namespace sg::graphics;
using namespace vislib::graphics;


namespace sg {
namespace graphics {
namespace jpegutil {

    /**
     * Convert "exit" to "throw"
     */
    void jpeg_error_exit_throw(j_common_ptr cinfo) {
        char buffer[JMSG_LENGTH_MAX];
        (*cinfo->err->format_message)(cinfo, buffer);
        vislib::StringA msg(buffer);
        ::jpeg_destroy(cinfo);
        throw vislib::Exception(msg, __FILE__, __LINE__);
    }

    /**
     * block any warning, trace, error of libJpeg
     */
    void jpeg_output_message_no(j_common_ptr cinfo) {
        // be silent!
    }

} /* end namespace jpegutil */
} /* end namespace graphics */
} /* end namespace sg */


/*
 * JpegBitmapCodec::JpegBitmapCodec
 */
JpegBitmapCodec::JpegBitmapCodec(void) : AbstractBitmapCodec(), quality(75) {
    // intentionally empty
}


/*
 * JpegBitmapCodec::~JpegBitmapCodec
 */
JpegBitmapCodec::~JpegBitmapCodec(void) {
    // intentionally empty
}


/*
 * JpegBitmapCodec::AutoDetect
 */
int JpegBitmapCodec::AutoDetect(const void *mem, SIZE_T size) const {
    const unsigned char *dat = static_cast<const unsigned char*>(mem);
    if (size < 3) return -1;
    if ((dat[0] == 0xFF) && (dat[1] == 0xD8) && (dat[2] == 0xFF)) {
        return 1; // looks like a jpeg, but I am not soooo sure
        // we could also test the application markers, but for the moment I don't care
    }
    return 0;
}


/*
 * JpegBitmapCodec::CanAutoDetect
 */
bool JpegBitmapCodec::CanAutoDetect(void) const {
    return true;
}


/*
 * JpegBitmapCodec::FileNameExtsA
 */
const char* JpegBitmapCodec::FileNameExtsA(void) const {
    return ".jpeg;.jpe;.jpg";
}


/*
 * JpegBitmapCodec::FileNameExtsW
 */
const wchar_t* JpegBitmapCodec::FileNameExtsW(void) const {
    return L".jpeg;.jpe;.jpg";
}


/*
 * JpegBitmapCodec::NameA
 */
const char * JpegBitmapCodec::NameA(void) const {
    return "Joint Photographic Experts Group";
}


/*
 * JpegBitmapCodec::NameW
 */
const wchar_t * JpegBitmapCodec::NameW(void) const {
    return L"Joint Photographic Experts Group";
}


/**
 * JpegBitmapCodec::OptimizeCompressionQuality
 */
void JpegBitmapCodec::OptimizeCompressionQuality(void) {
    using vislib::graphics::BitmapImage;
    using vislib::RawStorage;

    if ((this->Image() == NULL) || (this->image().Width() == 0) || (this->image().Height() == 0)) {
        throw vislib::IllegalStateException("No image data set", __FILE__, __LINE__);
    }

    JpegBitmapCodec c1, c2, *encoder(this), *decoder(&c2);
    vislib::SmartPtr<BitmapImage> i1;
    vislib::SmartPtr<BitmapImage> i2 = new BitmapImage();
    RawStorage mem;
    if (!this->image().EqualChannelLayout(BitmapImage::TemplateByteRGB)) {
        encoder = &c1;
        c1.Image() = (i1 = new BitmapImage()).operator->();
        c1.image().ConvertFrom(this->image(), BitmapImage::TemplateByteRGB);
    }
    c2.Image() = i2.operator->();


    encoder->SetCompressionQuality(100);
    if (!encoder->Save(mem) || !decoder->Load(mem)) {
        throw vislib::Exception("Internal memory IO error", __FILE__, __LINE__);
    }

    // TODO: Implement

}


/*
 * JpegBitmapCodec::loadFromMemory
 */
bool JpegBitmapCodec::loadFromMemory(const void *mem, SIZE_T size) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = ::jpeg_std_error(&jerr);
    cinfo.err->error_exit = jpegutil::jpeg_error_exit_throw;
    cinfo.err->output_message = jpegutil::jpeg_output_message_no;
    ::jpeg_create_decompress(&cinfo);
    ::jpeg_mem_src(&cinfo,
        static_cast<unsigned char*>(const_cast<void*>(mem)),
        static_cast<unsigned long>(size));

    if (::jpeg_read_header(&cinfo, TRUE) != 1) {
        ::jpeg_destroy_decompress(&cinfo);
        return false;
    }

    /* set parameters for decompression */
    /* We don't need to change any of the defaults set by
     * jpeg_read_header(), so we do nothing here.
     */

    if (!::jpeg_start_decompress(&cinfo)) {
        ::jpeg_destroy_decompress(&cinfo);
        return false;
    }

    if (sizeof(JSAMPLE) != 1) { // only support 8-bit jpegs ATM
        ::jpeg_destroy_decompress(&cinfo);
        return false;
    }
    if (cinfo.output_components < 1) {
        ::jpeg_destroy_decompress(&cinfo);
        return false;
    }

    this->image().CreateImage(cinfo.output_width, cinfo.output_height, cinfo.output_components, BitmapImage::CHANNELTYPE_BYTE);
    switch (cinfo.out_color_space) {
        case JCS_UNKNOWN:
            for (int i = 0; i < cinfo.output_components; i++) {
                this->image().SetChannelLabel(static_cast<unsigned int>(i), BitmapImage::CHANNEL_UNDEF);
            }
            break;
        case JCS_GRAYSCALE:
            for (int i = 0; i < cinfo.output_components; i++) {
                this->image().SetChannelLabel(static_cast<unsigned int>(i), BitmapImage::CHANNEL_GRAY);
            }
            break;
        case JCS_RGB:
            if (this->image().GetChannelCount() == 3) {
                this->image().SetChannelLabel(0, BitmapImage::CHANNEL_RED);
                this->image().SetChannelLabel(1, BitmapImage::CHANNEL_GREEN);
                this->image().SetChannelLabel(2, BitmapImage::CHANNEL_BLUE);
            } else {
                for (int i = 0; i < cinfo.output_components; i++) {
                    this->image().SetChannelLabel(static_cast<unsigned int>(i), BitmapImage::CHANNEL_GRAY);
                }
            }
            break;
        case JCS_YCbCr: /* Y/Cb/Cr (also known as YUV) */
            // not supported
            for (int i = 0; i < cinfo.output_components; i++) {
                this->image().SetChannelLabel(static_cast<unsigned int>(i), BitmapImage::CHANNEL_UNDEF);
            }
            break;
        case JCS_CMYK:
            if (this->image().GetChannelCount() == 4) {
                this->image().SetChannelLabel(0, BitmapImage::CHANNEL_CYAN);
                this->image().SetChannelLabel(1, BitmapImage::CHANNEL_MAGENTA);
                this->image().SetChannelLabel(2, BitmapImage::CHANNEL_YELLOW);
                this->image().SetChannelLabel(3, BitmapImage::CHANNEL_BLACK);
            } else {
                for (int i = 0; i < cinfo.output_components; i++) {
                    this->image().SetChannelLabel(static_cast<unsigned int>(i), BitmapImage::CHANNEL_GRAY);
                }
            }
            break;
        case JCS_YCCK: /* Y/Cb/Cr/K */
            // not supported
            for (int i = 0; i < cinfo.output_components; i++) {
                this->image().SetChannelLabel(static_cast<unsigned int>(i), BitmapImage::CHANNEL_UNDEF);
            }
            break;
        default: // not supported for SURE
            for (int i = 0; i < cinfo.output_components; i++) {
                this->image().SetChannelLabel(static_cast<unsigned int>(i), BitmapImage::CHANNEL_UNDEF);
            }
            break;
    }

    unsigned int row_stride = cinfo.output_width * cinfo.output_components;
    while (cinfo.output_scanline < cinfo.output_height) {
        JSAMPLE *ptr = this->image().PeekDataAs<JSAMPLE>() + row_stride * cinfo.output_scanline;
        ::jpeg_read_scanlines(&cinfo, &ptr, 1);
    }

    if (cinfo.out_color_space == JCS_CMYK) {
        this->image().Invert();
    }

    ::jpeg_finish_decompress(&cinfo);
    ::jpeg_destroy_decompress(&cinfo);

    return true;
}


/*
 * JpegBitmapCodec::loadFromMemoryImplemented
 */
bool JpegBitmapCodec::loadFromMemoryImplemented(void) const {
    return true;
}


/*
 * JpegBitmapCodec::saveToMemory
 */
bool JpegBitmapCodec::saveToMemory(vislib::RawStorage &mem) const {
    using vislib::graphics::BitmapImage;
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    cinfo.err->error_exit = jpegutil::jpeg_error_exit_throw;
    cinfo.err->output_message = jpegutil::jpeg_output_message_no;
    ::jpeg_create_compress(&cinfo);

    unsigned char *buf = NULL;
    unsigned long bufSize = 0;
    ::jpeg_mem_dest(&cinfo, &buf, &bufSize);

    const BitmapImage *src = &(this->image());
    vislib::SmartPtr<BitmapImage> alt;
    if (!src->EqualChannelLayout(BitmapImage::TemplateByteRGB)) {
        alt = new BitmapImage();
        alt->ConvertFrom(*src, BitmapImage::TemplateByteRGB);
        src = alt.operator->();
    }

    cinfo.image_width = src->Width();
    cinfo.image_height = src->Height();
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;
    ::jpeg_set_defaults(&cinfo);
    ::jpeg_set_quality(&cinfo, this->quality, TRUE);

    ::jpeg_start_compress(&cinfo, TRUE);

    JSAMPROW rowPointer[1];
    int rowStride = src->Width() * 3;
    rowPointer[0] = const_cast<JSAMPLE*>(src->PeekDataAs<JSAMPLE>());

    for (unsigned int y = 0; y < src->Height(); y++, rowPointer[0] += rowStride) {
        ::jpeg_write_scanlines(&cinfo, rowPointer, 1);
    }

    ::jpeg_finish_compress(&cinfo);
    ::jpeg_destroy_compress(&cinfo);
    mem.EnforceSize(bufSize);
    ::memcpy(mem, buf, bufSize);
    ::free(buf);

    return true;
}


/*
 * JpegBitmapCodec::saveToMemoryImplemented
 */
bool JpegBitmapCodec::saveToMemoryImplemented(void) const {
    return true;
}
