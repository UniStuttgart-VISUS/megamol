/**
 * MegaMol
 * Copyright (c) 2010, MegaMol Dev Team
 * All rights reserved.
 */

#include "JpegBitmapCodec.h"

#include <fstream>

#include "mmcore/utility/log/Log.h"
#include "vislib/IllegalStateException.h"
#include "vislib/RawStorage.h"
#include "vislib/SmartPtr.h"
#include "vislib/graphics/BitmapImage.h"

#ifndef _WIN32
#include <jpeglib.h>
#endif

using namespace sg::graphics;
using namespace vislib::graphics;

#ifndef _WIN32
namespace sg::graphics::jpegutil {

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

} // namespace sg::graphics::jpegutil
#endif

/*
 * JpegBitmapCodec::JpegBitmapCodec
 */
JpegBitmapCodec::JpegBitmapCodec() : AbstractBitmapCodec(), quality(75) {
#ifdef _WIN32
    // Initialize COM.
    comOK = false;
    HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);

    // Create the COM imaging factory.
    if (hr != RPC_E_CHANGED_MODE) {
        hr = CoCreateInstance(CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&piFactory));
        if (SUCCEEDED(hr)) {
            comOK = true;
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Could not create WICImagingFactory. At %s:%u", __FILE__, __LINE__);
        }
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "COM already initialized with different threading model. At %s:%u", __FILE__, __LINE__);
        hr = S_FALSE;
    }
#endif
}


/*
 * JpegBitmapCodec::~JpegBitmapCodec
 */
JpegBitmapCodec::~JpegBitmapCodec() {

#ifdef _WIN32
    if (piFactory)
        piFactory->Release();
#endif
}


/*
 * JpegBitmapCodec::AutoDetect
 */
int JpegBitmapCodec::AutoDetect(const void* mem, SIZE_T size) const {
    const unsigned char* dat = static_cast<const unsigned char*>(mem);
    if (size < 3)
        return -1;
    if ((dat[0] == 0xFF) && (dat[1] == 0xD8) && (dat[2] == 0xFF)) {
        return 1; // looks like a jpeg, but I am not soooo sure
        // we could also test the application markers, but for the moment I don't care
    }
    return 0;
}


/*
 * JpegBitmapCodec::CanAutoDetect
 */
bool JpegBitmapCodec::CanAutoDetect() const {
    return true;
}


/*
 * JpegBitmapCodec::FileNameExtsA
 */
const char* JpegBitmapCodec::FileNameExtsA() const {
    return ".jpeg;.jpe;.jpg";
}


/*
 * JpegBitmapCodec::FileNameExtsW
 */
const wchar_t* JpegBitmapCodec::FileNameExtsW() const {
    return L".jpeg;.jpe;.jpg";
}


/*
 * JpegBitmapCodec::NameA
 */
const char* JpegBitmapCodec::NameA() const {
    return "Joint Photographic Experts Group";
}


/*
 * JpegBitmapCodec::NameW
 */
const wchar_t* JpegBitmapCodec::NameW() const {
    return L"Joint Photographic Experts Group";
}


/**
 * JpegBitmapCodec::OptimizeCompressionQuality
 */
void JpegBitmapCodec::OptimizeCompressionQuality() {
    using vislib::RawStorage;
    using vislib::graphics::BitmapImage;

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


#ifdef _WIN32
/*
 * JpegBitmapCodec::loadFromMemory
 */
bool JpegBitmapCodec::loadFromMemory(const void* mem, SIZE_T size) {

    IWICBitmapDecoder* piDecoder = NULL;
    HRESULT hr;

    // Create the decoder.
    if (comOK) {
        IStream* stream = SHCreateMemStream(reinterpret_cast<const BYTE*>(mem), size);

        hr = piFactory->CreateDecoderFromStream(stream, NULL,
            WICDecodeMetadataCacheOnDemand, // For JPEG lossless decoding/encoding.
            &piDecoder);

        UINT count;
        if (SUCCEEDED(hr)) {
            hr = piDecoder->GetFrameCount(&count);
        }

        if (SUCCEEDED(hr) && count > 0) {
            // Process each frame of the image.
            UINT i = 0;
            UINT width, height;
            WICPixelFormatGUID pixelFormat;
            // Frame variables.
            IWICBitmapFrameDecode* piFrameDecode = NULL;

            // Get and create the image frame.
            if (SUCCEEDED(hr)) {
                hr = piDecoder->GetFrame(i, &piFrameDecode);
            }
            // Get and set the size.
            if (SUCCEEDED(hr)) {
                hr = piFrameDecode->GetSize(&width, &height);
            }
            // Set the pixel format.
            if (SUCCEEDED(hr)) {
                hr = piFrameDecode->GetPixelFormat(&pixelFormat);
            }


            UINT stride;
            if (pixelFormat == GUID_WICPixelFormat8bppGray) {
                this->image().CreateImage(width, height, 1, BitmapImage::CHANNELTYPE_BYTE);
                this->image().SetChannelLabel(0, BitmapImage::CHANNEL_GRAY);
                stride = 1;
            } else if (pixelFormat == GUID_WICPixelFormat32bppGrayFloat) {
                this->image().CreateImage(width, height, 1, BitmapImage::CHANNELTYPE_FLOAT);
                this->image().SetChannelLabel(0, BitmapImage::CHANNEL_GRAY);
                stride = 4;
            } else if (pixelFormat == GUID_WICPixelFormat24bppRGB) {
                this->image().CreateImage(width, height, 3, BitmapImage::CHANNELTYPE_BYTE);
                this->image().SetChannelLabel(0, BitmapImage::CHANNEL_RED);
                this->image().SetChannelLabel(1, BitmapImage::CHANNEL_GREEN);
                this->image().SetChannelLabel(2, BitmapImage::CHANNEL_BLUE);
                stride = 3;
            } else if (pixelFormat == GUID_WICPixelFormat24bppBGR) {
                this->image().CreateImage(width, height, 3, BitmapImage::CHANNELTYPE_BYTE);
                this->image().SetChannelLabel(2, BitmapImage::CHANNEL_RED);
                this->image().SetChannelLabel(1, BitmapImage::CHANNEL_GREEN);
                this->image().SetChannelLabel(0, BitmapImage::CHANNEL_BLUE);
                stride = 3;
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteError("Unknown image format");
                stride = 3;
                // create error image
                this->image().CreateImage(1, 1, 3, BitmapImage::CHANNELTYPE_BYTE);
                this->image().SetChannelLabel(0, BitmapImage::CHANNEL_RED);
                this->image().SetChannelLabel(1, BitmapImage::CHANNEL_GREEN);
                this->image().SetChannelLabel(2, BitmapImage::CHANNEL_BLUE);
            }

            hr = piFrameDecode->CopyPixels(
                NULL, width * stride, width * height * stride, this->image().PeekDataAs<BYTE>());

            // release stuff
            if (piDecoder)
                piDecoder->Release();
            if (piFrameDecode)
                piFrameDecode->Release();
            if (stream)
                stream->Release();

            return true;
        }
    }
    return false;
}

/*
 * JpegBitmapCodec::saveToMemory
 */
bool JpegBitmapCodec::saveToMemory(vislib::RawStorage& mem) const {

    HRESULT hr;
    IWICBitmapFrameEncode* piFrameEncode = NULL;
    IWICBitmapEncoder* piEncoder = NULL;
    WICPixelFormatGUID pixelFormat;
    UINT stride = 0;
    BYTE* memory;
    IStream* outputStream;
    IWICStream* pStream;
    bool ret = false;

    //  Setup memory stream, which is needed to stage raw image bits
    if (CreateStreamOnHGlobal(NULL, TRUE, &outputStream) != S_OK) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Could not create output stream. At %s:%u", __FILE__, __LINE__);
    }

    hr = piFactory->CreateStream(&pStream);

    if (SUCCEEDED(hr)) {
        hr = pStream->InitializeFromIStream(outputStream);
    }
    // Create the encoder.
    if (SUCCEEDED(hr)) {
        hr = piFactory->CreateEncoder(GUID_ContainerFormatJpeg, NULL, &piEncoder);
    }
    // Initialize the encoder
    if (SUCCEEDED(hr)) {
        hr = piEncoder->Initialize(pStream, WICBitmapEncoderNoCache);
    }

    if (SUCCEEDED(hr)) {
        hr = piEncoder->CreateNewFrame(&piFrameEncode, NULL);
    }

    // Initialize the encoder.
    if (SUCCEEDED(hr)) {
        hr = piFrameEncode->Initialize(NULL);
    }

    if (SUCCEEDED(hr)) {
        hr = piFrameEncode->SetSize(this->image().Width(), this->image().Height());
    }

    if (SUCCEEDED(hr)) {
        hr = piFrameEncode->SetResolution(72, 72);
    }

    if (SUCCEEDED(hr)) {
        if (this->image().GetChannelCount() == 1) {
            if (this->image().GetChannelType() == BitmapImage::CHANNELTYPE_BYTE) {
                pixelFormat = GUID_WICPixelFormat8bppGray;
                stride = 1;
            } else if (this->image().GetChannelType() == BitmapImage::CHANNELTYPE_FLOAT) {
                pixelFormat = GUID_WICPixelFormat32bppGrayFloat;
                stride = 4;
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "Unsupported image format - channels: %u, type: %u. At %s:%u", this->image().GetChannelCount(),
                    static_cast<std::underlying_type_t<BitmapImage::ChannelType>>(this->image().GetChannelType()),
                    __FILE__, __LINE__);
            }
        } else if (this->image().GetChannelCount() == 3) {
            if (this->image().GetChannelType() == BitmapImage::CHANNELTYPE_BYTE) {
                if (this->image().GetChannelLabel(0) == BitmapImage::CHANNEL_RED &&
                    this->image().GetChannelLabel(1) == BitmapImage::CHANNEL_GREEN) {
                    pixelFormat = GUID_WICPixelFormat24bppRGB;
                    stride = 3;
                } else if (this->image().GetChannelLabel(0) == BitmapImage::CHANNEL_BLUE &&
                           this->image().GetChannelLabel(1) == BitmapImage::CHANNEL_GREEN) {
                    pixelFormat = GUID_WICPixelFormat24bppBGR;
                    stride = 3;
                } else {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "Unsupported channel labels- channels: [%u, %u, %u]. At %s:%u",
                        static_cast<std::underlying_type_t<BitmapImage::ChannelLabel>>(
                            this->image().GetChannelLabel(0)),
                        static_cast<std::underlying_type_t<BitmapImage::ChannelLabel>>(
                            this->image().GetChannelLabel(1)),
                        static_cast<std::underlying_type_t<BitmapImage::ChannelLabel>>(
                            this->image().GetChannelLabel(2)),
                        __FILE__, __LINE__);
                }
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "Unsupported channel count - channels: %u, type: %u. At %s:%u", this->image().GetChannelCount(),
                    static_cast<std::underlying_type_t<BitmapImage::ChannelType>>(this->image().GetChannelType()),
                    __FILE__, __LINE__);
            }
        }
        hr = piFrameEncode->SetPixelFormat(&pixelFormat);
    }

    if (SUCCEEDED(hr) && stride > 0) {

        hr = piFrameEncode->WritePixels(this->image().Height(), this->image().Width() * stride,
            this->image().Height() * this->image().Width() * stride,
            const_cast<BYTE*>(this->image().PeekDataAs<BYTE>()));
    }

    // Commit the frame.
    if (SUCCEEDED(hr)) {
        hr = piFrameEncode->Commit();
    }

    if (SUCCEEDED(hr)) {
        piEncoder->Commit();
    }

    if (SUCCEEDED(hr)) {
        STATSTG stats;
        outputStream->Stat(&stats, STATFLAG_NONAME);
        ULONG size = static_cast<ULONG>(stats.cbSize.QuadPart);

        mem.AssertSize(size);
        ULONG counter;
        LARGE_INTEGER origin;
        origin.QuadPart = 0;
        outputStream->Seek(origin, STREAM_SEEK_SET, NULL);
        outputStream->Read(mem, size, &counter);


        ret = (counter == size);
    }


    if (outputStream)
        outputStream->Release();
    if (pStream)
        pStream->Release();
    if (piFrameEncode)
        piFrameEncode->Release();
    if (piEncoder)
        piEncoder->Release();

    return ret;
}

#else
/*
 * JpegBitmapCodec::loadFromMemory
 */
bool JpegBitmapCodec::loadFromMemory(const void* mem, SIZE_T size) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = ::jpeg_std_error(&jerr);
    cinfo.err->error_exit = jpegutil::jpeg_error_exit_throw;
    cinfo.err->output_message = jpegutil::jpeg_output_message_no;
    ::jpeg_create_decompress(&cinfo);
    ::jpeg_mem_src(&cinfo, static_cast<unsigned char*>(const_cast<void*>(mem)), static_cast<unsigned long>(size));

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

    this->image().CreateImage(
        cinfo.output_width, cinfo.output_height, cinfo.output_components, BitmapImage::CHANNELTYPE_BYTE);
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
        JSAMPLE* ptr = this->image().PeekDataAs<JSAMPLE>() + row_stride * cinfo.output_scanline;
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
 * JpegBitmapCodec::saveToMemory
 */
bool JpegBitmapCodec::saveToMemory(vislib::RawStorage& mem) const {
    using vislib::graphics::BitmapImage;
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    cinfo.err->error_exit = jpegutil::jpeg_error_exit_throw;
    cinfo.err->output_message = jpegutil::jpeg_output_message_no;
    ::jpeg_create_compress(&cinfo);

    unsigned char* buf = NULL;
    unsigned long bufSize = 0;
    ::jpeg_mem_dest(&cinfo, &buf, &bufSize);

    const BitmapImage* src = &(this->image());
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
#endif

/*
 * JpegBitmapCodec::loadFromMemoryImplemented
 */
bool JpegBitmapCodec::loadFromMemoryImplemented() const {
    return true;
}

/*
 * JpegBitmapCodec::saveToMemoryImplemented
 */
bool JpegBitmapCodec::saveToMemoryImplemented() const {
    return true;
}

bool JpegBitmapCodec::saveToStream(vislib::sys::File& stream) const {
    vislib::RawStorage mem;
    this->saveToMemory(mem);
    auto count = stream.Write(mem, mem.GetSize());
    return count == mem.GetSize();
}

bool JpegBitmapCodec::saveToStreamImplemented() const {
    return true;
}
