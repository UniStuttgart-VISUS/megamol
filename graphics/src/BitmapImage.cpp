/*
 * BitmapImage.cpp
 *
 * Copyright (C) 2009 by Sebastian Grottel.
 * (Copyright (C) 2009 by Visualisierungsinstitut Universitaet Stuttgart.)
 * Alle Rechte vorbehalten.
 */

#include "vislib/BitmapImage.h"
#include "vislib/assert.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/memutils.h"


/*
 * vislib::graphics::BitmapImage::Extension::Extension
 */
vislib::graphics::BitmapImage::Extension::Extension(
        vislib::graphics::BitmapImage& owner) : owner(owner) {
    // intentionally empty
}


/*
 * vislib::graphics::BitmapImage::Extension::~Extension
 */
vislib::graphics::BitmapImage::Extension::~Extension(void) {
    // intentionally empty
}



/*
 * vislib::graphics::BitmapImage::TemplateByteGray
 */
const vislib::graphics::BitmapImage
vislib::graphics::BitmapImage::TemplateByteGray(
    vislib::graphics::BitmapImage::CHANNELTYPE_BYTE,
    vislib::graphics::BitmapImage::CHANNEL_GRAY);


/*
 * vislib::graphics::BitmapImage::TemplateByteGrayAlpha
 */
const vislib::graphics::BitmapImage
vislib::graphics::BitmapImage::TemplateByteGrayAlpha(
    vislib::graphics::BitmapImage::CHANNELTYPE_BYTE,
    vislib::graphics::BitmapImage::CHANNEL_GRAY,
    vislib::graphics::BitmapImage::CHANNEL_ALPHA);


/*
 * vislib::graphics::BitmapImage::TemplateByteRGB
 */
const vislib::graphics::BitmapImage
vislib::graphics::BitmapImage::TemplateByteRGB(
    vislib::graphics::BitmapImage::CHANNELTYPE_BYTE,
    vislib::graphics::BitmapImage::CHANNEL_RED,
    vislib::graphics::BitmapImage::CHANNEL_GREEN,
    vislib::graphics::BitmapImage::CHANNEL_BLUE);


/*
 * vislib::graphics::BitmapImage::TemplateByteRGBA
 */
const vislib::graphics::BitmapImage
vislib::graphics::BitmapImage::TemplateByteRGBA(
    vislib::graphics::BitmapImage::CHANNELTYPE_BYTE,
    vislib::graphics::BitmapImage::CHANNEL_RED,
    vislib::graphics::BitmapImage::CHANNEL_GREEN,
    vislib::graphics::BitmapImage::CHANNEL_BLUE,
    vislib::graphics::BitmapImage::CHANNEL_ALPHA);


/*
 * vislib::graphics::BitmapImage::BitmapImage
 */
vislib::graphics::BitmapImage::BitmapImage(void) : data(NULL),
        chanType(CHANNELTYPE_BYTE), ext(NULL), height(0), labels(NULL),
        numChans(0), width(0) {
    // intentionally empty
}


/*
 * vislib::graphics::BitmapImage::BitmapImage
 */
vislib::graphics::BitmapImage::BitmapImage(unsigned int width,
        unsigned int height, unsigned int channels,
        vislib::graphics::BitmapImage::ChannelType type, const void *data)
        : data(NULL), chanType(CHANNELTYPE_BYTE), ext(NULL), height(0),
        labels(NULL), numChans(0), width(0) {
    this->CreateImage(width, height, channels, type, data);
}


/*
 * vislib::graphics::BitmapImage::BitmapImage
 */
vislib::graphics::BitmapImage::BitmapImage(unsigned int width,
        unsigned int height, const vislib::graphics::BitmapImage& tmpl,
        const void *data) : data(NULL), chanType(CHANNELTYPE_BYTE), ext(NULL),
        height(0), labels(NULL), numChans(0), width(0) {
    this->CreateImage(width, height, tmpl, data);
}


/*
 * vislib::graphics::BitmapImage::BitmapImage
 */
vislib::graphics::BitmapImage::BitmapImage(
        const vislib::graphics::BitmapImage& src, bool copyExt) : data(NULL),
        chanType(CHANNELTYPE_BYTE), ext(NULL), height(0), labels(NULL),
        numChans(0), width(0) {
    this->CopyFrom(src, copyExt);
}


/*
 * vislib::graphics::BitmapImage::BitmapImage
 */
vislib::graphics::BitmapImage::BitmapImage(
        vislib::graphics::BitmapImage::ChannelType type,
        vislib::graphics::BitmapImage::ChannelLabel label1) : data(NULL),
        chanType(type), ext(NULL), height(0), labels(NULL), numChans(1),
        width(0) {
    this->labels = new ChannelLabel[1];
    this->labels[0] = label1;
}


/*
 * vislib::graphics::BitmapImage::BitmapImage
 */
vislib::graphics::BitmapImage::BitmapImage(
        vislib::graphics::BitmapImage::ChannelType type,
        vislib::graphics::BitmapImage::ChannelLabel label1,
        vislib::graphics::BitmapImage::ChannelLabel label2) : data(NULL),
        chanType(type), ext(NULL), height(0), labels(NULL), numChans(2),
        width(0) {
    this->labels = new ChannelLabel[2];
    this->labels[0] = label1;
    this->labels[1] = label2;
}


/*
 * vislib::graphics::BitmapImage::BitmapImage
 */
vislib::graphics::BitmapImage::BitmapImage(
        vislib::graphics::BitmapImage::ChannelType type,
        vislib::graphics::BitmapImage::ChannelLabel label1,
        vislib::graphics::BitmapImage::ChannelLabel label2,
        vislib::graphics::BitmapImage::ChannelLabel label3) : data(NULL),
        chanType(type), ext(NULL), height(0), labels(NULL), numChans(3),
        width(0) {
    this->labels = new ChannelLabel[3];
    this->labels[0] = label1;
    this->labels[1] = label2;
    this->labels[2] = label3;
}


/*
 * vislib::graphics::BitmapImage::BitmapImage
 */
vislib::graphics::BitmapImage::BitmapImage(
        vislib::graphics::BitmapImage::ChannelType type,
        vislib::graphics::BitmapImage::ChannelLabel label1,
        vislib::graphics::BitmapImage::ChannelLabel label2,
        vislib::graphics::BitmapImage::ChannelLabel label3,
        vislib::graphics::BitmapImage::ChannelLabel label4) : data(NULL),
        chanType(type), ext(NULL), height(0), labels(NULL), numChans(4),
        width(0) {
    this->labels = new ChannelLabel[4];
    this->labels[0] = label1;
    this->labels[1] = label2;
    this->labels[2] = label3;
    this->labels[3] = label4;
}


/*
 * vislib::graphics::BitmapImage::~BitmapImage
 */
vislib::graphics::BitmapImage::~BitmapImage(void) {
    ARY_SAFE_DELETE(this->data);
    this->height = 0;   // set for paranoia reasons
    if (this->ext != NULL) {
        delete this->ext;
        this->ext = NULL;
    }
    ARY_SAFE_DELETE(this->labels);
    this->numChans = 0;
    this->width = 0;
}


/*
 * vislib::graphics::BitmapImage::CopyFrom
 */
void vislib::graphics::BitmapImage::CopyFrom(
        const vislib::graphics::BitmapImage& src, bool copyExt) {
    unsigned int len = src.width * src.height * src.BytesPerPixel();
    ARY_SAFE_DELETE(this->data);
    this->data = new char[len];
    memcpy(this->data, src.data, len);
    if (copyExt && (src.ext != NULL)) {
        this->SetExtension(src.ext->Clone(*this));
    } else {
        this->SetExtension(NULL);
    }
    this->chanType = src.chanType;
    this->height = src.height;
    ARY_SAFE_DELETE(this->labels);
    this->labels = new ChannelLabel[src.numChans];
    memcpy(this->labels, src.labels, sizeof(ChannelLabel) * src.numChans);
    this->numChans = src.numChans;
    this->width = src.width;
}


/*
 * vislib::graphics::BitmapImage::Convert
 */
void vislib::graphics::BitmapImage::Convert(const BitmapImage& tmpl) {
    this->ConvertFrom(*this, tmpl);
}


/*
 * vislib::graphics::BitmapImage::ConvertFrom
 */
void vislib::graphics::BitmapImage::ConvertFrom(const BitmapImage& src,
        const BitmapImage& tmpl) {

    // test for equality
    if ((tmpl.chanType == src.chanType) && (tmpl.numChans == src.numChans)
        && (::memcmp(tmpl.labels, src.labels,
            tmpl.numChans * sizeof(ChannelLabel)) == 0)) {
        // no conversion required
        if (&src != this) {
            this->CopyFrom(src);
        } // else we are the same object and no action is required
        return;
    }

    // perform a conversion
    unsigned int tmplStep = tmpl.BytesPerPixel();
    unsigned int srcStep = src.BytesPerPixel();
    unsigned int imgSize = src.width * src.height;
    unsigned int bufLen = imgSize * tmplStep;
    char *buf = new char[bufLen];

    // channel value mapping
    int *chMap = new int[tmpl.numChans];
    for (unsigned int i = 0; i < tmpl.numChans; i++) {
        chMap[i] = (tmpl.labels[i] == CHANNEL_ALPHA)
            ? -2  // init channel with white
            : -1; // init channel with black
        for (unsigned int j = 0; j < src.numChans; j++) {
            if (tmpl.labels[i] == src.labels[j]) {
                chMap[i] = j; // copy value
                break;
            }
        }
    }

    if (tmpl.chanType == src.chanType) {
        unsigned int chanSize = tmpl.BytesPerPixel() / tmpl.numChans;
        char *zero = new char[chanSize];
        char *one = new char[chanSize];
        char *ptr = NULL;
        switch (tmpl.chanType) {
            case CHANNELTYPE_BYTE:
                zero[0] = 0;
                reinterpret_cast<unsigned char*>(one)[0] = 0xff;
                break;
            case CHANNELTYPE_WORD:
                reinterpret_cast<unsigned short*>(zero)[0] = 0;
                reinterpret_cast<unsigned short*>(one)[0] = 0xffff;
                break;
            case CHANNELTYPE_FLOAT:
                reinterpret_cast<float*>(zero)[0] = 0.0f;
                reinterpret_cast<float*>(one)[0] = 1.0f;
                break;
        }
        for (unsigned int i = 0; i < imgSize; i++) {
            char *srcPixel = &src.data[i * srcStep];
            char *dstPixel = &buf[i * tmplStep];
            for (unsigned int j = 0; j < tmpl.numChans; j++) {
                if (chMap[j] == -2) ptr = one;
                else if (chMap[j] >= 0) {
                    ptr = &srcPixel[chMap[j] * chanSize];
                } else ptr = zero;
                ::memcpy(&dstPixel[j * chanSize], ptr, chanSize);
            }
        }
        delete[] one;
        delete[] zero;

    } else {

        for (unsigned int i = 0; i < imgSize; i++) {
            char *srcPixel = &src.data[i * srcStep];
            char *dstPixel = &buf[i * tmplStep];
            for (unsigned int j = 0; j < tmpl.numChans; j++) {
                float value = 0.0f;
                if (chMap[j] == -2) value = 1.0f;
                else if (chMap[j] >= 0) {
                    switch (src.chanType) {
                        case CHANNELTYPE_BYTE:
                            value = static_cast<float>(
                                reinterpret_cast<unsigned char*>(srcPixel)[
                                    chMap[j]]) / static_cast<float>(0xff);
                            break;
                        case CHANNELTYPE_WORD:
                            value = static_cast<float>(
                                reinterpret_cast<unsigned short*>(srcPixel)[
                                    chMap[j]]) / static_cast<float>(0xffff);
                            break;
                        case CHANNELTYPE_FLOAT:
                            value = reinterpret_cast<float*>(srcPixel)[chMap[j]];
                            break;
                    }
                }

                switch (tmpl.chanType) {
                    case CHANNELTYPE_BYTE:
                        if (value < 0.0f) value = 0.0f;
                        else if (value > 1.0f) value = 1.0f;
                        value *= static_cast<float>(0xff);
                        reinterpret_cast<unsigned char*>(dstPixel)[j]
                            = static_cast<unsigned char>(value);
                        break;
                    case CHANNELTYPE_WORD:
                        if (value < 0.0f) value = 0.0f;
                        else if (value > 1.0f) value = 1.0f;
                        value *= static_cast<float>(0xffff);
                        reinterpret_cast<unsigned short*>(dstPixel)[j]
                            = static_cast<unsigned short>(value);
                        break;
                    case CHANNELTYPE_FLOAT:
                        reinterpret_cast<float*>(dstPixel)[j] = value;
                        break;
                }
            }
        }

    }

    this->chanType = tmpl.chanType;
    delete[] this->data;
    this->data = buf;
    this->height = src.height;
    ChannelLabel *tmplLabels = tmpl.labels;
    ChannelLabel *oldLabels = this->labels;
    this->labels = new ChannelLabel[tmpl.numChans];
    ::memcpy(this->labels, tmplLabels, this->numChans * sizeof(ChannelLabel));
    delete[] oldLabels;
    this->numChans = tmpl.numChans;
    this->width = src.width;
}


/*
 * vislib::graphics::BitmapImage::Crop
 */
void vislib::graphics::BitmapImage::Crop(unsigned int left, unsigned int top,
        unsigned int width, unsigned int height) {
    if (left >= this->width) {
        left = this->width - 1;
    }
    if (width < 1) width = 1;
    if (left + width > this->width) {
        width = this->width - left;
    }
    if (top >= this->height) {
        top = this->height -1;
    }
    if (height < 1) height = 1;
    if (top + height > this->height) {
        height = this->height - top;
    }
    unsigned int bpp = this->BytesPerPixel();
    if (bpp < 1) throw vislib::IllegalStateException(
        "Image has no colour channels", __FILE__, __LINE__);

    char *newData = new char[bpp * width * height];

    try {
        this->cropCopy(newData, this->data, this->width, this->height,
            left, top, width, height, bpp);
        delete[] this->data;
        this->data = newData;
        this->width = width;
        this->height = height;

    } catch(...) {
        delete[] newData;
        throw;
    }

}


/*
 * vislib::graphics::BitmapImage::CreateImage
 */
void vislib::graphics::BitmapImage::CreateImage(unsigned int width,
        unsigned int height, unsigned int channels,
        vislib::graphics::BitmapImage::ChannelType type, const void *data) {
    ARY_SAFE_DELETE(this->data);
    ARY_SAFE_DELETE(this->labels);
    this->chanType = type;
    this->height = height;
    this->numChans = channels;
    this->width = width;

    this->labels = new ChannelLabel[channels];
    for (unsigned int i = 0; i < channels; i++) {
        this->labels[i] = CHANNEL_UNDEF;
    }

    unsigned int len = this->width * this->height * this->BytesPerPixel();
    this->data = new char[len];
    if (data != NULL) {
        memcpy(this->data, data, len);
    } else {
        ZeroMemory(this->data, len);
    }
}


/*
 * vislib::graphics::BitmapImage::CreateImage
 */
void vislib::graphics::BitmapImage::CreateImage(unsigned int width,
        unsigned int height, const vislib::graphics::BitmapImage& tmpl,
        const void *data) {
    ASSERT(&tmpl != this);

    ChannelLabel *tmplLabels = new ChannelLabel[tmpl.numChans];
    ::memcpy(tmplLabels, tmpl.labels, tmpl.numChans * sizeof(ChannelLabel));

    this->CreateImage(width, height, tmpl.numChans, tmpl.chanType, data);

    ::memcpy(this->labels, tmplLabels, tmpl.numChans * sizeof(ChannelLabel));
    delete[] tmplLabels;
}


/*
 * vislib::graphics::BitmapImage::ExtractFrom
 */
void vislib::graphics::BitmapImage::ExtractFrom(
        const vislib::graphics::BitmapImage& src, unsigned int left,
        unsigned int top, unsigned int width, unsigned int height) {
    if (left >= src.Width()) {
        left = src.Width() - 1;
    }
    if (width < 1) width = 1;
    if (left + width > src.Width()) {
        width = src.Width() - left;
    }
    if (top >= src.Height()) {
        top = src.Height() -1;
    }
    if (height < 1) height = 1;
    if (top + height > src.Height()) {
        height = src.Height() - top;
    }
    unsigned int bpp = src.BytesPerPixel();
    if (bpp < 1) throw vislib::IllegalStateException(
        "Source image has no colour channels", __FILE__, __LINE__);

    this->CreateImage(width, height, src);
    this->cropCopy(this->data, src.data, src.Width(), src.Height(),
        left, top, width, height, bpp);

}


/*
 * vislib::graphics::BitmapImage::FlipVertical
 */
void vislib::graphics::BitmapImage::FlipVertical(void) {
    unsigned int lineSize = this->width * this->BytesPerPixel();
    unsigned int hh = this->height / 2;
    unsigned int mh = this->height - 1;
    unsigned char *tmpbuf = new unsigned char[lineSize];

    for (unsigned int y = 0; y < hh; y++) {
        memcpy(tmpbuf, this->data + (y * lineSize), lineSize);
        memcpy(this->data + (y * lineSize), this->data + ((mh - y) * lineSize), lineSize);
        memcpy(this->data + ((mh - y) * lineSize), tmpbuf, lineSize);
    }

    delete[] tmpbuf;
}


/*
 * vislib::graphics::BitmapImage::SetExtension
 */
void vislib::graphics::BitmapImage::SetExtension(
        vislib::graphics::BitmapImage::Extension *ext) {
    if (ext != NULL) {
        if (this->ext != ext) {
            if (&(ext->Owner()) != this) {
                throw vislib::IllegalParamException("ext owner != image", __FILE__, __LINE__);
            }
            if (this->ext != NULL) {
                delete this->ext;
            }
            this->ext = ext;
        }

    } else if (this->ext != NULL) {
        delete this->ext;
        this->ext = NULL;
    }
}


/*
 * vislib::graphics::BitmapImage::cropCopy
 */
void vislib::graphics::BitmapImage::cropCopy(char *to, char *from,
        unsigned int fromWidth, unsigned int fromHeight, unsigned int cropX,
        unsigned int cropY, unsigned int cropWidth, unsigned int cropHeight,
        unsigned int bpp) {
    ASSERT(cropX < fromWidth);
    ASSERT(cropY < fromHeight);
    ASSERT(cropX + cropWidth <= fromWidth);
    ASSERT(cropY + cropHeight <= fromHeight);
    ASSERT((cropWidth < fromWidth) || (cropHeight < fromHeight));
    ASSERT(bpp > 0);
    ASSERT(to != NULL);
    ASSERT(from != NULL);

    from += (cropY * fromWidth * bpp); // skip 'cropY' lines
    for (unsigned int y = 0; y < cropHeight; y++) {
        ::memcpy(to, from + (cropX * bpp), cropWidth * bpp);
        from += fromWidth * bpp;
        to += cropWidth * bpp;
    }

}
