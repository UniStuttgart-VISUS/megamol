/*
 * BitmapImage.cpp
 *
 * Copyright (C) 2009 by Sebastian Grottel.
 * (Copyright (C) 2009 by Visualisierungsinstitut Universitaet Stuttgart.)
 * Alle Rechte vorbehalten.
 */

#include "vislib/BitmapImage.h"
#include "vislib/memutils.h"
#ifdef _WIN32
#include <windows.h>
#endif /* _WIN32 */


/*
 * vislib::graphics::BitmapImage::BitmapImage
 */
vislib::graphics::BitmapImage::BitmapImage(unsigned int width,
        unsigned int height, unsigned int channels,
        vislib::graphics::BitmapImage::ChannelType type, const void *data)
        : data(NULL), chanType(CHANNELTYPE_BYTE), height(0), labels(NULL), 
        numChans(0), width(0) {
    this->CreateImage(width, height, channels, type, data);
}


/*
 * vislib::graphics::BitmapImage::BitmapImage
 */
vislib::graphics::BitmapImage::BitmapImage(
        const vislib::graphics::BitmapImage& src) : data(NULL),
        chanType(CHANNELTYPE_BYTE), height(0), labels(NULL), numChans(0),
        width(0) {
    this->CopyFrom(src);
}


/*
 * vislib::graphics::BitmapImage::~BitmapImage
 */
vislib::graphics::BitmapImage::~BitmapImage(void) {
    ARY_SAFE_DELETE(this->data);
    this->height = 0;   // set for paranoia reasons
    ARY_SAFE_DELETE(this->labels);
    this->numChans = 0;
    this->width = 0;
}


/*
 * vislib::graphics::BitmapImage::CopyFrom
 */
void vislib::graphics::BitmapImage::CopyFrom(
        const vislib::graphics::BitmapImage& src) {
    unsigned int len = src.width * src.height * src.BytesPerPixel();
    ARY_SAFE_DELETE(this->data);
    this->data = new char[len];
    memcpy(this->data, src.data, len);
    this->chanType = src.chanType;
    this->height = src.height;
    ARY_SAFE_DELETE(this->labels);
    this->labels = new ChannelLabel[src.numChans];
    memcpy(this->labels, src.labels, sizeof(ChannelLabel) * src.numChans);
    this->numChans = src.numChans;
    this->width = src.width;
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
