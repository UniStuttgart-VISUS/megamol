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
#include "vislib/mathfunctions.h"
#include "vislib/memutils.h"
#include <climits>

/****************************************************************************/


/*
 * vislib::graphics::BitmapImage::Conversion<ST>::Conversion
 */
template<class ST>
vislib::graphics::BitmapImage::Conversion<ST>::Conversion(ST* source,
        unsigned int chanCnt) : source(source), sourceChanCnt(chanCnt) {
    for (int i = 0; i < static_cast<int>(SC_LASTCHANNEL); i++) {
        this->func[i] = NULL;
        this->param[i] = 0;
    }
}


/*
 * vislib::graphics::BitmapImage::Conversion<ST>::~Conversion
 */
template<class ST>
vislib::graphics::BitmapImage::Conversion<ST>::~Conversion(void) {
    this->source = NULL; // DO NOT DELETE
}


/*
 * vislib::graphics::BitmapImage::Conversion<ST>::AddSourceChannel
 */
template<class ST>
void vislib::graphics::BitmapImage::Conversion<ST>::AddSourceChannel(
        unsigned int chan, ChannelLabel label) {
    SourceChannel sc = SC_LASTCHANNEL;
    switch (label) {
        case CHANNEL_UNDEF: sc = SC_UNDEFINED; break;
        case CHANNEL_RED: sc = SC_RED; break;
        case CHANNEL_GREEN: sc = SC_GREEN; break;
        case CHANNEL_BLUE: sc = SC_BLUE; break;
        case CHANNEL_GRAY: sc = SC_GRAY; break;
        case CHANNEL_ALPHA: sc = SC_ALPHA; break;
        case CHANNEL_CYAN: sc = SC_CMYK_CYAN; break;
        case CHANNEL_MAGENTA: sc = SC_CMYK_MAGENTA; break;
        case CHANNEL_YELLOW: sc = SC_CMYK_YELLOW; break;
        case CHANNEL_BLACK: sc = SC_CMYK_BLACK; break;
        default: sc = SC_LASTCHANNEL; break;
    }
    if (sc == SC_LASTCHANNEL) return;
    int i = static_cast<int>(sc);
    if (this->func[i] == NULL) {
        this->func[i] = &Conversion<ST>::directSource;
        this->param[i] = chan;
    }
}


/*
 * vislib::graphics::BitmapImage::Conversion<ST>::FinalizeInitialization
 */
template<class ST> void
vislib::graphics::BitmapImage::Conversion<ST>::FinalizeInitialization(void) {
    bool gray = (this->func[SC_GRAY] != NULL);
    bool rgb = (this->func[SC_RED] != NULL) && (this->func[SC_GREEN] != NULL)
        && (this->func[SC_BLUE] != NULL);
    bool cmyk = (this->func[SC_CMYK_CYAN] != NULL)
        && (this->func[SC_CMYK_MAGENTA] != NULL)
        && (this->func[SC_CMYK_YELLOW] != NULL)
        && (this->func[SC_CMYK_BLACK] != NULL);
    bool cmy = (this->func[SC_CMYK_CYAN] != NULL)
        && (this->func[SC_CMYK_MAGENTA] != NULL)
        && (this->func[SC_CMYK_YELLOW] != NULL)
        && (this->func[SC_CMYK_BLACK] == NULL);

    if (this->func[SC_UNDEFINED] == NULL) {
        this->func[SC_UNDEFINED] = &Conversion<ST>::constZero;
    }
    if (this->func[CHANNEL_ALPHA] == NULL) {
        this->func[CHANNEL_ALPHA] = &Conversion<ST>::constOne;
    }

    if (cmy) {
        this->func[SC_CMY_CYAN] = this->func[SC_CMYK_CYAN];
        this->param[SC_CMY_CYAN] = this->param[SC_CMYK_CYAN];
        this->func[SC_CMY_MAGENTA] = this->func[SC_CMYK_MAGENTA];
        this->param[SC_CMY_MAGENTA] = this->param[SC_CMYK_MAGENTA];
        this->func[SC_CMY_YELLOW] = this->func[SC_CMYK_YELLOW];
        this->param[SC_CMY_YELLOW] = this->param[SC_CMYK_YELLOW];
    }

    if (!gray) {
        if (rgb || cmy || cmyk) {
            // convert gray from rgb
            this->func[SC_GRAY] = &Conversion<ST>::grayFromRGB;

        } else {
            this->func[SC_GRAY] = &Conversion<ST>::constZero;
        }
    }

    if (!rgb) {
        if (cmy || cmyk) {
            // convert rgb from cmy
            this->func[SC_RED] = &Conversion<ST>::rgbFromCMY;
            this->param[SC_RED] = 0;
            this->func[SC_GREEN] = &Conversion<ST>::rgbFromCMY;
            this->param[SC_GREEN] = 1;
            this->func[SC_BLUE] = &Conversion<ST>::rgbFromCMY;
            this->param[SC_BLUE] = 2;

        } else if (gray) {
            // 'convert' rgb from gray
            this->func[SC_RED] = this->func[SC_GRAY];
            this->param[SC_RED] = this->param[SC_GRAY];
            this->func[SC_GREEN] = this->func[SC_GRAY];
            this->param[SC_GREEN] = this->param[SC_GRAY];
            this->func[SC_BLUE] = this->func[SC_GRAY];
            this->param[SC_BLUE] = this->param[SC_GRAY];

        } else {
            this->func[SC_RED] = &Conversion<ST>::constZero;
            this->func[SC_GREEN] = &Conversion<ST>::constZero;
            this->func[SC_BLUE] = &Conversion<ST>::constZero;
        }
    }

    if (!cmyk) {
        if (cmy || rgb || gray) {
            // convert cmyk from cmy
            this->func[SC_CMYK_CYAN] = &Conversion<ST>::cmykFromCMY;
            this->param[SC_CMYK_CYAN] = 0;
            this->func[SC_CMYK_MAGENTA] = &Conversion<ST>::cmykFromCMY;
            this->param[SC_CMYK_MAGENTA] = 1;
            this->func[SC_CMYK_YELLOW] = &Conversion<ST>::cmykFromCMY;
            this->param[SC_CMYK_YELLOW] = 2;
            this->func[SC_CMYK_BLACK] = &Conversion<ST>::cmykFromCMY;
            this->param[SC_CMYK_BLACK] = 3;

        } else {
            this->func[SC_CMYK_CYAN] = &Conversion<ST>::constOne;
            this->func[SC_CMYK_MAGENTA] = &Conversion<ST>::constOne;
            this->func[SC_CMYK_YELLOW] = &Conversion<ST>::constOne;
            this->func[SC_CMYK_BLACK] = &Conversion<ST>::constOne;
        }
    }

    if (!cmy) {
        if (cmyk) {
            // convert cmy from cmyk
            this->func[SC_CMY_CYAN] = &Conversion<ST>::cmyFromCMYK;
            this->param[SC_CMY_CYAN] = 0;
            this->func[SC_CMY_MAGENTA] = &Conversion<ST>::cmyFromCMYK;
            this->param[SC_CMY_MAGENTA] = 1;
            this->func[SC_CMY_YELLOW] = &Conversion<ST>::cmyFromCMYK;
            this->param[SC_CMY_YELLOW] = 2;

        } else if (rgb || gray) {
            // convert cmy from rgb
            this->func[SC_CMY_CYAN] = &Conversion<ST>::cmyFromRGB;
            this->param[SC_CMY_CYAN] = 0;
            this->func[SC_CMY_MAGENTA] = &Conversion<ST>::cmyFromRGB;
            this->param[SC_CMY_MAGENTA] = 1;
            this->func[SC_CMY_YELLOW] = &Conversion<ST>::cmyFromRGB;
            this->param[SC_CMY_YELLOW] = 2;

        } else {
            this->func[SC_CMY_CYAN] = &Conversion<ST>::constOne;
            this->func[SC_CMY_MAGENTA] = &Conversion<ST>::constOne;
            this->func[SC_CMY_YELLOW] = &Conversion<ST>::constOne;
        }
    }

}


/*
 * vislib::graphics::BitmapImage::Conversion<ST>::ChannelMapping
 */
template<class ST>
void vislib::graphics::BitmapImage::Conversion<ST>::ChannelMapping(
        typename vislib::graphics::BitmapImage::Conversion<ST>::SourceChannel *map,
        vislib::graphics::BitmapImage::ChannelLabel *chan, unsigned int cnt) {
    bool hasBlack = false;
    for (unsigned int i = 0; i < cnt; i++) {
        if (chan[i] == CHANNEL_BLACK) {
            hasBlack = true;
        }
    }
    for (unsigned int i = 0; i < cnt; i++) {
        switch(chan[i]) {
            case CHANNEL_UNDEF: map[i] = SC_UNDEFINED; break;
            case CHANNEL_RED: map[i] = SC_RED; break;
            case CHANNEL_GREEN: map[i] = SC_GREEN; break;
            case CHANNEL_BLUE: map[i] = SC_BLUE; break;
            case CHANNEL_GRAY: map[i] = SC_GRAY; break;
            case CHANNEL_ALPHA: map[i] = SC_ALPHA; break;
            case CHANNEL_CYAN: map[i] = hasBlack ? SC_CMYK_CYAN : SC_CMY_CYAN; break;
            case CHANNEL_MAGENTA: map[i] = hasBlack ? SC_CMYK_MAGENTA : SC_CMY_MAGENTA; break;
            case CHANNEL_YELLOW: map[i] = hasBlack ? SC_CMYK_YELLOW : SC_CMY_YELLOW; break;
            case CHANNEL_BLACK: map[i] = SC_CMYK_BLACK; break;
            default: map[i] = SC_UNDEFINED; break;
        }
    }

}


/*
 * vislib::graphics::BitmapImage::Conversion<ST>::constOne
 */
template<class ST>
float vislib::graphics::BitmapImage::Conversion<ST>::constOne(
        vislib::graphics::BitmapImage::Conversion<ST> *conv, int param) {
    return 1.0f;
}


/*
 * vislib::graphics::BitmapImage::Conversion<ST>::constZero
 */
template<class ST>
float vislib::graphics::BitmapImage::Conversion<ST>::constZero(
        vislib::graphics::BitmapImage::Conversion<ST> *conv, int param) {
    return 0.0f;
}


/*
 * vislib::graphics::BitmapImage::Conversion<ST>::directSource
 */
template<class ST>
float vislib::graphics::BitmapImage::Conversion<ST>::directSource(
        vislib::graphics::BitmapImage::Conversion<ST> *conv, int param) {
    float rv;
    Conversion<ST>::CopyBit(rv, conv->source[param]);
    return rv;
}


/*
 * vislib::graphics::BitmapImage::Conversion<ST>::grayFromRGB
 */
template<class ST>
float vislib::graphics::BitmapImage::Conversion<ST>::grayFromRGB(
        vislib::graphics::BitmapImage::Conversion<ST> *conv, int param) {
    return vislib::math::Max(vislib::math::Max(
        conv->func[SC_RED](conv, conv->param[SC_RED]),
        conv->func[SC_GREEN](conv, conv->param[SC_GREEN])),
        conv->func[SC_BLUE](conv, conv->param[SC_BLUE]));
}


/*
 * vislib::graphics::BitmapImage::Conversion<ST>::rgbFromCMY
 */
template<class ST>
float vislib::graphics::BitmapImage::Conversion<ST>::rgbFromCMY(
        vislib::graphics::BitmapImage::Conversion<ST> *conv, int param) {
    ASSERT((param >= 0) && (param <= 2));
    float Cmy = conv->func[SC_CMY_CYAN](conv, conv->param[SC_CMY_CYAN]);
    float cMy = conv->func[SC_CMY_MAGENTA](conv, conv->param[SC_CMY_MAGENTA]);
    float cmY = conv->func[SC_CMY_YELLOW](conv, conv->param[SC_CMY_YELLOW]);

    // http://www.easyrgb.com/index.php?X=MATH&H=13#text13

    float Rgb = 1.0f - Cmy;
    float rGb = 1.0f - cMy;
    float rgB = 1.0f - cmY;

    switch (param) {
        case 0: return Rgb;
        case 1: return rGb;
        case 2: return rgB;
    }
    return 0.0f;
}


/*
 * vislib::graphics::BitmapImage::Conversion<ST>::cmyFromRGB
 */
template<class ST>
float vislib::graphics::BitmapImage::Conversion<ST>::cmyFromRGB(
        vislib::graphics::BitmapImage::Conversion<ST> *conv, int param) {
    ASSERT((param >= 0) && (param <= 2));
    float Rgb = conv->func[SC_RED](conv, conv->param[SC_RED]);
    float rGb = conv->func[SC_GREEN](conv, conv->param[SC_GREEN]);
    float rgB = conv->func[SC_BLUE](conv, conv->param[SC_BLUE]);

    // http://www.easyrgb.com/index.php?X=MATH&H=13#text13

    float Cmy = 1.0f - Rgb;
    float cMy = 1.0f - rGb;
    float cmY = 1.0f - rgB;

    switch (param) {
        case 0: return Cmy;
        case 1: return cMy;
        case 2: return cmY;
    }
    return 0.0f;
}


/*
 * vislib::graphics::BitmapImage::Conversion<ST>::cmyFromCMYK
 */
template<class ST>
float vislib::graphics::BitmapImage::Conversion<ST>::cmyFromCMYK(
        vislib::graphics::BitmapImage::Conversion<ST> *conv, int param) {
    ASSERT((param >= 0) && (param <= 2));
    float Cmyk = conv->func[SC_CMYK_CYAN](conv, conv->param[SC_CMYK_CYAN]);
    float cMyk = conv->func[SC_CMYK_MAGENTA](conv, conv->param[SC_CMYK_MAGENTA]);
    float cmYk = conv->func[SC_CMYK_YELLOW](conv, conv->param[SC_CMYK_YELLOW]);
    float cmyK = conv->func[SC_CMYK_BLACK](conv, conv->param[SC_CMYK_BLACK]);

    // http://www.easyrgb.com/index.php?X=MATH&H=13#text13

    float Cmy = Cmyk * (1.0f - cmyK) + cmyK;
    float cMy = cMyk * (1.0f - cmyK) + cmyK;
    float cmY = cmYk * (1.0f - cmyK) + cmyK;

    switch (param) {
        case 0: return Cmy;
        case 1: return cMy;
        case 2: return cmY;
    }
    return 0.0f;
}


/*
 * vislib::graphics::BitmapImage::Conversion<ST>::cmykFromCMY
 */
template<class ST>
float vislib::graphics::BitmapImage::Conversion<ST>::cmykFromCMY(
        vislib::graphics::BitmapImage::Conversion<ST> *conv, int param) {
    ASSERT((param >= 0) && (param <= 3));
    float Cmy = conv->func[SC_CMY_CYAN](conv, conv->param[SC_CMY_CYAN]);
    float cMy = conv->func[SC_CMY_MAGENTA](conv, conv->param[SC_CMY_MAGENTA]);
    float cmY = conv->func[SC_CMY_YELLOW](conv, conv->param[SC_CMY_YELLOW]);

    // http://www.easyrgb.com/index.php?X=MATH&H=13#text13

    float cmyK = vislib::math::Min(vislib::math::Min(Cmy, cMy), cmY);
    float Cmyk = 0.0f;
    float cMyk = 0.0f;
    float cmYk = 0.0f;
    if (!vislib::math::IsEqual(cmyK, 1.0f)) {
        Cmyk = (Cmy - cmyK) / (1.0f - cmyK);
        cMyk = (cMy - cmyK) / (1.0f - cmyK);
        cmYk = (cmY - cmyK) / (1.0f - cmyK);
    }

    switch (param) {
        case 0: return Cmyk;
        case 1: return cMyk;
        case 2: return cmYk;
        case 3: return cmyK;
    }
    return 0.0f;
}

/****************************************************************************/


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

/****************************************************************************/


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
 * vislib::graphics::BitmapImage::TemplateFloatGray
 */
const vislib::graphics::BitmapImage
vislib::graphics::BitmapImage::TemplateFloatGray(
    vislib::graphics::BitmapImage::CHANNELTYPE_FLOAT,
    vislib::graphics::BitmapImage::CHANNEL_GRAY);


/*
 * vislib::graphics::BitmapImage::TemplateFloatGrayAlpha
 */
const vislib::graphics::BitmapImage
vislib::graphics::BitmapImage::TemplateFloatGrayAlpha(
    vislib::graphics::BitmapImage::CHANNELTYPE_FLOAT,
    vislib::graphics::BitmapImage::CHANNEL_GRAY,
    vislib::graphics::BitmapImage::CHANNEL_ALPHA);


/*
 * vislib::graphics::BitmapImage::TemplateFloatRGB
 */
const vislib::graphics::BitmapImage
vislib::graphics::BitmapImage::TemplateFloatRGB(
    vislib::graphics::BitmapImage::CHANNELTYPE_FLOAT,
    vislib::graphics::BitmapImage::CHANNEL_RED,
    vislib::graphics::BitmapImage::CHANNEL_GREEN,
    vislib::graphics::BitmapImage::CHANNEL_BLUE);


/*
 * vislib::graphics::BitmapImage::TemplateFloatRGBA
 */
const vislib::graphics::BitmapImage
vislib::graphics::BitmapImage::TemplateFloatRGBA(
    vislib::graphics::BitmapImage::CHANNELTYPE_FLOAT,
    vislib::graphics::BitmapImage::CHANNEL_RED,
    vislib::graphics::BitmapImage::CHANNEL_GREEN,
    vislib::graphics::BitmapImage::CHANNEL_BLUE,
    vislib::graphics::BitmapImage::CHANNEL_ALPHA);


/*
 * vislib::graphics::BitmapImage::BitmapImage
 */
vislib::graphics::BitmapImage::BitmapImage(void) : data(NULL),
        chanType(CHANNELTYPE_BYTE), exts(), height(0), labels(NULL),
        numChans(0), width(0) {
    // intentionally empty
}


/*
 * vislib::graphics::BitmapImage::BitmapImage
 */
vislib::graphics::BitmapImage::BitmapImage(unsigned int width,
        unsigned int height, unsigned int channels,
        vislib::graphics::BitmapImage::ChannelType type, const void *data)
        : data(NULL), chanType(CHANNELTYPE_BYTE), exts(), height(0),
        labels(NULL), numChans(0), width(0) {
    this->CreateImage(width, height, channels, type, data);
}


/*
 * vislib::graphics::BitmapImage::BitmapImage
 */
vislib::graphics::BitmapImage::BitmapImage(unsigned int width,
        unsigned int height, const vislib::graphics::BitmapImage& tmpl,
        const void *data) : data(NULL), chanType(CHANNELTYPE_BYTE), exts(),
        height(0), labels(NULL), numChans(0), width(0) {
    this->CreateImage(width, height, tmpl, data);
}


/*
 * vislib::graphics::BitmapImage::BitmapImage
 */
vislib::graphics::BitmapImage::BitmapImage(
        const vislib::graphics::BitmapImage& src, bool copyExt) : data(NULL),
        chanType(CHANNELTYPE_BYTE), exts(), height(0), labels(NULL),
        numChans(0), width(0) {
    this->CopyFrom(src, copyExt);
}


/*
 * vislib::graphics::BitmapImage::BitmapImage
 */
vislib::graphics::BitmapImage::BitmapImage(
        vislib::graphics::BitmapImage::ChannelType type,
        vislib::graphics::BitmapImage::ChannelLabel label1) : data(NULL),
        chanType(type), exts(), height(0), labels(NULL), numChans(1),
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
        chanType(type), exts(), height(0), labels(NULL), numChans(2),
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
        chanType(type), exts(), height(0), labels(NULL), numChans(3),
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
        chanType(type), exts(), height(0), labels(NULL), numChans(4),
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
    this->exts.Clear();
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
    this->exts.Clear();
    if (copyExt && (!src.exts.IsEmpty())) {
        SIZE_T eCnt = src.exts.Count();
        for (SIZE_T i = 0; i < eCnt; i++) {
            this->exts.Append(src.exts[i]->Clone(*this));
        }
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
    bool chanTypeEq = (tmpl.chanType == src.chanType);

    // test for equality
    if (chanTypeEq && (tmpl.numChans == src.numChans)
        && (::memcmp(tmpl.labels, src.labels,
            tmpl.numChans * sizeof(ChannelLabel)) == 0)) {
        // no conversion required

        if (&src != this) {
            this->CopyFrom(src);
        } // else we are the same object and no action is required

        return;
    }

    // test if simple re-sorting is sufficient
    bool allChanPresent = true;
    for (unsigned int i = 0; i < tmpl.GetChannelCount(); i++) {
        bool found = false;

        for (unsigned int j = 0; j < src.GetChannelCount(); j++) {
            if (src.GetChannelLabel(j) == tmpl.GetChannelLabel(i)) {
                found = true;
                break;
            }
        }

        if (!found) {
            allChanPresent = false;
            break;
        }
    }

    char *buf = new char[src.Width() * src.Height() * tmpl.BytesPerPixel()];

    if (allChanPresent) {
        // all channels are present, no colour conversion required ...

        unsigned int dstChanCnt = tmpl.GetChannelCount();
        unsigned int srcChanCnt = src.GetChannelCount();
        int *chanMap = new int[dstChanCnt];
        for (unsigned int i = 0; i < dstChanCnt; i++) {
            for (unsigned int j = 0; j < srcChanCnt; j++) {
                if (src.GetChannelLabel(j) == tmpl.GetChannelLabel(i)) {
                    chanMap[i] = j;
                    break;
                }
            }
        }

        switch (tmpl.GetChannelType()) {
            case CHANNELTYPE_BYTE: {
                unsigned char *dstBuf = reinterpret_cast<unsigned char*>(buf);
                switch (src.GetChannelType()) {
                    case CHANNELTYPE_BYTE: {
                        unsigned char *srcBuf
                            = reinterpret_cast<unsigned char*>(src.data);
                        this->copyBits(src.Width(), src.Height(), dstBuf,
                            srcBuf, srcChanCnt, chanMap, dstChanCnt);
                    } break;
                    case CHANNELTYPE_WORD: {
                        unsigned short *srcBuf
                            = reinterpret_cast<unsigned short*>(src.data);
                        this->copyBits(src.Width(), src.Height(), dstBuf,
                            srcBuf, srcChanCnt, chanMap, dstChanCnt);
                    } break;
                    case CHANNELTYPE_FLOAT: {
                        float *srcBuf = reinterpret_cast<float*>(src.data);
                        this->copyBits(src.Width(), src.Height(), dstBuf,
                            srcBuf, srcChanCnt, chanMap, dstChanCnt);
                    } break;
                }
            } break;
            case CHANNELTYPE_WORD: {
                unsigned short *dstBuf
                    = reinterpret_cast<unsigned short*>(buf);
                switch (src.GetChannelType()) {
                    case CHANNELTYPE_BYTE: {
                        unsigned char *srcBuf
                            = reinterpret_cast<unsigned char*>(src.data);
                        this->copyBits(src.Width(), src.Height(), dstBuf,
                            srcBuf, srcChanCnt, chanMap, dstChanCnt);
                    } break;
                    case CHANNELTYPE_WORD: {
                        unsigned short *srcBuf
                            = reinterpret_cast<unsigned short*>(src.data);
                        this->copyBits(src.Width(), src.Height(), dstBuf,
                            srcBuf, srcChanCnt, chanMap, dstChanCnt);
                    } break;
                    case CHANNELTYPE_FLOAT: {
                        float *srcBuf = reinterpret_cast<float*>(src.data);
                        this->copyBits(src.Width(), src.Height(), dstBuf,
                            srcBuf, srcChanCnt, chanMap, dstChanCnt);
                    } break;
                }
            } break;
            case CHANNELTYPE_FLOAT: {
                float *dstBuf = reinterpret_cast<float*>(buf);
                switch (src.GetChannelType()) {
                    case CHANNELTYPE_BYTE: {
                        unsigned char *srcBuf
                            = reinterpret_cast<unsigned char*>(src.data);
                        this->copyBits(src.Width(), src.Height(), dstBuf,
                            srcBuf, srcChanCnt, chanMap, dstChanCnt);
                    } break;
                    case CHANNELTYPE_WORD: {
                        unsigned short *srcBuf
                            = reinterpret_cast<unsigned short*>(src.data);
                        this->copyBits(src.Width(), src.Height(), dstBuf,
                            srcBuf, srcChanCnt, chanMap, dstChanCnt);
                    } break;
                    case CHANNELTYPE_FLOAT: {
                        float *srcBuf = reinterpret_cast<float*>(src.data);
                        this->copyBits(src.Width(), src.Height(), dstBuf,
                            srcBuf, srcChanCnt, chanMap, dstChanCnt);
                    } break;
                }
            } break;
        }

        delete[] chanMap;

    } else {
        // full conversion required :-( This is going to be slow!

        switch (tmpl.GetChannelType()) {
            case CHANNELTYPE_BYTE: {
                unsigned char *dstBuf = reinterpret_cast<unsigned char*>(buf);
                switch (src.GetChannelType()) {
                    case CHANNELTYPE_BYTE: {
                        unsigned char *srcBuf
                            = reinterpret_cast<unsigned char*>(src.data);
                        this->fullConvert(src.Width(), src.Height(), dstBuf,
                            tmpl.labels, tmpl.numChans, srcBuf, src.labels,
                            src.numChans);
                    } break;
                    case CHANNELTYPE_WORD: {
                        unsigned short *srcBuf
                            = reinterpret_cast<unsigned short*>(src.data);
                        this->fullConvert(src.Width(), src.Height(), dstBuf,
                            tmpl.labels, tmpl.numChans, srcBuf, src.labels,
                            src.numChans);
                    } break;
                    case CHANNELTYPE_FLOAT: {
                        float *srcBuf = reinterpret_cast<float*>(src.data);
                        this->fullConvert(src.Width(), src.Height(), dstBuf,
                            tmpl.labels, tmpl.numChans, srcBuf, src.labels,
                            src.numChans);
                    } break;
                }
            } break;
            case CHANNELTYPE_WORD: {
                unsigned short *dstBuf
                    = reinterpret_cast<unsigned short*>(buf);
                switch (src.GetChannelType()) {
                    case CHANNELTYPE_BYTE: {
                        unsigned char *srcBuf
                            = reinterpret_cast<unsigned char*>(src.data);
                        this->fullConvert(src.Width(), src.Height(), dstBuf,
                            tmpl.labels, tmpl.numChans, srcBuf, src.labels,
                            src.numChans);
                    } break;
                    case CHANNELTYPE_WORD: {
                        unsigned short *srcBuf
                            = reinterpret_cast<unsigned short*>(src.data);
                        this->fullConvert(src.Width(), src.Height(), dstBuf,
                            tmpl.labels, tmpl.numChans, srcBuf, src.labels,
                            src.numChans);
                    } break;
                    case CHANNELTYPE_FLOAT: {
                        float *srcBuf = reinterpret_cast<float*>(src.data);
                        this->fullConvert(src.Width(), src.Height(), dstBuf,
                            tmpl.labels, tmpl.numChans, srcBuf, src.labels,
                            src.numChans);
                    } break;
                }
            } break;
            case CHANNELTYPE_FLOAT: {
                float *dstBuf = reinterpret_cast<float*>(buf);
                switch (src.GetChannelType()) {
                    case CHANNELTYPE_BYTE: {
                        unsigned char *srcBuf
                            = reinterpret_cast<unsigned char*>(src.data);
                        this->fullConvert(src.Width(), src.Height(), dstBuf,
                            tmpl.labels, tmpl.numChans, srcBuf, src.labels,
                            src.numChans);
                    } break;
                    case CHANNELTYPE_WORD: {
                        unsigned short *srcBuf
                            = reinterpret_cast<unsigned short*>(src.data);
                        this->fullConvert(src.Width(), src.Height(), dstBuf,
                            tmpl.labels, tmpl.numChans, srcBuf, src.labels,
                            src.numChans);
                    } break;
                    case CHANNELTYPE_FLOAT: {
                        float *srcBuf = reinterpret_cast<float*>(src.data);
                        this->fullConvert(src.Width(), src.Height(), dstBuf,
                            tmpl.labels, tmpl.numChans, srcBuf, src.labels,
                            src.numChans);
                    } break;
                }
            } break;
        }

    }

    this->chanType = tmpl.chanType;
    delete[] this->data;
    this->data = buf;
    this->height = src.height;
    ChannelLabel *tmplLabels = tmpl.labels;
    ChannelLabel *oldLabels = this->labels;
    ChannelLabel *newLabels = new ChannelLabel[tmpl.numChans];
    ::memcpy(newLabels, tmplLabels, this->numChans * sizeof(ChannelLabel));
    delete[] oldLabels;
    this->labels = newLabels;
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
 * vislib::graphics::BitmapImage::EqualChannelLayout
 */
bool vislib::graphics::BitmapImage::EqualChannelLayout(
        const vislib::graphics::BitmapImage& tmpl) const {
    if ((this->chanType != tmpl.chanType) || (this->numChans != tmpl.numChans)) return false;
    for (unsigned int i = 0; i < this->numChans; i++) {
        if (this->labels[i] != tmpl.labels[i]) return false;
    }
    return true;
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
 * vislib::graphics::BitmapImage::HasChannel
 */
bool vislib::graphics::BitmapImage::HasChannel(ChannelLabel label) const {
    for (unsigned int c = 0; c < this->numChans; c++) {
        if (this->labels[c] == label) {
            return true;
        }
    }
    return false;
}


/*
 * vislib::graphics::BitmapImage::Invert
 */
void vislib::graphics::BitmapImage::Invert(void) {
    switch (this->chanType) {
        case CHANNELTYPE_BYTE:
            this->invert<unsigned char>(255u, UINT_MAX);
            break;
        case CHANNELTYPE_WORD:
            this->invert<unsigned short>(65535u, UINT_MAX);
            break;
        case CHANNELTYPE_FLOAT:
            this->invert<float>(1.0f, UINT_MAX);
            break;
    }
}


/*
 * vislib::graphics::BitmapImage::Invert
 */
void vislib::graphics::BitmapImage::Invert(unsigned int channel) {
    if (channel >= this->numChans) {
        channel = UINT_MAX;
    }
    switch (this->chanType) {
        case CHANNELTYPE_BYTE:
            this->invert<unsigned char>(255u, channel);
            break;
        case CHANNELTYPE_WORD:
            this->invert<unsigned short>(65535u, channel);
            break;
        case CHANNELTYPE_FLOAT:
            this->invert<float>(1.0f, channel);
            break;
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


/*
 * vislib::graphics::BitmapImage::copyBits
 */
template<class DT, class ST>
void vislib::graphics::BitmapImage::copyBits(unsigned int w, unsigned int h,
        DT *dst, ST *src, unsigned int srcChanCnt, int *chanMap,
        unsigned int chanCnt) {
    for (unsigned int y = 0; y < h; y++) {
        for (unsigned int x = 0; x < w; x++) {
            for (unsigned int c = 0; c < chanCnt; c++) {
                Conversion<ST>::CopyBit(dst[c], src[chanMap[c]]);
            }
            dst += chanCnt;
            src += srcChanCnt;
        }
    }
}


/*
 * vislib::graphics::BitmapImage::fullConvert
 */
template<class DT, class ST>
void vislib::graphics::BitmapImage::fullConvert(unsigned int w, unsigned int h,
            DT* dst, ChannelLabel *dstChan, unsigned int dstChanCnt,
            ST* src, ChannelLabel *srcChan, unsigned int srcChanCnt) {
    Conversion<ST> conv(src, srcChanCnt);
    for (unsigned int i = 0; i < srcChanCnt; i++) {
        conv.AddSourceChannel(i, srcChan[i]);
    }
    conv.FinalizeInitialization();
    typename Conversion<ST>::SourceChannel *dstSrcChan
        = new typename Conversion<ST>::SourceChannel[dstChanCnt];
    conv.ChannelMapping(dstSrcChan, dstChan, dstChanCnt);

    for (unsigned int y = 0; y < h; y++) {
        for (unsigned int x = 0; x < w; x++) {
            for (unsigned int c = 0; c < dstChanCnt; c++) {
                Conversion<float>::CopyBit(dst[c], conv.GetValue(dstSrcChan[c]));
            }
            dst += dstChanCnt;
            ++conv;
        }
    }

    delete[] dstSrcChan;

}


/*
 * vislib::graphics::BitmapImage::invert
 */
template<class T> void vislib::graphics::BitmapImage::invert(T maxval, unsigned int chan) {
    T *buf = reinterpret_cast<T*>(this->data);

    for (unsigned int y = 0; y < this->height; y++) {
        for (unsigned int x = 0; x < this->width; x++) {
            if (chan == UINT_MAX) {
                for (unsigned int c = 0; c < this->numChans; c++) {
                    buf[c] = maxval - buf[c];
                }
            } else {
                buf[chan] = maxval - buf[chan];
            }
            buf += this->numChans;
        }
    }
}
