/*
 * testmisc.cpp
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "testBitmapImage.h"
#define _USE_MATH_DEFINES
#include "testhelper.h"

#include "vislib/BitmapImage.h"
#include "vislib/PpmBitmapCodec.h"
#include "vislib/BmpBitmapCodec.h"
#include "vislib/RawStorage.h"
#include "vislib/BitmapPainter.h"
#include "vislib/ColourRGBAu8.h"
#include "vislib/NamedColours.h"
#include <cmath>


/*
 * TestBitmapCodecSimple
 */
void TestBitmapCodecSimple(void) {
    using vislib::graphics::PpmBitmapCodec;
    using vislib::graphics::BitmapImage;
    using vislib::RawStorage;
    using vislib::graphics::BmpBitmapCodec;

    const unsigned char bmpdata1[] = {
          0,  0,  0,  85,  0,  0, 170,  0,  0, 255,  0,  0,
          0, 85,  0,  85, 85,  0, 170, 85,  0, 255, 85,  0,
          0,170,  0,  85,170,  0, 170,170,  0, 255,170,  0,
          0,255,  0,  85,255,  0, 170,255,  0, 255,255,  0
    };

    const unsigned char bmpdata2[] = {
          0,  0,  0,  85,  0,  0, 170,  0,  0, 255,  0,  0,
          0,  0, 85,  85,  0, 85, 170,  0, 85, 255,  0, 85,
          0,  0,170,  85,  0,170, 170,  0,170, 255,  0,170,
          0,  0,255,  85,  0,255, 170,  0,255, 255,  0,255
    };

    PpmBitmapCodec codec;
    BitmapImage img(4, 4, 3, BitmapImage::CHANNELTYPE_BYTE, NULL);
    RawStorage mem;
    img.SetChannelLabel(0, BitmapImage::CHANNEL_RED);
    img.SetChannelLabel(1, BitmapImage::CHANNEL_GREEN);
    img.SetChannelLabel(2, BitmapImage::CHANNEL_BLUE);
    memcpy(img.PeekData(), bmpdata1, 4 * 4 *3);
    codec.Image() = &img;

    codec.SetSaveOption(true);
    AssertTrue("Working with binary PPM", codec.GetSaveOption());
    AssertTrue("Codec can store to memory", codec.CanSave());
    AssertTrue("Codec can load from memory", codec.CanLoad());
    AssertTrue("Bitmap data stored in image", memcmp(img.PeekData(), bmpdata1, 4 * 4 * 3) == 0);

    AssertTrue("Image stored in memory", codec.Save(mem));
    AssertTrue("Memory not empty", mem.GetSize() > 0);
    AssertTrue("Image data not changed", memcmp(img.PeekData(), bmpdata1, 4 * 4 * 3) == 0);

    memcpy(img.PeekData(), bmpdata2, 4 * 4 *3);
    AssertTrue("Image data updated (1/2)", memcmp(img.PeekData(), bmpdata1, 4 * 4 * 3) != 0);
    AssertTrue("Image data updated (2/2)", memcmp(img.PeekData(), bmpdata2, 4 * 4 * 3) == 0);

    AssertTrue("Image loaded from memory", codec.Load(mem));
    AssertTrue("Memory not cleared", mem.GetSize() > 0);
    AssertTrue("Image data restored (1/2)", memcmp(img.PeekData(), bmpdata1, 4 * 4 * 3) == 0);
    AssertTrue("Image data restored (2/2)", memcmp(img.PeekData(), bmpdata2, 4 * 4 * 3) != 0);

    codec.SetSaveOption(false);
    AssertFalse("Working with ASCII PPM", codec.GetSaveOption());
    AssertTrue("Codec can store to memory", codec.CanSave());
    AssertTrue("Codec can load from memory", codec.CanLoad());
    AssertTrue("Bitmap data stored in image", memcmp(img.PeekData(), bmpdata1, 4 * 4 * 3) == 0);

    AssertTrue("Image stored in memory", codec.Save(mem));
    AssertTrue("Memory not empty", mem.GetSize() > 0);
    AssertTrue("Image data not changed", memcmp(img.PeekData(), bmpdata1, 4 * 4 * 3) == 0);

    memcpy(img.PeekData(), bmpdata2, 4 * 4 *3);
    AssertTrue("Image data updated (1/2)", memcmp(img.PeekData(), bmpdata1, 4 * 4 * 3) != 0);
    AssertTrue("Image data updated (2/2)", memcmp(img.PeekData(), bmpdata2, 4 * 4 * 3) == 0);

    AssertTrue("Image loaded from memory", codec.Load(mem));
    AssertTrue("Memory not cleared", mem.GetSize() > 0);
    AssertTrue("Image data restored (1/2)", memcmp(img.PeekData(), bmpdata1, 4 * 4 * 3) == 0);
    AssertTrue("Image data restored (2/2)", memcmp(img.PeekData(), bmpdata2, 4 * 4 * 3) != 0);

    /* vislib::sys::File f;
    if (f.Open("C:\\temp\\test.ppm", vislib::sys::File::WRITE_ONLY,
            vislib::sys::File::SHARE_READ, vislib::sys::File::CREATE_OVERWRITE)) {
        f.Write(mem, mem.GetSize());
        f.Close();
    } */

    //BitmapImage bmp;
    //BmpBitmapCodec bmpCodec;
    //PpmBitmapCodec ppmCodec;
    //bmpCodec.Image() = &bmp;
    //ppmCodec.Image() = &bmp;

    //AssertTrue("Test Image loaded", bmpCodec.Load("C:\\tmp\\test_1.bmp"));
    //AssertTrue("Image present", bmpCodec.Image() != NULL);
    //AssertEqual("Image width correct", bmpCodec.Image()->Width(), 600U);
    //AssertEqual("Image height correct", bmpCodec.Image()->Height(), 902U);
    //AssertTrue("Test Image saved", ppmCodec.Save("C:\\tmp\\test_1.ppm"));

    //AssertTrue("Test Image loaded", bmpCodec.Load("C:\\tmp\\test_4.bmp"));
    //AssertTrue("Image present", bmpCodec.Image() != NULL);
    //AssertEqual("Image width correct", bmpCodec.Image()->Width(), 500U);
    //AssertEqual("Image height correct", bmpCodec.Image()->Height(), 500U);
    //AssertTrue("Test Image saved", ppmCodec.Save("C:\\tmp\\test_4.ppm"));

    //AssertTrue("Test Image loaded", bmpCodec.Load("C:\\tmp\\test_8.bmp"));
    //AssertTrue("Image present", bmpCodec.Image() != NULL);
    //AssertEqual("Image width correct", bmpCodec.Image()->Width(), 1000U);
    //AssertEqual("Image height correct", bmpCodec.Image()->Height(), 667U);
    //AssertTrue("Test Image saved", ppmCodec.Save("C:\\tmp\\test_8.ppm"));

    //AssertTrue("Test Image loaded", bmpCodec.Load("C:\\tmp\\test_16.bmp"));
    //AssertTrue("Image present", bmpCodec.Image() != NULL);
    //AssertEqual("Image width correct", bmpCodec.Image()->Width(), 546U);
    //AssertEqual("Image height correct", bmpCodec.Image()->Height(), 640U);
    //AssertTrue("Test Image saved", ppmCodec.Save("C:\\tmp\\test_16.ppm"));
    //AssertTrue("Test Image saved again", bmpCodec.Save("C:\\tmp\\test_16_test.bmp"));

    //AssertTrue("Test Image loaded", bmpCodec.Load("C:\\tmp\\test_24.bmp"));
    //AssertTrue("Image present", bmpCodec.Image() != NULL);
    //AssertEqual("Image width correct", bmpCodec.Image()->Width(), 692U);
    //AssertEqual("Image height correct", bmpCodec.Image()->Height(), 900U);
    //AssertTrue("Test Image saved", ppmCodec.Save("C:\\tmp\\test_24.ppm"));
    //AssertTrue("Test Image saved again", bmpCodec.Save("C:\\tmp\\test_24_test.bmp"));

    //AssertTrue("Test Image loaded", bmpCodec.Load("C:\\tmp\\test_32.bmp"));
    //AssertTrue("Image present", bmpCodec.Image() != NULL);
    //AssertEqual("Image width correct", bmpCodec.Image()->Width(), 800U);
    //AssertEqual("Image height correct", bmpCodec.Image()->Height(), 800U);
    //AssertTrue("Test Image saved", ppmCodec.Save("C:\\tmp\\test_32.ppm"));

    //// RLE is not really supported
    ////AssertTrue("Test Image loaded", bmpCodec.Load("C:\\tmp\\test_4_rle.bmp"));
    ////AssertTrue("Image present", bmpCodec.Image() != NULL);
    ////AssertEqual("Image width correct", bmpCodec.Image()->Width(), 868U);
    ////AssertEqual("Image height correct", bmpCodec.Image()->Height(), 1100U);
    ////AssertTrue("Test Image saved", ppmCodec.Save("C:\\tmp\\test_4_rle.ppm"));

}


/*
 * TestBitmapImage
 */
void TestBitmapImage(void) {
    using vislib::graphics::BitmapImage;
    BitmapImage bi(256, 256, 3, BitmapImage::CHANNELTYPE_FLOAT);
    bi.SetChannelLabel(0, BitmapImage::CHANNEL_BLUE);
    bi.SetChannelLabel(1, BitmapImage::CHANNEL_RED);
    bi.SetChannelLabel(2, BitmapImage::CHANNEL_GREEN);

    for (unsigned int x = 0; x < 256; x++) {
        for (unsigned int y = 0; y < 256; y++) {
            float *px = bi.PeekDataAs<float>() + (x + y * 256) * 3;
            px[0] = 0.0f;
            px[1] = static_cast<float>(x) / 255.0f;
            px[2] = static_cast<float>(y) / 255.0f;
        }
    }

    bi.Convert(BitmapImage::TemplateByteRGB);

    bool imgCorrect = true;
    for (unsigned int x = 0; x < 256; x++) {
        for (unsigned int y = 0; y < 256; y++) {
            unsigned char *px = bi.PeekDataAs<unsigned char>() + (x + y * 256) * 3;
            if ((px[0] != x) || (px[1] != y) || (px[2] != 0)) {
                imgCorrect = false;
            }
        }
    }

    AssertTrue("Image correct after first conversion", imgCorrect);

    bi.Convert(BitmapImage::TemplateByteGray);

    imgCorrect = true;
    for (unsigned int x = 0; x < 256; x++) {
        for (unsigned int y = 0; y < 256; y++) {
            unsigned char *px = bi.PeekDataAs<unsigned char>() + (x + y * 256) * 1;
            if (px[0] != vislib::math::Max(x, y)) {
                imgCorrect = false;
            }
        }
    }

    AssertTrue("Image correct after second conversion", imgCorrect);

    BitmapImage bi2(1, 1, 4, BitmapImage::CHANNELTYPE_FLOAT);
    bi2.SetChannelLabel(0, BitmapImage::CHANNEL_CYAN);
    bi2.SetChannelLabel(1, BitmapImage::CHANNEL_MAGENTA);
    bi2.SetChannelLabel(2, BitmapImage::CHANNEL_YELLOW);
    bi2.SetChannelLabel(3, BitmapImage::CHANNEL_BLACK);
    bi2.PeekDataAs<float>()[0] = 0.0f;
    bi2.PeekDataAs<float>()[1] = 0.0f;
    bi2.PeekDataAs<float>()[2] = 0.0f;
    bi2.PeekDataAs<float>()[3] = 0.0f;

    bi.ConvertFrom(bi2, BitmapImage::TemplateByteRGB);

    vislib::graphics::ColourRGBAu8 col(
        bi.PeekDataAs<unsigned char>()[0],
        bi.PeekDataAs<unsigned char>()[1],
        bi.PeekDataAs<unsigned char>()[2], 255);

    AssertEqual("CMYKConversion is correct", col, vislib::graphics::NamedColours::White);

    bi2.PeekDataAs<float>()[0] = 1.0f;
    bi.ConvertFrom(bi2, BitmapImage::TemplateByteRGB);
    col.Set(
        bi.PeekDataAs<unsigned char>()[0],
        bi.PeekDataAs<unsigned char>()[1],
        bi.PeekDataAs<unsigned char>()[2], 255);
    AssertEqual("CMYKConversion is correct", col, vislib::graphics::NamedColours::Cyan);

    bi2.PeekDataAs<float>()[3] = 1.0f;
    bi.ConvertFrom(bi2, BitmapImage::TemplateByteRGB);
    col.Set(
        bi.PeekDataAs<unsigned char>()[0],
        bi.PeekDataAs<unsigned char>()[1],
        bi.PeekDataAs<unsigned char>()[2], 255);
    AssertEqual("CMYKConversion is correct", col, vislib::graphics::NamedColours::Black);

}


/*
 * TestBitmapPainter
 */
void TestBitmapPainter(void) {
    using vislib::graphics::BitmapImage;
    using vislib::graphics::BitmapPainter;
    using vislib::graphics::BmpBitmapCodec;
    typedef vislib::math::Point<int, 2> Point;

    BitmapImage img(200, 200, BitmapImage::TemplateByteRGB);
    BitmapPainter draw(&img);

    draw.SetColour<BYTE, BYTE, BYTE>(127, 92, 64);
    draw.Clear();

    vislib::Array<Point> polygon;
    double a = 0.0;
    for (int stp = 0; stp < 5; stp++, a += ((M_PI * 6.0) / 5.0)) {
        double c = cos(a);
        double s = sin(a);
        polygon.Add(Point(
            static_cast<int>(99.5 + s * 40.0),
            static_cast<int>(99.5 - c * 40.0)));
    }
    draw.SetColour<BYTE, BYTE, BYTE>(200, 0, 0);
    draw.FillPolygon(polygon);

    draw.SetColour<BYTE, BYTE, BYTE>(0, 127, 255);
    draw.SetPixel(99, 100);
    draw.SetPixel(100, 100);
    draw.SetPixel(101, 100);
    draw.SetPixel(100, 99);
    draw.SetPixel(100, 101);

    draw.SetColour<BYTE, BYTE, BYTE>(0, 200, 0);
    a = 0.0;
    for (int stp = 0; stp < 40; stp++, a += 2.0 * M_PI / double(40)) {
        double c = cos(a);
        double s = sin(a);

        draw.DrawLine(
            99.5 + s * 30.0,
            99.5 + c * 30.0,
            99.5 + s * 98.0,
            99.5 + c * 98.0);
    }

    //BmpBitmapCodec codec;
    //codec.Image() = &img;
    //codec.Save("C:\\tmp\\paintertest.bmp");

}
