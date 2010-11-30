/*
 * ImageViewer.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
//#define _USE_MATH_DEFINES
#include "ImageViewer.h"
#include "param/FilePathParam.h"
#include "param/StringParam.h"
#include "view/CallRender3D.h"
#include "vislib/Log.h"
#include <GL/gl.h>
//#include <cmath>

using namespace megamol::core;

/*****************************************************************************/
#include "png.h"

using namespace sg::graphics;
using namespace vislib::graphics;


namespace sg {
namespace graphics {
namespace pngutil {

    /**
     * Utility buffer to read a png from memory
     */
    typedef struct __png_readbuffer_t {
        const unsigned char *buffer;
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
        pngReadBuffer *rd = static_cast<pngReadBuffer*>(png_get_io_ptr(png_ptr));
        SIZE_T remaining = rd->bufferSize - rd->bufferPos;
        if (length > remaining) {
            png_error(png_ptr, "unexpected end-of-data");
        }
        ::memcpy(data, rd->buffer + rd->bufferPos, length);
        rd->bufferPos += length;
    }


} /* end namespace pngutil */
} /* end namespace graphics */
} /* end namespace sg */


/*
 * PngBitmapCodec::PngBitmapCodec
 */
PngBitmapCodec::PngBitmapCodec(void) : AbstractBitmapCodec() {
    // intentionally empty
}


/*
 * PngBitmapCodec::~PngBitmapCodec
 */
PngBitmapCodec::~PngBitmapCodec(void) {
    // intentionally empty
}


/*
 * PngBitmapCodec::AutoDetect
 */
int PngBitmapCodec::AutoDetect(const void *mem, SIZE_T size) const {
    const unsigned char *dat = static_cast<const unsigned char*>(mem);
    if (size < 8) return -1; // insufficient data

    if ((dat[0] == 137) && (dat[1] == 80) && (dat[2] == 78) && (dat[3] == 71)
            && (dat[4] == 13) && (dat[5] == 10) && (dat[6] == 26) && (dat[7] == 10)) {
        return 1; // Png file signature
    }

    return 0;
}


/*
 * PngBitmapCodec::CanAutoDetect
 */
bool PngBitmapCodec::CanAutoDetect(void) const {
    return true;
}


/*
 * PngBitmapCodec::CanLoadFromMemory
 */
bool PngBitmapCodec::CanLoadFromMemory(void) const {
    return true;
}


/*
 * PngBitmapCodec::CanSaveToMemory
 */
bool PngBitmapCodec::CanSaveToMemory(void) const {
    return false; // ATM
}


/*
 * PngBitmapCodec::FileNameExtsA
 */
const char* PngBitmapCodec::FileNameExtsA(void) const {
    return ".png";
}


/*
 * PngBitmapCodec::FileNameExtsW
 */
const wchar_t* PngBitmapCodec::FileNameExtsW(void) const {
    return L".png";
}


/*
 * PngBitmapCodec::Load
 */
bool PngBitmapCodec::Load(const void *mem, SIZE_T size) {

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        printf("Argl: PNG create failed\n");
        return false;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        printf("Argl: PNG create info failed\n");
        return false;
    }

    png_infop end_info = png_create_info_struct(png_ptr);
    if (!end_info) {
        printf("Argl: PNG create end info failed\n");
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return false;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
        printf("Argl: PNG set jump failed\n");
        return false;
    }

    pngutil::pngReadBuffer rd;
    rd.buffer = static_cast<const unsigned char*>(mem);
    rd.bufferSize = size;
    rd.bufferPos = 0;

    png_set_read_fn(png_ptr, static_cast<void*>(&rd), &pngutil::pngReadData);

    png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_STRIP_16 | PNG_TRANSFORM_PACKING, NULL);

    unsigned int bit_depth = png_get_bit_depth(png_ptr, info_ptr);
    if (bit_depth != 8) {
        printf("Argl: PNG bit_depth != 8\n");
        return false; // currently only 8-bit is supported (input transfors should fix other depths)
    }

    unsigned int width = png_get_image_width(png_ptr, info_ptr);
    unsigned int height = png_get_image_height(png_ptr, info_ptr);
    unsigned int color_type = png_get_color_type(png_ptr, info_ptr);
    if ((color_type & ~6) != 0) {
        printf("Argl: Illegal colour type\n");
        return false; // illegal colour type (palette or something even stranger)
    }
    unsigned int chanCnt = (((color_type & 2) == 2) ? 3 : 1) + ((color_type & 4) ? 1 : 0);

    this->image().CreateImage(width, height, chanCnt, BitmapImage::CHANNELTYPE_BYTE);
    if ((color_type & 2) == 2) {
        this->image().SetChannelLabel(0, BitmapImage::CHANNEL_RED);
        this->image().SetChannelLabel(1, BitmapImage::CHANNEL_GREEN);
        this->image().SetChannelLabel(2, BitmapImage::CHANNEL_BLUE);
    } else {
        this->image().SetChannelLabel(0, BitmapImage::CHANNEL_GRAY);
    }
    if ((color_type & 4) == 4) {
        this->image().SetChannelLabel(chanCnt - 1,
            BitmapImage::CHANNEL_ALPHA);
    }

    png_bytepp row_pointers = png_get_rows(png_ptr, info_ptr);
    unsigned char *imgptr = this->image().PeekDataAs<unsigned char>();
    unsigned int lineSize = (width * chanCnt * 1);
    for (unsigned int y = 0; y < height; y++) {
        ::memcpy(imgptr + y * lineSize, row_pointers[y], lineSize);
    }

    png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);

    return true;
}


/*
 * PngBitmapCodec::NameA
 */
const char * PngBitmapCodec::NameA(void) const {
    return "Portable Network Graphics";
}


/*
 * PngBitmapCodec::NameW
 */
const wchar_t * PngBitmapCodec::NameW(void) const {
    return L"Portable Network Graphics";
}


/*
 * PngBitmapCodec::Save
 */
bool PngBitmapCodec::Save(vislib::RawStorage& outmem) const {
    // not implemented ATM
    return false;
}
/*****************************************************************************/


/*
 * misc::ImageViewer::ImageViewer
 */
misc::ImageViewer::ImageViewer(void) : Renderer3DModule(),
        leftFilenameSlot("leftImg", "The image file name"),
        rightFilenameSlot("rightImg", "The image file name"),
        pasteFilenamesSlot("pasteFiles", "Slot to paste both file names at once (semicolon-separated)"),
        width(1), height(1), tiles() {

    this->leftFilenameSlot << new param::FilePathParam("");
    this->MakeSlotAvailable(&this->leftFilenameSlot);

    this->rightFilenameSlot << new param::FilePathParam("");
    this->MakeSlotAvailable(&this->rightFilenameSlot);

    this->pasteFilenamesSlot << new param::StringParam("");
    this->pasteFilenamesSlot.SetUpdateCallback(&ImageViewer::onFilesPasted);
    this->MakeSlotAvailable(&this->pasteFilenamesSlot);
}


/*
 * misc::ImageViewer::~ImageViewer
 */
misc::ImageViewer::~ImageViewer(void) {
    this->Release();
}


/*
 * misc::ImageViewer::create
 */
bool misc::ImageViewer::create(void) {
    // intentionally empty
    return true;
}


/*
 * misc::ImageViewer::GetCapabilities
 */
bool misc::ImageViewer::GetCapabilities(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    cr->SetCapabilities(
        view::CallRender3D::CAP_RENDER
        );

    return true;
}


/*
 * misc::ImageViewer::GetExtents
 */
bool misc::ImageViewer::GetExtents(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    cr->SetTimeFramesCount(1);
    cr->AccessBoundingBoxes().Clear();
    cr->AccessBoundingBoxes().SetObjectSpaceBBox(0.0f, 0.0f, -0.5f,
        static_cast<float>(this->width), static_cast<float>(this->height), 0.5f);
    cr->AccessBoundingBoxes().SetObjectSpaceClipBox(cr->AccessBoundingBoxes().ObjectSpaceBBox());
    cr->AccessBoundingBoxes().MakeScaledWorld(1.0f);

    return true;
}


/*
 * misc::ImageViewer::release
 */
void misc::ImageViewer::release(void) {
//    this->image.Release();
}


/*
 * misc::ImageViewer::Render
 */
bool misc::ImageViewer::Render(Call& call) {
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D*>(&call);
    if (cr3d == NULL) return false;
    bool rightEye = (cr3d->GetCameraParameters()->Eye() == vislib::graphics::CameraParameters::RIGHT_EYE);
    param::ParamSlot *filenameSlot = rightEye ? (&this->rightFilenameSlot) : (&this->leftFilenameSlot);
    ::glEnable(GL_TEXTURE_2D);
    if (filenameSlot->IsDirty()) {
        filenameSlot->ResetDirty();
        const vislib::TString& filename = filenameSlot->Param<param::FilePathParam>()->Value();
        static vislib::graphics::BitmapImage img;
        static PngBitmapCodec codec;
        static const unsigned int TILE_SIZE = 2 * 1024;
        codec.Image() = &img;
        ::glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        try {
            if (codec.Load(filename)) {
                img.Convert(vislib::graphics::BitmapImage::TemplateByteRGB);
                this->width = img.Width();
                this->height = img.Height();
                this->tiles.Clear();
                BYTE *buf = new BYTE[TILE_SIZE * TILE_SIZE * 3];
                for (unsigned int y = 0; y < this->height; y += TILE_SIZE) {
                    unsigned int h = vislib::math::Min(TILE_SIZE, this->height - y);
                    for (unsigned int x = 0; x < this->width; x += TILE_SIZE) {
                        unsigned int w = vislib::math::Min(TILE_SIZE, this->width - x);
                        for (unsigned int l = 0; l < h; l++) {
                            ::memcpy(buf + (l * w * 3), img.PeekDataAs<BYTE>() + ((y + l) * this->width * 3 + x * 3), w * 3);
                        }
                        this->tiles.Add(vislib::Pair<vislib::math::Rectangle<float>, vislib::SmartPtr<vislib::graphics::gl::OpenGLTexture2D> >());
                        this->tiles.Last().First().Set(static_cast<float>(x), static_cast<float>(this->height - y), static_cast<float>(x + w), static_cast<float>(this->height - (y + h)));
                        this->tiles.Last().SetSecond(new vislib::graphics::gl::OpenGLTexture2D());
                        if (this->tiles.Last().Second()->Create(w, h, false, buf, GL_RGB) != GL_NO_ERROR) {
                            this->tiles.RemoveLast();
                        } else {
                            this->tiles.Last().Second()->SetFilter(GL_LINEAR, GL_LINEAR);
                            this->tiles.Last().Second()->SetWrap(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
                        }
                    }
                }
                delete[] buf;
                img.CreateImage(1, 1, vislib::graphics::BitmapImage::TemplateByteRGB);
            } else {
                printf("Failed: Load\n");
            }
        } catch(vislib::Exception ex) {
            printf("Failed: %s (%s;%d)\n", ex.GetMsgA(), ex.GetFile(), ex.GetLine());
        } catch(...) {
            printf("Failed\n");
        }
    }

    ::glDisable(GL_LINE_SMOOTH);
    ::glDisable(GL_BLEND);
    ::glDisable(GL_LIGHTING);
    ::glEnable(GL_DEPTH_TEST);
    ::glLineWidth(1.0f);
    ::glDisable(GL_LINE_SMOOTH);

    ::glColor3ub(255, 255, 255);
    for (SIZE_T i = 0; i < this->tiles.Count(); i++) {
        this->tiles[i].Second()->Bind();
        ::glBegin(GL_QUADS);
        ::glTexCoord2i(0, 0); ::glVertex2f(this->tiles[i].First().Left(), this->tiles[i].First().Bottom());
        ::glTexCoord2i(0, 1); ::glVertex2f(this->tiles[i].First().Left(), this->tiles[i].First().Top());
        ::glTexCoord2i(1, 1); ::glVertex2f(this->tiles[i].First().Right(), this->tiles[i].First().Top());
        ::glTexCoord2i(1, 0); ::glVertex2f(this->tiles[i].First().Right(), this->tiles[i].First().Bottom());
        ::glEnd();
    }
    ::glBindTexture(GL_TEXTURE_2D, 0);

    ::glDisable(GL_TEXTURE);

    return true;
}


/*
 * misc::ImageViewer::onFilesPasted
 */
bool misc::ImageViewer::onFilesPasted(param::ParamSlot &slot) {
    vislib::TString str(this->pasteFilenamesSlot.Param<param::StringParam>()->Value());
    vislib::TString::Size scp = str.Find(_T(";"));
    if (scp != vislib::TString::INVALID_POS) {
        this->leftFilenameSlot.Param<param::FilePathParam>()->SetValue(str.Substring(0, scp));
        this->rightFilenameSlot.Param<param::FilePathParam>()->SetValue(str.Substring(scp + 1));
    }
    return true;
}
