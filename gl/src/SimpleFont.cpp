/*
 * SimpleFont.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/SimpleFont.h"
#include "vislib/types.h"
#include "glh/glh_genext.h"
#include <GL/glu.h>
#include <float.h>
#include <climits>
#include "glh/glh_extensions.h"
#include "vislib/StringConverter.h"
#include "vislib/mathfunctions.h"

#include "SimpleFontData.inc"


/*
 * vislib::graphics::gl::SimpleFont::SimpleFont
 */
vislib::graphics::gl::SimpleFont::SimpleFont(void)
        : vislib::graphics::AbstractFont(), texId(0) {
    // Intentionally empty
}


/*
 * vislib::graphics::gl::SimpleFont::~SimpleFont
 */
vislib::graphics::gl::SimpleFont::~SimpleFont(void) {
    this->Deinitialise();
}


/*
 * vislib::graphics::gl::SimpleFont::BlockLines
 */
unsigned int vislib::graphics::gl::SimpleFont::BlockLines(float maxWidth,
        float size, const char *txt) const {
    unsigned int lineCnt = 0;
    TextLine *lines = this->layoutText(txt,
        maxWidth / (size * 16.0f), // TODO: Not sure with this scaling!
        lineCnt);
    delete[] lines;
    return lineCnt;
}


/*
 * vislib::graphics::gl::SimpleFont::BlockLines
 */
unsigned int vislib::graphics::gl::SimpleFont::BlockLines(float maxWidth,
        float size, const wchar_t *txt) const {
    return this->BlockLines(maxWidth, size, W2A(txt));
}


/*
 * vislib::graphics::gl::SimpleFont::DrawString
 */
void vislib::graphics::gl::SimpleFont::DrawString(float x, float y,
        float size, bool flipY, const char *txt,
        vislib::graphics::AbstractFont::Alignment align) const {
    unsigned int lineCnt = 0;
    const float lh = (flipY ? -1.0f : 1.0f) * this->LineHeight(size);
    TextLine *lines = this->layoutText(txt, FLT_MAX, lineCnt);
    if ((lines == NULL) || (lineCnt == 0)) {
        // unexpected. Maybe 'txt' was NULL? Better fail silently.
        return;
    }

    switch (align) {
        case ALIGN_LEFT_MIDDLE:
        case ALIGN_CENTER_MIDDLE:
        case ALIGN_RIGHT_MIDDLE:
            y -= 0.5f * lh * lineCnt;
            break;
        case ALIGN_LEFT_BOTTOM:
        case ALIGN_CENTER_BOTTOM:
        case ALIGN_RIGHT_BOTTOM:
            y -= lh * lineCnt;
            break;
#ifndef _WIN32
        default:
            break;
#endif /* !_win32 */
    }

    this->drawText(x, y, lines, lineCnt, size, flipY, align);

    delete[] lines;
}


/*
 * vislib::graphics::gl::SimpleFont::DrawString
 */
void vislib::graphics::gl::SimpleFont::DrawString(float x, float y,
        float w, float h, float size, bool flipY, const char *txt,
        vislib::graphics::AbstractFont::Alignment align) const {
    unsigned int lineCnt = 0;
    TextLine *lines = this->layoutText(txt, w / (size * 16.0f), lineCnt);
    if ((lines == NULL) || (lineCnt == 0)) {
        // unexpected. Maybe 'txt' was NULL? Better fail silently.
        return;
    }
    switch (align) {
        case ALIGN_CENTER_TOP:
        case ALIGN_CENTER_MIDDLE:
        case ALIGN_CENTER_BOTTOM:
            x += w * 0.5f;
            break;
        case ALIGN_RIGHT_TOP:
        case ALIGN_RIGHT_MIDDLE:
        case ALIGN_RIGHT_BOTTOM:
            x += w;
            break;
#ifndef _WIN32
        default:
            break;
#endif /* !_win32 */
    }

    if (flipY) {
        y += h;
        switch (align) {
            case ALIGN_LEFT_MIDDLE:
            case ALIGN_CENTER_MIDDLE:
            case ALIGN_RIGHT_MIDDLE:
                y -= 0.5f * (h - this->LineHeight(size) * lineCnt);
                break;
            case ALIGN_LEFT_BOTTOM:
            case ALIGN_CENTER_BOTTOM:
            case ALIGN_RIGHT_BOTTOM:
                y -= h - this->LineHeight(size) * lineCnt;
                break;
#ifndef _WIN32
            default:
                break;
#endif /* !_win32 */
        }
    } else {
        switch (align) {
            case ALIGN_LEFT_MIDDLE:
            case ALIGN_CENTER_MIDDLE:
            case ALIGN_RIGHT_MIDDLE:
                y += 0.5f * (h - this->LineHeight(size) * lineCnt);
                break;
            case ALIGN_LEFT_BOTTOM:
            case ALIGN_CENTER_BOTTOM:
            case ALIGN_RIGHT_BOTTOM:
                y += h - this->LineHeight(size) * lineCnt;
                break;
#ifndef _WIN32
            default:
                break;
#endif /* !_win32 */
        }
    }
    this->drawText(x, y, lines, lineCnt, size, flipY, align);

    delete[] lines;
}


/*
 * vislib::graphics::gl::SimpleFont::DrawString
 */
void vislib::graphics::gl::SimpleFont::DrawString(float x, float y, float size,
        bool flipY, const wchar_t *txt,
        vislib::graphics::AbstractFont::Alignment align) const {
    this->DrawString(x, y, size, flipY, W2A(txt), align);
}


/*
 * vislib::graphics::gl::SimpleFont::DrawString
 */
void vislib::graphics::gl::SimpleFont::DrawString(float x, float y,
        float w, float h, float size, bool flipY, const wchar_t *txt,
        vislib::graphics::AbstractFont::Alignment align) const {
    this->DrawString(x, y, w, h, size, flipY, W2A(txt), align);
}


/*
 * vislib::graphics::gl::SimpleFont::LineWidth
 */
float vislib::graphics::gl::SimpleFont::LineWidth(float size, const char *txt)
        const {
    unsigned int lineCnt = 0;
    TextLine *lines = this->layoutText(txt, FLT_MAX, lineCnt);
    if ((lines == NULL) || (lineCnt == 0)) {
        // unexpected. Maybe 'txt' was NULL? Better fail silently.
        return 0.0f;
    }

    float width = 0.0f;
    for (unsigned int i = 0; i < lineCnt; i++) {
        width = vislib::math::Max(width, lines[i].width);
    }

    delete[] lines;

    return width * size * 16.0f;
}


/*
 * vislib::graphics::gl::SimpleFont::LineWidth
 */
float vislib::graphics::gl::SimpleFont::LineWidth(float size,
        const wchar_t *txt) const {
    return this->LineWidth(size, W2A(txt));
}


/*
 * vislib::graphics::gl::SimpleFont::initialise
 */
bool vislib::graphics::gl::SimpleFont::initialise(void) {
    if (this->texId != 0) {
        ::glDeleteTextures(1, &this->texId);
    }

    ::glGetError();
    bool texEnabled = (::glIsEnabled(GL_TEXTURE_2D) == GL_TRUE);
    if (!texEnabled) {
        ::glEnable(GL_TEXTURE_2D);
    }

    ::glGenTextures(1, &this->texId);
    if (::glGetError() != GL_NO_ERROR) return false;
    ::glBindTexture(GL_TEXTURE_2D, this->texId);
    if (::glGetError() != GL_NO_ERROR) return false;

    ::glPushAttrib(GL_PIXEL_MODE_BIT);

    ::glPixelTransferf(GL_RED_BIAS, 1.0f);
    ::glPixelTransferf(GL_GREEN_BIAS, 1.0f);
    ::glPixelTransferf(GL_BLUE_BIAS, 1.0f);

    ::glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE_ALPHA, 256, 256, 0,
        GL_ALPHA, GL_UNSIGNED_BYTE, bmpSimpleFontTexture);

    ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

    ::glPopAttrib();

    ::glBindTexture(GL_TEXTURE_2D, 0);
    if (!texEnabled) {
        ::glDisable(GL_TEXTURE_2D);
    }

    return (::glGetError() == GL_NO_ERROR);
}


/*
 * vislib::graphics::gl::SimpleFont::deinitialise
 */
void vislib::graphics::gl::SimpleFont::deinitialise(void) {
    if (this->texId != 0) {
        ::glDeleteTextures(1, &this->texId);
        this->texId = 0;
    }
}


/*
 * vislib::graphics::gl::SimpleFont::drawText
 */
void vislib::graphics::gl::SimpleFont::drawText(float x, float y,
        vislib::graphics::gl::SimpleFont::TextLine *lines,
        unsigned int lineCnt, float size, bool flipY,
        vislib::graphics::AbstractFont::Alignment halign) const {
    const float texOff = 1.0f / 512.0f;
    const float lh = (flipY ? -1.0f : 1.0f) * this->LineHeight(size);
    this->enterTextMode();

    float xp, yp = y;
    float xo = 0.0f;
    switch (halign) {
        case ALIGN_CENTER_TOP:
        case ALIGN_CENTER_MIDDLE:
        case ALIGN_CENTER_BOTTOM:
            xo = -0.5f;
            break;
        case ALIGN_RIGHT_TOP:
        case ALIGN_RIGHT_MIDDLE:
        case ALIGN_RIGHT_BOTTOM:
            xo = -1.0f;
            break;
#ifndef _WIN32
        default:
            break;
#endif /* !_win32 */
    }

    glBegin(GL_QUADS);
    for (TextLine *line = lines; lineCnt > 0; line++, lineCnt--) {
        xp = x + line->width * xo * 16.0f * size;
        unsigned int len = line->length;

        for (const unsigned char *c 
                = reinterpret_cast<const unsigned char*>(line->text);
                len > 0; c++, len--) {

            float gy = (float(int(*c) / 16) / 16.0f) + texOff;
            float gx = bmpSimpleFontGlyphSize[*c].x + texOff;
            float gw = bmpSimpleFontGlyphSize[*c].width;
            float w = gw * 16.0f * size;

            glTexCoord2f(gx, gy);
            glVertex2f(xp, yp);

            glTexCoord2f(gx + gw, gy);
            glVertex2f(xp + w, yp);

            glTexCoord2f(gx + gw, gy + (1.0f / 16.0f));
            glVertex2f(xp + w, yp + lh);

            glTexCoord2f(gx, gy + (1.0f / 16.0f));
            glVertex2f(xp, yp + lh);

            xp += w;
        }

        yp += lh;
    }
    glEnd();

    this->leaveTextMode();
}



/*
 * vislib::graphics::gl::SimpleFont::enterTextMode
 */
void vislib::graphics::gl::SimpleFont::enterTextMode(void) const {
    this->texEnabled = (::glIsEnabled(GL_TEXTURE_2D) == GL_TRUE);
    if (!this->texEnabled) {
        ::glEnable(GL_TEXTURE_2D);
    }
    this->blendEnabled = (::glIsEnabled(GL_BLEND) == GL_TRUE);
    if (!this->blendEnabled) {
        ::glEnable(GL_BLEND);
    }
    ::glGetIntegerv(GL_BLEND_SRC, &this->blendS);
    ::glGetIntegerv(GL_BLEND_DST, &this->blendD);
    ::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
#if defined(GL_TEXTURE_2D_BINDING)
    ::glGetIntegerv(GL_TEXTURE_2D_BINDING, &this->oldTex);
#elif defined(GL_TEXTURE_2D_BINDING_EXT)
    ::glGetIntegerv(GL_TEXTURE_2D_BINDING_EXT, &this->oldTex);
#endif
    ::glBindTexture(GL_TEXTURE_2D, this->texId);
    ::glGetTexEnviv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, &this->oldMode);
    ::glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    this->depthEnabled = (::glIsEnabled(GL_DEPTH_TEST) == GL_TRUE);
    if (this->depthEnabled) {
        ::glDisable(GL_DEPTH_TEST);
    }
}


/*
 * vislib::graphics::gl::SimpleFont::layoutText
 */
vislib::graphics::gl::SimpleFont::TextLine*
vislib::graphics::gl::SimpleFont::layoutText(const char *text, float maxWidth,
        unsigned int &outLineCnt) const {
    unsigned int len = vislib::CharTraitsA::SafeStringLength(text);
    TextLine *retval = new TextLine[len]; // heavily overestimated ... Boh
    outLineCnt = 0;
    const unsigned char *ls = NULL;

    retval[0].text = text;
    retval[0].length = 0;
    retval[0].width = 0.0f;

    for (const unsigned char *c 
            = reinterpret_cast<const unsigned char*>(text);
            len > 0; c++, len--) {
        float gw = bmpSimpleFontGlyphSize[*c].width;
        if (vislib::CharTraitsA::IsSpace(
                *reinterpret_cast<const char*>(c))) {
            ls = c;
        }
        if (*c == '\n') {
            outLineCnt++;
            retval[outLineCnt].text = reinterpret_cast<const char*>(c + 1);
            retval[outLineCnt].width = 0.0f;
            retval[outLineCnt].length = 0;
            ls = NULL;
        } else if (retval[outLineCnt].width + gw > maxWidth) {
            if (ls != NULL) {
                retval[outLineCnt].length = static_cast<unsigned int>(ls -
                    reinterpret_cast<const unsigned char*>(
                    retval[outLineCnt].text));
                retval[outLineCnt].width = 0.0f;
                for (unsigned int i = 0; i < retval[outLineCnt].length; i++) {
                    retval[outLineCnt].width += bmpSimpleFontGlyphSize[
                        reinterpret_cast<const unsigned char*>(
                            retval[outLineCnt].text)[i]].width;
                }
                len += static_cast<unsigned int>(c - (ls + 1));
                c = ls + 1;
                gw = bmpSimpleFontGlyphSize[*c].width;
            }
            outLineCnt++;
            retval[outLineCnt].text = reinterpret_cast<const char*>(c);
            retval[outLineCnt].width = gw;
            retval[outLineCnt].length = 1;
            ls = NULL;
        } else {
            retval[outLineCnt].width += gw;
            retval[outLineCnt].length++;
        }
    }

    if (retval[outLineCnt].length > 0) {
        outLineCnt++;
    }
    return retval;
}

/*
 * vislib::graphics::gl::SimpleFont::leaveTextMode
 */
void vislib::graphics::gl::SimpleFont::leaveTextMode(void) const {
    if (!this->texEnabled) {
        ::glDisable(GL_TEXTURE_2D);
    }
    if (!this->blendEnabled) {
        ::glDisable(GL_BLEND);
    }
    ::glBlendFunc(this->blendS, this->blendD);
#if defined(GL_TEXTURE_2D_BINDING) || defined(GL_TEXTURE_2D_BINDING_EXT)
    ::glBindTexture(GL_TEXTURE_2D, this->oldTex);
#endif
    ::glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, this->oldMode);
    if (this->depthEnabled) {
        ::glEnable(GL_DEPTH_TEST);
    }
}
