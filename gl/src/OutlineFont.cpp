/*
 * OutlineFont.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/OutlineFont.h"
#include <cfloat>
#include "glh/glh_genext.h"
#include "vislib/CharTraits.h"
#include "vislib/memutils.h"
#include "vislib/UTF8Encoder.h"

using namespace vislib::graphics::gl;


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFontInfo& ofi) : AbstractFont(),
        data(ofi), renderType(OutlineFont::RENDERTYPE_FILL) {
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFontInfo& ofi,
        OutlineFont::RenderType render) : AbstractFont(), data(ofi),
        renderType(render) {
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFontInfo& ofi, float size)
        : AbstractFont(), data(ofi), renderType(OutlineFont::RENDERTYPE_FILL) {
    this->SetSize(size);
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFontInfo& ofi, bool flipY)
        : AbstractFont(), data(ofi), renderType(OutlineFont::RENDERTYPE_FILL) {
    this->SetFlipY(flipY);
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFontInfo& ofi,
        OutlineFont::RenderType render, bool flipY) : AbstractFont(),
        data(ofi), renderType(render) {
    this->SetFlipY(flipY);
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFontInfo& ofi, float size, bool flipY)
        : AbstractFont(), data(ofi), renderType(OutlineFont::RENDERTYPE_FILL) {
    this->SetSize(size);
    this->SetFlipY(flipY);
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFontInfo& ofi, float size,
        OutlineFont::RenderType render) : AbstractFont(), data(ofi),
        renderType(render) {
    this->SetSize(size);
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFontInfo& ofi, float size,
        OutlineFont::RenderType render, bool flipY) : AbstractFont(),
        data(ofi), renderType(render) {
    this->SetSize(size);
    this->SetFlipY(flipY);
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFont& src) : AbstractFont(),
        data(src.data), renderType(src.renderType) {
    this->SetSize(src.GetSize());
    this->SetFlipY(src.IsFlipY());
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFont& src,
        OutlineFont::RenderType render) : AbstractFont(), data(src.data),
        renderType(render) {
    this->SetSize(src.GetSize());
    this->SetFlipY(src.IsFlipY());
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFont& src, float size) : AbstractFont(),
        data(src.data), renderType(src.renderType) {
    this->SetSize(size);
    this->SetFlipY(src.IsFlipY());
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFont& src, bool flipY) : AbstractFont(),
        data(src.data), renderType(src.renderType) {
    this->SetSize(src.GetSize());
    this->SetFlipY(flipY);
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFont& src,
        OutlineFont::RenderType render, bool flipY) : AbstractFont(),
        data(src.data), renderType(render) {
    this->SetSize(src.GetSize());
    this->SetFlipY(flipY);
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFont& src, float size, bool flipY)
        : AbstractFont(), data(src.data), renderType(src.renderType) {
    this->SetSize(size);
    this->SetFlipY(flipY);
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFont& src, float size,
        OutlineFont::RenderType render) : AbstractFont(), data(src.data),
        renderType(render) {
    this->SetSize(size);
    this->SetFlipY(src.IsFlipY());
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFont& src, float size,
        OutlineFont::RenderType render, bool flipY) : AbstractFont(),
        data(src.data), renderType(render) {
    this->SetSize(size);
    this->SetFlipY(flipY);
}


/*
 * OutlineFont::~OutlineFont
 */
OutlineFont::~OutlineFont(void) {
    this->Deinitialise();
}


/*
 * OutlineFont::BlockLines
 */
unsigned int OutlineFont::BlockLines(float maxWidth, float size,
        const char *txt) const {
    return this->lineCount(this->buildGlyphRun(txt, maxWidth / size), true);
}


/*
 * OutlineFont::BlockLines
 */
unsigned int OutlineFont::BlockLines(float maxWidth, float size,
        const wchar_t *txt) const {
    return this->lineCount(this->buildGlyphRun(txt, maxWidth / size), true);
}


/*
 * OutlineFont::DrawString
 */
void OutlineFont::DrawString(float x, float y, float size, bool flipY,
        const char *txt, AbstractFont::Alignment align) const {
    int *run = this->buildGlyphRun(txt, FLT_MAX);

    if ((align == ALIGN_CENTER_MIDDLE) || (align == ALIGN_LEFT_MIDDLE)
            || (align == ALIGN_RIGHT_MIDDLE)) {
        y += static_cast<float>(this->lineCount(run, false)) * 0.5f * size
            * (flipY ? 1.0f : -1.0f);

    } else if ((align == ALIGN_CENTER_BOTTOM) || (align == ALIGN_LEFT_BOTTOM)
            || (align == ALIGN_RIGHT_BOTTOM)) {
        y += static_cast<float>(this->lineCount(run, false)) * size
            * (flipY ? 1.0f : -1.0f);

    }

    if ((this->renderType == RENDERTYPE_FILL)
            || (this->renderType == RENDERTYPE_FILL_AND_OUTLINE)) {
        this->drawFilled(run, x, y, size, flipY, align);
    }
    if ((this->renderType == RENDERTYPE_OUTLINE)
            || (this->renderType == RENDERTYPE_FILL_AND_OUTLINE)) {
        this->drawOutline(run, x, y, size, flipY, align);
    }

    delete[] run;
}


/*
 * OutlineFont::DrawString
 */
void OutlineFont::DrawString(float x, float y, float w, float h, float size,
        bool flipY, const char *txt, AbstractFont::Alignment align) const {
    int *run = this->buildGlyphRun(txt, w / size);

    if (flipY) y += h;

    switch (align) {
        case ALIGN_CENTER_BOTTOM:
            x += w * 0.5f;
            y += (flipY ? -1.0f : 1.0f)
                * (h - this->lineCount(run, false) * size);
            break;
        case ALIGN_CENTER_MIDDLE:
            x += w * 0.5f;
            y += (flipY ? -1.0f : 1.0f)
                * (h - this->lineCount(run, false) * size) * 0.5f;
            break;
        case ALIGN_CENTER_TOP:
            x += w * 0.5f;
            break;
        case ALIGN_LEFT_BOTTOM:
            y += (flipY ? -1.0f : 1.0f)
                * (h - this->lineCount(run, false) * size);
            break;
        case ALIGN_LEFT_MIDDLE:
            y += (flipY ? -1.0f : 1.0f)
                * (h - this->lineCount(run, false) * size) * 0.5f;
            break;
        case ALIGN_RIGHT_BOTTOM:
            x += w;
            y += (flipY ? -1.0f : 1.0f)
                * (h - this->lineCount(run, false) * size);
            break;
        case ALIGN_RIGHT_MIDDLE:
            x += w;
            y += (flipY ? -1.0f : 1.0f)
                * (h - this->lineCount(run, false) * size) * 0.5f;
            break;
        case ALIGN_RIGHT_TOP:
            x += w;
            break;
#ifndef _WIN32
        default:
            break;
#endif /* !_WIN32 */
    }

    if ((this->renderType == RENDERTYPE_FILL)
            || (this->renderType == RENDERTYPE_FILL_AND_OUTLINE)) {
        this->drawFilled(run, x, y, size, flipY, align);
    }
    if ((this->renderType == RENDERTYPE_OUTLINE)
            || (this->renderType == RENDERTYPE_FILL_AND_OUTLINE)) {
        this->drawOutline(run, x, y, size, flipY, align);
    }

    delete[] run;
}


/*
 * OutlineFont::DrawString
 */
void OutlineFont::DrawString(float x, float y, float size, bool flipY,
        const wchar_t *txt, AbstractFont::Alignment align) const {
    int *run = this->buildGlyphRun(txt, FLT_MAX);

    if ((align == ALIGN_CENTER_MIDDLE) || (align == ALIGN_LEFT_MIDDLE)
            || (align == ALIGN_RIGHT_MIDDLE)) {
        y += static_cast<float>(this->lineCount(run, false)) * 0.5f * size
            * (flipY ? 1.0f : -1.0f);

    } else if ((align == ALIGN_CENTER_BOTTOM) || (align == ALIGN_LEFT_BOTTOM)
            || (align == ALIGN_RIGHT_BOTTOM)) {
        y += static_cast<float>(this->lineCount(run, false)) * size
            * (flipY ? 1.0f : -1.0f);

    }

    if ((this->renderType == RENDERTYPE_FILL)
            || (this->renderType == RENDERTYPE_FILL_AND_OUTLINE)) {
        this->drawFilled(run, x, y, size, flipY, align);
    }
    if ((this->renderType == RENDERTYPE_OUTLINE)
            || (this->renderType == RENDERTYPE_FILL_AND_OUTLINE)) {
        this->drawOutline(run, x, y, size, flipY, align);
    }

    delete[] run;
}


/*
 * OutlineFont::DrawString
 */
void OutlineFont::DrawString(float x, float y, float w, float h, float size,
        bool flipY, const wchar_t *txt, AbstractFont::Alignment align) const {
    int *run = this->buildGlyphRun(txt, w / size);

    if (flipY) y += h;

    switch (align) {
        case ALIGN_CENTER_BOTTOM:
            x += w * 0.5f;
            y += (flipY ? -1.0f : 1.0f)
                * (h - this->lineCount(run, false) * size);
            break;
        case ALIGN_CENTER_MIDDLE:
            x += w * 0.5f;
            y += (flipY ? -1.0f : 1.0f)
                * (h - this->lineCount(run, false) * size) * 0.5f;
            break;
        case ALIGN_CENTER_TOP:
            x += w * 0.5f;
            break;
        case ALIGN_LEFT_BOTTOM:
            y += (flipY ? -1.0f : 1.0f)
                * (h - this->lineCount(run, false) * size);
            break;
        case ALIGN_LEFT_MIDDLE:
            y += (flipY ? -1.0f : 1.0f)
                * (h - this->lineCount(run, false) * size) * 0.5f;
            break;
        case ALIGN_RIGHT_BOTTOM:
            x += w;
            y += (flipY ? -1.0f : 1.0f)
                * (h - this->lineCount(run, false) * size);
            break;
        case ALIGN_RIGHT_MIDDLE:
            x += w;
            y += (flipY ? -1.0f : 1.0f)
                * (h - this->lineCount(run, false) * size) * 0.5f;
            break;
        case ALIGN_RIGHT_TOP:
            x += w;
            break;
#ifndef _WIN32
        default:
            break;
#endif /* !_WIN32 */
    }

    if ((this->renderType == RENDERTYPE_FILL)
            || (this->renderType == RENDERTYPE_FILL_AND_OUTLINE)) {
        this->drawFilled(run, x, y, size, flipY, align);
    }
    if ((this->renderType == RENDERTYPE_OUTLINE)
            || (this->renderType == RENDERTYPE_FILL_AND_OUTLINE)) {
        this->drawOutline(run, x, y, size, flipY, align);
    }

    delete[] run;
}


/*
 * OutlineFont::LineWidth
 */
float OutlineFont::LineWidth(float size, const char *txt) const {
    int *run = this->buildGlyphRun(txt, FLT_MAX);
    int *i = run;
    float len = 0.0f;
    float comlen = 0.0f;
    while (*i != 0) {
        comlen = this->lineWidth(i, true);
        if (comlen > len) len = comlen;
    }
    delete[] run;
    return len * size;
}


/*
 * OutlineFont::LineWidth
 */
float OutlineFont::LineWidth(float size, const wchar_t *txt) const {
    int *run = this->buildGlyphRun(txt, FLT_MAX);
    int *i = run;
    float len = 0.0f;
    float comlen = 0.0f;
    while (*i != 0) {
        comlen = this->lineWidth(i, true);
        if (comlen > len) len = comlen;
    }
    delete[] run;
    return len * size;
}


/*
 * OutlineFont::initialise
 */
bool OutlineFont::initialise(void) {
    // intentionally empty
    return true;
}


/*
 * OutlineFont::deinitialise
 */
void OutlineFont::deinitialise(void) {
    // intentionally empty
}


/*
 * OutlineFont::buildGlyphRun
 */
int *OutlineFont::buildGlyphRun(const char *txt, float maxWidth) const {
    vislib::StringA txtutf8;
    if (!vislib::UTF8Encoder::Encode(txtutf8, txt)) {
        // encoding failed ... how?
        char *t = txtutf8.AllocateBuffer(
            vislib::CharTraitsA::SafeStringLength(txt));
        for (; *txt != 0; txt++) {
            if ((*txt & 0x80) == 0) {
                *t = *txt;
                t++;
            }
        }
        *t = 0;
    }
    return this->buildUpGlyphRun(txtutf8, maxWidth);
}


/*
 * OutlineFont::buildGlyphRun
 */
int *OutlineFont::buildGlyphRun(const wchar_t *txt, float maxWidth) const {
    vislib::StringA txtutf8;
    if (!vislib::UTF8Encoder::Encode(txtutf8, txt)) {
        // encoding failed ... how?
        char *t = txtutf8.AllocateBuffer(
            vislib::CharTraitsW::SafeStringLength(txt));
        for (; *txt != 0; txt++) {
            if ((*txt & 0x80) == 0) {
                *t = static_cast<char>(*txt);
                t++;
            }
        }
        *t = 0;
    }
    return this->buildUpGlyphRun(txtutf8, maxWidth);
}


/*
 * OutlineFont::buildUpGlyphRun
 */
int *OutlineFont::buildUpGlyphRun(const char *txtutf8, float maxWidth) const {
    SIZE_T txtlen = static_cast<SIZE_T>(
        CharTraitsA::SafeStringLength(txtutf8));
    SIZE_T pos = 0;
    int *glyphrun = new int[txtlen + 1];
    bool knowLastWhite = false;
    bool blackspace = true;
    SIZE_T lastWhiteGlyph = 0;
    SIZE_T lastWhiteSpace = 0;
    float lineLength = 0.0f;
    bool nextAsNewLine = false;
    signed short idx;
    ::memset(glyphrun, 0, sizeof(int) * (txtlen + 1));
    // > 0 1+index of the glyph to use
    // < 0 -(1+index) of the glyph and new line
    // = 0 end

    // build glyph run
    idx = 0;
    for (SIZE_T i = 0; i < txtlen; i++) {
        if (txtutf8[i] == '\n') { // special handle new lines
            nextAsNewLine = true;
            continue;
        }

        // select glyph
        idx = this->data.glyphIndex[idx * 16 + ((unsigned char)txtutf8[i] % 0x10)];
        if (idx == 0) continue; // glyph not found
        if (idx > 0) {
            // second part of byte
            idx = this->data.glyphIndex[idx * 16 + ((unsigned char)txtutf8[i] / 0x10)];
            if (idx == 0) continue; // glyph not found
            if (idx > 0) continue; // glyph key not complete
        }
        idx = -idx; // glyph found

        // add glyph to run
        if (txtutf8[i] == ' ') { // the only special white-space
            glyphrun[pos++] = 1 + idx;
            lineLength += this->data.glyph[idx].width;
            // no test for soft break here!
            if (!knowLastWhite || blackspace) {
                knowLastWhite = true;
                blackspace = false;
                lastWhiteGlyph = pos - 1;
            }
            lastWhiteSpace = i;

        } else if (nextAsNewLine) {
            nextAsNewLine = false;
            glyphrun[pos++] = -static_cast<int>(1 + idx);
            knowLastWhite = false;
            blackspace = true;
            lineLength = this->data.glyph[idx].width;

        } else {
            blackspace = true;
            glyphrun[pos++] = 1 + idx;
            lineLength += this->data.glyph[idx].width;
            // test for soft break
            if (lineLength > maxWidth) {
                // soft break
                if (knowLastWhite) {
                    i = lastWhiteSpace;
                    pos = lastWhiteGlyph + 1;
                    lineLength = 0.0f;
                    knowLastWhite = false;
                    nextAsNewLine = true;

                } else {
                    // last word to long
                    glyphrun[pos - 1] = -glyphrun[pos - 1];
                    lineLength = this->data.glyph[idx].width;

                }
            }

        }

        idx = 0; // start with new glyph search
    }

    return glyphrun;
}


/*
 * OutlineFont::drawFilled
 */
void OutlineFont::drawFilled(int *run, float x, float y, float size,
        bool flipY, Alignment align) const {
    float gx = x;
    float gy = y;
    float sy = flipY ? -size : size;

    if ((align == ALIGN_CENTER_BOTTOM) || (align == ALIGN_CENTER_MIDDLE)
            || (align == ALIGN_CENTER_TOP)) {
        gx -= this->lineWidth(run, false) * size * 0.5f;

    } else if ((align == ALIGN_RIGHT_BOTTOM) || (align == ALIGN_RIGHT_MIDDLE)
            || (align == ALIGN_RIGHT_TOP)) {
        gx -= this->lineWidth(run, false) * size;

    }

    ::glEnableClientState(GL_VERTEX_ARRAY);
    ::glDisable(GL_CULL_FACE);

    while (*run != 0) {
        const OutlineGlyphInfo &glyph = this->data.glyph[
            (*run < 0) ? (-1 - *run) : (*run - 1)];

        if (*run < 0) {
            gx = x;

            if ((align == ALIGN_CENTER_BOTTOM) || (align == ALIGN_CENTER_MIDDLE)
                    || (align == ALIGN_CENTER_TOP)) {
                gx -= this->lineWidth(run, false) * size * 0.5f;

            } else if ((align == ALIGN_RIGHT_BOTTOM) || (align == ALIGN_RIGHT_MIDDLE)
                    || (align == ALIGN_RIGHT_TOP)) {
                gx -= this->lineWidth(run, false) * size;

            }

            gy += sy;
        }

        ::glPushMatrix();
        ::glTranslatef(gx, gy, 0.0f);
        ::glScalef(size, sy, 1.0f);
        ::glVertexPointer(2, GL_FLOAT, 0, glyph.points);
        ::glDrawElements(GL_TRIANGLES, glyph.triCount, GL_UNSIGNED_SHORT, glyph.tris);
        ::glPopMatrix();

        gx += glyph.width * size;

        run++;
    }

    ::glDisableClientState(GL_VERTEX_ARRAY);
}


/*
 * OutlineFont::drawOutline
 */
void OutlineFont::drawOutline(int *run, float x, float y, float size,
        bool flipY, Alignment align) const {
    float gx = x;
    float gy = y;
    float sy = flipY ? -size : size;

    if ((align == ALIGN_CENTER_BOTTOM) || (align == ALIGN_CENTER_MIDDLE)
            || (align == ALIGN_CENTER_TOP)) {
        gx -= this->lineWidth(run, false) * size * 0.5f;

    } else if ((align == ALIGN_RIGHT_BOTTOM) || (align == ALIGN_RIGHT_MIDDLE)
            || (align == ALIGN_RIGHT_TOP)) {
        gx -= this->lineWidth(run, false) * size;

    }

    ::glEnableClientState(GL_VERTEX_ARRAY);

    while (*run != 0) {
        const OutlineGlyphInfo &glyph = this->data.glyph[
            (*run < 0) ? (-1 - *run) : (*run - 1)];

        if (*run < 0) {
            gx = x;

            if ((align == ALIGN_CENTER_BOTTOM) || (align == ALIGN_CENTER_MIDDLE)
                    || (align == ALIGN_CENTER_TOP)) {
                gx -= this->lineWidth(run, false) * size * 0.5f;

            } else if ((align == ALIGN_RIGHT_BOTTOM) || (align == ALIGN_RIGHT_MIDDLE)
                    || (align == ALIGN_RIGHT_TOP)) {
                gx -= this->lineWidth(run, false) * size;

            }

            gy += sy;
        }

        ::glPushMatrix();
        ::glTranslatef(gx, gy, 0.0f);
        ::glScalef(size, sy, 1.0f);
        ::glVertexPointer(2, GL_FLOAT, 0, glyph.points);
        unsigned int off = 0;
        for (unsigned int l = 0; l < glyph.loopCount; l++) {
            ::glDrawArrays(GL_LINE_LOOP, off, glyph.loopLength[l]);
            off += glyph.loopLength[l];
        }
        ::glPopMatrix();

        gx += glyph.width * size;

        run++;
    }

    ::glDisableClientState(GL_VERTEX_ARRAY);
}



/*
 * OutlineFont::lineCount
 */
int OutlineFont::lineCount(int *run, bool deleterun) const {
    if ((run == NULL) || (run[0] == 0)) return 0;
    int i = 1;
    for (int j = 0; run[j] != 0; j++) {
        if (run[j] < 0) i++;
    }
    if (deleterun) delete[] run;
    return i;
}


/*
 * OutlineFont::lineWidth
 */
float OutlineFont::lineWidth(int *&run, bool iterate) const {
    int *i = run;
    float len = 0.0f;
    while (*i != 0) {
        len += this->data.glyph[(*i < 0) ? (-1 - *i) : (*i - 1)].width;
        i++;
        if (*i < 0) break;
    }
    if (iterate) run = i;
    return len;
}
