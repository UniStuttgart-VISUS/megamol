/*
 * OutlineFont.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/OutlineFont.h"
#include <cfloat>
#include <GL/GL.h>
#include "vislib/CharTraits.h"
#include "vislib/memutils.h"
#include "vislib/UTF8Encoder.h"

using namespace vislib::graphics::gl;


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFontInfo& ofi) : AbstractFont(),
        data(ofi), renderType(OutlineFont::RENDERTYPE_FILL), glyphMesh(NULL) {
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFontInfo& ofi,
        OutlineFont::RenderType render) : AbstractFont(), data(ofi),
        renderType(render), glyphMesh(NULL) {
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFontInfo& ofi, float size)
        : AbstractFont(), data(ofi), renderType(OutlineFont::RENDERTYPE_FILL),
        glyphMesh(NULL) {
    this->SetSize(size);
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFontInfo& ofi, bool flipY)
        : AbstractFont(), data(ofi), renderType(OutlineFont::RENDERTYPE_FILL),
        glyphMesh(NULL) {
    this->SetFlipY(flipY);
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFontInfo& ofi,
        OutlineFont::RenderType render, bool flipY) : AbstractFont(),
        data(ofi), renderType(render), glyphMesh(NULL) {
    this->SetFlipY(flipY);
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFontInfo& ofi, float size, bool flipY)
        : AbstractFont(), data(ofi), renderType(OutlineFont::RENDERTYPE_FILL),
        glyphMesh(NULL) {
    this->SetSize(size);
    this->SetFlipY(flipY);
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFontInfo& ofi, float size,
        OutlineFont::RenderType render) : AbstractFont(), data(ofi),
        renderType(render), glyphMesh(NULL) {
    this->SetSize(size);
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFontInfo& ofi, float size,
        OutlineFont::RenderType render, bool flipY) : AbstractFont(),
        data(ofi), renderType(render), glyphMesh(NULL) {
    this->SetSize(size);
    this->SetFlipY(flipY);
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFont& src) : AbstractFont(),
        data(src.data), renderType(src.renderType), glyphMesh(NULL) {
    this->SetSize(src.GetSize());
    this->SetFlipY(src.IsFlipY());
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFont& src,
        OutlineFont::RenderType render) : AbstractFont(), data(src.data),
        renderType(render), glyphMesh(NULL) {
    this->SetSize(src.GetSize());
    this->SetFlipY(src.IsFlipY());
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFont& src, float size) : AbstractFont(),
        data(src.data), renderType(src.renderType), glyphMesh(NULL) {
    this->SetSize(size);
    this->SetFlipY(src.IsFlipY());
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFont& src, bool flipY) : AbstractFont(),
        data(src.data), renderType(src.renderType), glyphMesh(NULL) {
    this->SetSize(src.GetSize());
    this->SetFlipY(flipY);
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFont& src,
        OutlineFont::RenderType render, bool flipY) : AbstractFont(),
        data(src.data), renderType(render), glyphMesh(NULL) {
    this->SetSize(src.GetSize());
    this->SetFlipY(flipY);
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFont& src, float size, bool flipY)
        : AbstractFont(), data(src.data), renderType(src.renderType),
        glyphMesh(NULL) {
    this->SetSize(size);
    this->SetFlipY(flipY);
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFont& src, float size,
        OutlineFont::RenderType render) : AbstractFont(), data(src.data),
        renderType(render), glyphMesh(NULL) {
    this->SetSize(size);
    this->SetFlipY(src.IsFlipY());
}


/*
 * OutlineFont::OutlineFont
 */
OutlineFont::OutlineFont(const OutlineFont& src, float size,
        OutlineFont::RenderType render, bool flipY) : AbstractFont(),
        data(src.data), renderType(render), glyphMesh(NULL) {
    this->SetSize(size);
    this->SetFlipY(flipY);
}


/*
 * OutlineFont::~OutlineFont
 */
OutlineFont::~OutlineFont(void) {
    this->Deinitialise();
    ARY_SAFE_DELETE(this->glyphMesh);
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
    if ((this->renderType == RENDERTYPE_FILL)
            || (this->renderType == RENDERTYPE_FILL_AND_OUTLINE)) {
        this->buildUpFillingMeshes();
    }
    return true;
}


/*
 * OutlineFont::deinitialise
 */
void OutlineFont::deinitialise(void) {
    ARY_SAFE_DELETE(this->glyphMesh);
}


/*
 * buildUpFillingMeshes
 */
void OutlineFont::buildUpFillingMeshes(void) {
    // TODO: Implement
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
    int keylen = 1;
    bool knowLastWhite = false;
    bool blackspace = true;
    SIZE_T lastWhiteGlyph;
    SIZE_T lastWhiteSpace;
    float lineLength = 0.0f;
    bool nextAsNewLine = false;
    unsigned int idx;
    ::memset(glyphrun, 0, sizeof(int) * (txtlen + 1));
    // > 0 1+index of the glyph to use
    // < 0 -(1+index) of the glyph and new line
    // = 0 end

    // build glyph run
    for (SIZE_T i = 0; i < txtlen; i += keylen) {
        // special handle new lines
        if (txtutf8[i] == '\n') {
            nextAsNewLine = true;
            continue;
        }

        // select glyph
        idx = 0xFFFFFFFF;
        keylen = 1;
        while ((txtutf8[i - 1 + keylen] & 0x80) != 0) keylen++;
        // TODO: Tree search!!!!
        for (unsigned int j = 0; j < this->data.glyphCount; j++) {
            if (strncmp(txtutf8 + i, this->data.glyph[j].utf8char, keylen) == 0) {
                idx = j;
                break;
            }
        }

        if (idx == 0xFFFFFFFF) { // glyph not found
            continue;
        }
        if (this->data.glyph[idx].utf8char[0] == ' ') {
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
                    pos = lastWhiteGlyph;
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

        // TODO: Implement

        //const float *pt = glyph.points;
        //for (unsigned int l = 0; l < glyph.loopCount; l++) {
        //    ::glBegin(GL_LINE_LOOP);
        //    for (unsigned int p = glyph.loopLength[l]; p > 0; p--) {
        //        ::glVertex2f(gx + pt[0] * size, gy + pt[1] * sy);
        //        pt += 2;
        //    }
        //    ::glEnd();
        //}

        gx += glyph.width * size;

        run++;
    }
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

        const float *pt = glyph.points;
        for (unsigned int l = 0; l < glyph.loopCount; l++) {
            ::glBegin(GL_LINE_LOOP);
            for (unsigned int p = glyph.loopLength[l]; p > 0; p--) {
                ::glVertex2f(gx + pt[0] * size, gy + pt[1] * sy);
                pt += 2;
            }
            ::glEnd();
        }

        gx += glyph.width * size;

        run++;
    }
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
