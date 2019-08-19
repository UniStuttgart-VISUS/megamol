/*
 * SDFFont.cpp
 *
 * Copyright (C) 2006 - 2018 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 *
 * This implementation is based on "vislib/graphics/OutlinetFont.h"
 */

#include "mmcore/utility/SDFFont.h"


using namespace vislib;
using namespace megamol::core::utility;


/* PUBLIC ********************************************************************/


/*
* SDFFont::SDFFont
*/
SDFFont::SDFFont(FontName fn) : AbstractFont(),
    renderType(SDFFont::RENDERTYPE_FILL), billboard(false), rotation(), initialised(false), shader(), shadervertcol(), texture(), vbos(), glyphs(), glyphIdcs(), glyphKrns(), useBatchDraw(false) {

    this->fontFileName = this->translateFontName(fn);
}


/*
* SDFFont::SDFFont
*/
SDFFont::SDFFont(FontName fn, SDFFont::RenderType render) : AbstractFont(),
    renderType(render), billboard(false), rotation(), initialised(false), shader(), shadervertcol(), texture(), vbos(), glyphs(), glyphIdcs(), glyphKrns(), useBatchDraw(false) {

    this->fontFileName = this->translateFontName(fn);
}


/*
* SDFFont::SDFFont
*/
SDFFont::SDFFont(FontName fn, float size) : AbstractFont(),
    renderType(SDFFont::RENDERTYPE_FILL), billboard(false), rotation(), initialised(false), shader(), shadervertcol(), texture(), vbos(), glyphs(), glyphIdcs(), glyphKrns(), useBatchDraw(false) {

    this->SetSize(size);
    this->fontFileName = this->translateFontName(fn);
}


/*
* SDFFont::SDFFont
*/
SDFFont::SDFFont(FontName fn, bool flipY) : AbstractFont(),
    renderType(SDFFont::RENDERTYPE_FILL), billboard(false), rotation(), initialised(false), shader(), shadervertcol(), texture(), vbos(), glyphs(), glyphIdcs(), glyphKrns(), useBatchDraw(false) {

    this->SetFlipY(flipY);
    this->fontFileName = this->translateFontName(fn);
}


/*
* SDFFont::SDFFont
*/
SDFFont::SDFFont(FontName fn, SDFFont::RenderType render, bool flipY) : AbstractFont(),
    renderType(render), billboard(false), rotation(), initialised(false), shader(), shadervertcol(), texture(), vbos(), glyphs(), glyphIdcs(), glyphKrns(), useBatchDraw(false) {

    this->SetFlipY(flipY);
    this->fontFileName = this->translateFontName(fn);
}


/*
* SDFFont::SDFFont
*/
SDFFont::SDFFont(FontName fn, float size, bool flipY) : AbstractFont(),
    renderType(SDFFont::RENDERTYPE_FILL), billboard(false), rotation(), initialised(false), shader(), shadervertcol(), texture(), vbos(), glyphs(), glyphIdcs(), glyphKrns(), useBatchDraw(false) {

    this->SetSize(size);
    this->SetFlipY(flipY);
    this->fontFileName = this->translateFontName(fn);
}


/*
* SDFFont::SDFFont
*/
SDFFont::SDFFont(FontName fn, float size, SDFFont::RenderType render) : AbstractFont(),
    renderType(render), billboard(false), rotation(), initialised(false), shader(), shadervertcol(), texture(), vbos(), glyphs(), glyphIdcs(), glyphKrns(), useBatchDraw(false) {

    this->SetSize(size);
    this->fontFileName = this->translateFontName(fn);
}


/*
* SDFFont::SDFFont
*/
SDFFont::SDFFont(FontName fn, float size, SDFFont::RenderType render, bool flipY) : AbstractFont(),
    renderType(render), billboard(false), rotation(), initialised(false), shader(), shadervertcol(), texture(), vbos(), glyphs(), glyphIdcs(), glyphKrns(), useBatchDraw(false) {

    this->SetSize(size);
    this->SetFlipY(flipY);
    this->fontFileName = this->translateFontName(fn);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(vislib::StringA fn) : AbstractFont(),
    fontFileName(fn), renderType(SDFFont::RENDERTYPE_FILL), billboard(false), rotation(), initialised(false), shader(), shadervertcol(), texture(), vbos(), glyphs(), glyphIdcs(), glyphKrns(), useBatchDraw(false)  {

}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(vislib::StringA fn,  SDFFont::RenderType render) : AbstractFont(),
    fontFileName(fn), renderType(render), billboard(false), rotation(), initialised(false), shader(), shadervertcol(), texture(), vbos(), glyphs(), glyphIdcs(), glyphKrns(), useBatchDraw(false) {

}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(vislib::StringA fn, float size)  : AbstractFont(),
    fontFileName(fn), renderType(SDFFont::RENDERTYPE_FILL), billboard(false), rotation(), initialised(false), shader(), shadervertcol(), texture(), vbos(), glyphs(), glyphIdcs(), glyphKrns(), useBatchDraw(false) {

    this->SetSize(size);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(vislib::StringA fn, bool flipY) : AbstractFont(),
    fontFileName(fn), renderType(SDFFont::RENDERTYPE_FILL), billboard(false), rotation(), initialised(false), shader(), shadervertcol(), texture(), vbos(), glyphs(), glyphIdcs(), glyphKrns(), useBatchDraw(false) {

    this->SetFlipY(flipY);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(vislib::StringA fn, SDFFont::RenderType render, bool flipY) : AbstractFont(),
    fontFileName(fn), renderType(render), billboard(false), rotation(), initialised(false), shader(), shadervertcol(), texture(), vbos(), glyphs(), glyphIdcs(), glyphKrns(), useBatchDraw(false) {

    this->SetFlipY(flipY);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(vislib::StringA fn, float size, bool flipY) : AbstractFont(),
    fontFileName(fn), renderType(SDFFont::RENDERTYPE_FILL), billboard(false), rotation(), initialised(false), shader(), shadervertcol(), texture(), vbos(), glyphs(), glyphIdcs(), glyphKrns(), useBatchDraw(false) {

    this->SetSize(size);
    this->SetFlipY(flipY);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(vislib::StringA fn, float size, SDFFont::RenderType render) : AbstractFont(),
    fontFileName(fn), renderType(render), billboard(false), rotation(), initialised(false), shader(), shadervertcol(), texture(), vbos(), glyphs(), glyphIdcs(), glyphKrns(), useBatchDraw(false) {

    this->SetSize(size);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(vislib::StringA fn, float size, SDFFont::RenderType render, bool flipY) : AbstractFont(),
        fontFileName(fn), renderType(render), billboard(false), rotation(), initialised(false), shader(), shadervertcol(), texture(), vbos(), glyphs(), glyphIdcs(), glyphKrns(), useBatchDraw(false) {

    this->SetSize(size);
    this->SetFlipY(flipY);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src) : AbstractFont(),
    fontFileName(src.fontFileName), renderType(src.renderType), billboard(false), rotation(), initialised(false), shader(), shadervertcol(), texture(), vbos(), glyphs(), glyphIdcs(), glyphKrns(), useBatchDraw(false) {

    this->SetSize(src.GetSize());
    this->SetFlipY(src.IsFlipY());
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src, SDFFont::RenderType render) : AbstractFont(),
    fontFileName(src.fontFileName), renderType(render), billboard(false), rotation(), initialised(false), shader(), shadervertcol(), texture(), vbos(), glyphs(), glyphIdcs(), glyphKrns(), useBatchDraw(false) {

    this->SetSize(src.GetSize());
    this->SetFlipY(src.IsFlipY());
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src, float size) : AbstractFont(),
        fontFileName(src.fontFileName), renderType(src.renderType), billboard(false), rotation(), initialised(false), shader(), shadervertcol(), texture(), vbos(), glyphs(), glyphIdcs(), glyphKrns(), useBatchDraw(false) {

    this->SetSize(size);
    this->SetFlipY(src.IsFlipY());
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src, bool flipY) : AbstractFont(),
    fontFileName(src.fontFileName), renderType(src.renderType), billboard(false), rotation(), initialised(false), shader(), shadervertcol(), texture(), vbos(), glyphs(), glyphIdcs(), glyphKrns(), useBatchDraw(false){

    this->SetSize(src.GetSize());
    this->SetFlipY(flipY);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src, SDFFont::RenderType render, bool flipY) : AbstractFont(),
        fontFileName(src.fontFileName), renderType(render), billboard(false), rotation(), initialised(false), shader(), shadervertcol(), texture(), vbos(), glyphs(), glyphIdcs(), glyphKrns(), useBatchDraw(false) {

    this->SetSize(src.GetSize());
    this->SetFlipY(flipY);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src, float size, bool flipY) : AbstractFont(),
    fontFileName(src.fontFileName), renderType(src.renderType), billboard(false), rotation(), initialised(false), shader(), shadervertcol(), texture(), vbos(), glyphs(), glyphIdcs(), glyphKrns(), useBatchDraw(false) {

    this->SetSize(size);
    this->SetFlipY(flipY);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src, float size,  SDFFont::RenderType render) : AbstractFont(),
    fontFileName(src.fontFileName),  renderType(render), billboard(false), rotation(), initialised(false), shader(), shadervertcol(), texture(), vbos(), glyphs(), glyphIdcs(), glyphKrns(), useBatchDraw(false) {

    this->SetSize(size);
    this->SetFlipY(src.IsFlipY());
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src, float size, SDFFont::RenderType render, bool flipY) : AbstractFont(),
        fontFileName(src.fontFileName), renderType(render), billboard(false), rotation(), initialised(false), shader(), shadervertcol(), texture(), vbos(), glyphs(), glyphIdcs(), glyphKrns(), useBatchDraw(false) {

    this->SetSize(size);
    this->SetFlipY(flipY);
}


/*
 * SDFFont::~SDFFont
 */
SDFFont::~SDFFont(void) {

    this->Deinitialise();
}


/*
 * SDFFont::BlockLines
 */
unsigned int SDFFont::BlockLines(float maxWidth, float size, const char *txt) const {

    return this->lineCount(this->buildGlyphRun(txt, maxWidth / size), true);
}


/*
 * SDFFont::BlockLines
 */
unsigned int SDFFont::BlockLines(float maxWidth, float size, const wchar_t *txt) const {

    return this->lineCount(this->buildGlyphRun(txt, maxWidth / size), true);
}


/*
 * SDFFont::DrawString
 */
void SDFFont::DrawString(const float col[4], float x, float y, float size, bool flipY, const char *txt, AbstractFont::Alignment align) const {

    if (!this->initialised || (this->renderType == RenderType::RENDERTYPE_NONE)) return;

    int *run = this->buildGlyphRun(txt, FLT_MAX);

    if ((align == ALIGN_CENTER_MIDDLE) || (align == ALIGN_LEFT_MIDDLE) || (align == ALIGN_RIGHT_MIDDLE)) {
        y += static_cast<float>(this->lineCount(run, false)) * 0.5f * size *  (flipY ? -1.0f : 1.0f);

    } else if ((align == ALIGN_CENTER_BOTTOM) || (align == ALIGN_LEFT_BOTTOM) || (align == ALIGN_RIGHT_BOTTOM)) {
        y += static_cast<float>(this->lineCount(run, false)) * size *  (flipY ? -1.0f : 1.0f);
    }

    this->drawGlyphs(col, run, x, y, 0.0f, size, flipY, align);

    ARY_SAFE_DELETE(run);
}


/*
* SDFFont::DrawString
*/
void SDFFont::DrawString(const float col[4], float x, float y, float size, bool flipY, const wchar_t *txt, AbstractFont::Alignment align) const {

    if (!this->initialised || (this->renderType == RenderType::RENDERTYPE_NONE)) return;

    int *run = this->buildGlyphRun(txt, FLT_MAX);

    if ((align == ALIGN_CENTER_MIDDLE) || (align == ALIGN_LEFT_MIDDLE) || (align == ALIGN_RIGHT_MIDDLE)) {
        y += static_cast<float>(this->lineCount(run, false)) * 0.5f * size *  (flipY ? -1.0f : 1.0f);

    }
    else if ((align == ALIGN_CENTER_BOTTOM) || (align == ALIGN_LEFT_BOTTOM) || (align == ALIGN_RIGHT_BOTTOM)) {
        y += static_cast<float>(this->lineCount(run, false)) * size *  (flipY ? -1.0f : 1.0f);
    }
    
    this->drawGlyphs(col, run, x, y, 0.0f, size, flipY, align);

    ARY_SAFE_DELETE(run);
}


/*
 * SDFFont::DrawString
 */
void SDFFont::DrawString(const float col[4], float x, float y, float w, float h, float size, bool flipY, const char *txt, AbstractFont::Alignment align) const {

    if (!this->initialised || (this->renderType == RenderType::RENDERTYPE_NONE)) return;

    int *run = this->buildGlyphRun(txt, w / size);

    if (flipY) y += h;

    switch (align) {
    case ALIGN_CENTER_BOTTOM:
        x += w * 0.5f;
        y +=  (flipY ? 1.0f : -1.0f) * (h - static_cast<float>(this->lineCount(run, false)) * size);
        break;
    case ALIGN_CENTER_MIDDLE:
        x += w * 0.5f;
        y +=  (flipY ? 1.0f : -1.0f) * (h - static_cast<float>(this->lineCount(run, false)) * size) * 0.5f;
        break;
    case ALIGN_CENTER_TOP:
        x += w * 0.5f;
        break;
    case ALIGN_LEFT_BOTTOM:
        y +=  (flipY ? 1.0f : -1.0f) * (h - static_cast<float>(this->lineCount(run, false)) * size);
        break;
    case ALIGN_LEFT_MIDDLE:
        y +=  (flipY ? 1.0f : -1.0f) * (h - static_cast<float>(this->lineCount(run, false)) * size) * 0.5f;
        break;
    case ALIGN_RIGHT_BOTTOM:
        x += w;
        y +=  (flipY ? 1.0f : -1.0f) * (h - static_cast<float>(this->lineCount(run, false)) * size);
        break;
    case ALIGN_RIGHT_MIDDLE:
        x += w;
        y +=  (flipY ? 1.0f : -1.0f) * (h - static_cast<float>(this->lineCount(run, false)) * size) * 0.5f;
        break;
    case ALIGN_RIGHT_TOP:
        x += w;
        break;
    default:
        break;
    }

    this->drawGlyphs(col, run, x, y, 0.0f, size, flipY, align);

    ARY_SAFE_DELETE(run);
}


/*
 * SDFFont::DrawString
 */
void SDFFont::DrawString(const float col[4], float x, float y, float w, float h, float size,  bool flipY, const wchar_t *txt, AbstractFont::Alignment align) const {

    if (!this->initialised || (this->renderType == RenderType::RENDERTYPE_NONE)) return;

    int *run = this->buildGlyphRun(txt, w / size);

    if (flipY) y += h;

    switch (align) {
    case ALIGN_CENTER_BOTTOM:
        x += w * 0.5f;
        y += (flipY ? 1.0f : -1.0f) * (h - static_cast<float>(this->lineCount(run, false)) * size);
        break;
    case ALIGN_CENTER_MIDDLE:
        x += w * 0.5f;
        y += (flipY ? 1.0f : -1.0f) * (h - static_cast<float>(this->lineCount(run, false)) * size) * 0.5f;
        break;
    case ALIGN_CENTER_TOP:
        x += w * 0.5f;
        break;
    case ALIGN_LEFT_BOTTOM:
        y += (flipY ? 1.0f : -1.0f) * (h - static_cast<float>(this->lineCount(run, false)) * size);
        break;
    case ALIGN_LEFT_MIDDLE:
        y += (flipY ? 1.0f : -1.0f) * (h - static_cast<float>(this->lineCount(run, false)) * size) * 0.5f;
        break;
    case ALIGN_RIGHT_BOTTOM:
        x += w;
        y += (flipY ? 1.0f : -1.0f) * (h - static_cast<float>(this->lineCount(run, false)) * size);
        break;
    case ALIGN_RIGHT_MIDDLE:
        x += w;
        y += (flipY ? 1.0f : -1.0f) * (h - static_cast<float>(this->lineCount(run, false)) * size) * 0.5f;
        break;
    case ALIGN_RIGHT_TOP:
        x += w;
        break;
    default:
        break;
    }

    this->drawGlyphs(col, run, x, y, 0.0f, size, flipY, align);

    ARY_SAFE_DELETE(run);
}


/*
* SDFFont::DrawString
*/
void SDFFont::DrawString(const float col[4], float x, float y, float z, float w, float h, float size, bool flipY, const char *txt, AbstractFont::Alignment align) const {

    if (!this->initialised || (this->renderType == RenderType::RENDERTYPE_NONE)) return;

    int *run = this->buildGlyphRun(txt, w / size);

    if (flipY) y += h;

    switch (align) {
    case ALIGN_CENTER_BOTTOM:
        x += w * 0.5f;
        y += (flipY ? 1.0f : -1.0f) * (h - static_cast<float>(this->lineCount(run, false)) * size);
        break;
    case ALIGN_CENTER_MIDDLE:
        x += w * 0.5f;
        y += (flipY ? 1.0f : -1.0f) * (h - static_cast<float>(this->lineCount(run, false)) * size) * 0.5f;
        break;
    case ALIGN_CENTER_TOP:
        x += w * 0.5f;
        break;
    case ALIGN_LEFT_BOTTOM:
        y += (flipY ? 1.0f : -1.0f) * (h - static_cast<float>(this->lineCount(run, false)) * size);
        break;
    case ALIGN_LEFT_MIDDLE:
        y += (flipY ? 1.0f : -1.0f) * (h - static_cast<float>(this->lineCount(run, false)) * size) * 0.5f;
        break;
    case ALIGN_RIGHT_BOTTOM:
        x += w;
        y += (flipY ? 1.0f : -1.0f) * (h - static_cast<float>(this->lineCount(run, false)) * size);
        break;
    case ALIGN_RIGHT_MIDDLE:
        x += w;
        y += (flipY ? 1.0f : -1.0f) * (h - static_cast<float>(this->lineCount(run, false)) * size) * 0.5f;
        break;
    case ALIGN_RIGHT_TOP:
        x += w;
        break;
    default:
        break;
    }

    this->drawGlyphs(col, run, x, y, z, size, flipY, align);

    ARY_SAFE_DELETE(run);
}


/*
* SDFFont::DrawString
*/
void SDFFont::DrawString(const float col[4], float x, float y, float z, float w, float h, float size, bool flipY, const wchar_t *txt, AbstractFont::Alignment align) const {

    if (!this->initialised || (this->renderType == RenderType::RENDERTYPE_NONE)) return;

    int *run = this->buildGlyphRun(txt, w / size);

    if (flipY) y += h;

    switch (align) {
    case ALIGN_CENTER_BOTTOM:
        x += w * 0.5f;
        y += (flipY ? 1.0f : -1.0f) * (h - static_cast<float>(this->lineCount(run, false)) * size);
        break;
    case ALIGN_CENTER_MIDDLE:
        x += w * 0.5f;
        y += (flipY ? 1.0f : -1.0f) * (h - static_cast<float>(this->lineCount(run, false)) * size) * 0.5f;
        break;
    case ALIGN_CENTER_TOP:
        x += w * 0.5f;
        break;
    case ALIGN_LEFT_BOTTOM:
        y += (flipY ? 1.0f : -1.0f) * (h - static_cast<float>(this->lineCount(run, false)) * size);
        break;
    case ALIGN_LEFT_MIDDLE:
        y += (flipY ? 1.0f : -1.0f) * (h - static_cast<float>(this->lineCount(run, false)) * size) * 0.5f;
        break;
    case ALIGN_RIGHT_BOTTOM:
        x += w;
        y += (flipY ? 1.0f : -1.0f) * (h - static_cast<float>(this->lineCount(run, false)) * size);
        break;
    case ALIGN_RIGHT_MIDDLE:
        x += w;
        y += (flipY ? 1.0f : -1.0f) * (h - static_cast<float>(this->lineCount(run, false)) * size) * 0.5f;
        break;
    case ALIGN_RIGHT_TOP:
        x += w;
        break;
    default:
        break;
    }

    this->drawGlyphs(col, run, x, y, z, size, flipY, align);

    ARY_SAFE_DELETE(run);
}


/*
* SDFFont::DrawString
*/
void SDFFont::DrawString(const float col[4], float x, float y, float z, float size, bool flipY, const char * txt, Alignment align) const {

    if (!this->initialised || (this->renderType == RenderType::RENDERTYPE_NONE)) return;

    int *run = this->buildGlyphRun(txt, FLT_MAX);

    if ((align == ALIGN_CENTER_MIDDLE) || (align == ALIGN_LEFT_MIDDLE) || (align == ALIGN_RIGHT_MIDDLE)) {
        y += static_cast<float>(this->lineCount(run, false)) * 0.5f * size *  (flipY ? -1.0f : 1.0f);
    }
    else if ((align == ALIGN_CENTER_BOTTOM) || (align == ALIGN_LEFT_BOTTOM) || (align == ALIGN_RIGHT_BOTTOM)) {
        y += static_cast<float>(this->lineCount(run, false)) * size *  (flipY ? -1.0f : 1.0f);
    }

    this->drawGlyphs(col, run, x, y, z, size, flipY, align);

    ARY_SAFE_DELETE(run);
}


/*
* SDFFont::DrawString
*/
void SDFFont::DrawString(const float col[4], float x, float y, float z, float size, bool flipY, const wchar_t * txt, Alignment align) const {

    if (!this->initialised || (this->renderType == RenderType::RENDERTYPE_NONE)) return;

    int *run = this->buildGlyphRun(txt, FLT_MAX);

    if ((align == ALIGN_CENTER_MIDDLE) || (align == ALIGN_LEFT_MIDDLE) || (align == ALIGN_RIGHT_MIDDLE)) {
        y += static_cast<float>(this->lineCount(run, false)) * 0.5f * size  *  (flipY ? -1.0f : 1.0f);
    }
    else if ((align == ALIGN_CENTER_BOTTOM) || (align == ALIGN_LEFT_BOTTOM) || (align == ALIGN_RIGHT_BOTTOM)) {
        y += static_cast<float>(this->lineCount(run, false)) * size *  (flipY ? -1.0f : 1.0f);
    }

    this->drawGlyphs(col, run, x, y, z, size, flipY, align);

    ARY_SAFE_DELETE(run);
}


/*
* SDFFont::LineWidth
*/
float SDFFont::LineWidth(float size, const char *txt) const {

    int *run = this->buildGlyphRun(txt, FLT_MAX);
    int *i = run;
    float len = 0.0f;
    float comlen = 0.0f;
    while (*i != 0) {
        comlen = this->lineWidth(i, true);
        if (comlen > len) {
            len = comlen;
        }
    }
    ARY_SAFE_DELETE(run);
    return len * size;
}


/*
* SDFFont::LineWidth
*/
float SDFFont::LineWidth(float size, const wchar_t *txt) const {

    int *run = this->buildGlyphRun(txt, FLT_MAX);
    int *i = run;
    float len = 0.0f;
    float comlen = 0.0f;
    while (*i != 0) {
        comlen = this->lineWidth(i, true);
        if (comlen > len) {
            len = comlen;
        }
    }
    ARY_SAFE_DELETE(run);
    return len * size;
}


/*
 * SDFFont::BatchDrawString
 */
void SDFFont::BatchDrawString(const float col[4]) const {

    // Bind glyph data in batch cache
    for (unsigned int i = 0; i < (unsigned int)this->vbos.size(); i++) {
        glBindBuffer(GL_ARRAY_BUFFER, this->vbos[i].handle);
        if (this->vbos[i].index == (GLuint)VBOAttrib::POSITION) {
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)this->posBatchCache.size() * sizeof(GLfloat), &this->posBatchCache.front(), GL_STATIC_DRAW);
        }
        else if (this->vbos[i].index == (GLuint)VBOAttrib::TEXTURE) {
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)this->texBatchCache.size() * sizeof(GLfloat), &this->texBatchCache.front(), GL_STATIC_DRAW);
        } 
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    // Draw batch cache 
    unsigned int glyphCnt = ((unsigned int)this->posBatchCache.size() / 18); // 18 = 2 Triangles * 3 Vertices * 3 Coordinates
    this->render(glyphCnt, &col);
}


/*
 * SDFFont::BatchDrawString
 */
void SDFFont::BatchDrawString() const {

    // Bind glyph data in batch cache
    for (unsigned int i = 0; i < (unsigned int)this->vbos.size(); i++) {
        glBindBuffer(GL_ARRAY_BUFFER, this->vbos[i].handle);
        if (this->vbos[i].index == (GLuint)VBOAttrib::POSITION) {
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)this->posBatchCache.size() * sizeof(GLfloat), &this->posBatchCache.front(), GL_STATIC_DRAW);
        } 
        else if (this->vbos[i].index == (GLuint)VBOAttrib::TEXTURE) {
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)this->texBatchCache.size() * sizeof(GLfloat), &this->texBatchCache.front(), GL_STATIC_DRAW);
        } 
        else if (this->vbos[i].index == (GLuint)VBOAttrib::COLOR) {
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)this->colBatchCache.size() * sizeof(GLfloat), &this->colBatchCache.front(), GL_STATIC_DRAW);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    // Draw batch cache
    unsigned int glyphCnt = ((unsigned int)this->posBatchCache.size() / 18); // 18 = 2 Triangles * 3 Vertices * 3 Coordinates
    this->render(glyphCnt, nullptr);
}

/* PRIVATE********************************************************************/


/*
 * SDFFont::initialise
 */
bool SDFFont::initialise(megamol::core::CoreInstance *core) {
    
    if (!this->initialised)
        this->loadFont(core);

    return this->initialised;
}


/*
 * SDFFont::deinitialise
 */
void SDFFont::deinitialise(void) {

    // String cache
    this->ClearBatchDrawCache();

    // Texture
    this->texture.Release();

    // Shader
    this->shader.Release();
    this->shadervertcol.Release();

    // VBOs
    for (unsigned int i = 0; i < (unsigned int)this->vbos.size(); i++) {
        glDeleteBuffers(1, &this->vbos[i].handle);
    }
    this->vbos.clear();

    // VAO
    glDeleteVertexArrays(1, &this->vaoHandle);
    // Delete allocated memory
    for (unsigned int i = 0; i < this->glyphs.size(); i++) {
        if (this->glyphs[i].kerns != nullptr) {
            delete [] this->glyphs[i].kerns;
        }
    }

    // Set glyph indey pointer to null
    for (size_t i = 0; i < this->glyphIdcs.size(); i++) {
        this->glyphIdcs[i] = nullptr;
    }

    this->initialised = false;
}


/*
* SDFFont::lineCount
*/
int SDFFont::lineCount(int *run, bool deleterun) const {
    if ((run == nullptr) || (run[0] == 0)) return 0;
    int i = 1;
    for (int j = 0; run[j] != 0; j++) {
        if (run[j] < 0) i++;
    }
    if (deleterun) ARY_SAFE_DELETE(run);

return i;
}


/*
* SDFFont::lineWidth
*/
float SDFFont::lineWidth(int *&run, bool iterate) const {

    int *i = run;
    float len = 0.0f;
    while (*i != 0) {
        len += this->glyphIdcs[((*i) < 0) ? (-1 - (*i)) : ((*i) - 1)]->xadvance; // No check -> requires valid run!
        i++;
        if (*i < 0) break;
    }
    if (iterate) run = i;

    return len;
}


/*
* SDFFont::buildGlyphRun
*/
int *SDFFont::buildGlyphRun(const char *txt, float maxWidth) const {

    vislib::StringA txtutf8;
    if (!vislib::UTF8Encoder::Encode(txtutf8, txt)) {
        // encoding failed ... how?
        char *t = txtutf8.AllocateBuffer(vislib::CharTraitsA::SafeStringLength(txt));
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
* SDFFont::buildGlyphRun
*/
int *SDFFont::buildGlyphRun(const wchar_t *txt, float maxWidth) const {

    vislib::StringA txtutf8;
    if (!vislib::UTF8Encoder::Encode(txtutf8, txt)) {
        // encoding failed ... how?
        char *t = txtutf8.AllocateBuffer(vislib::CharTraitsW::SafeStringLength(txt));
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
* SDFFont::buildUpGlyphRun
*/
int *SDFFont::buildUpGlyphRun(const char *txtutf8, float maxWidth) const {

    size_t txtlen = static_cast<size_t>(CharTraitsA::SafeStringLength(txtutf8));
    size_t pos = 0;
    bool knowLastWhite = false;
    bool blackspace = true;
    size_t lastWhiteGlyph = 0;
    size_t lastWhiteSpace = 0;
    float lineLength = 0.0f;
    bool nextAsNewLine = false;

    unsigned int folBytes = 0; // following bytes
    unsigned int idx = 0;
    unsigned int tmpIdx = 0;

    int *glyphrun = new int[txtlen + 1];
    ::memset(glyphrun, 0, sizeof(int) * (txtlen + 1));

    // A run creates an array of decimal utf8 indices for each (utf8) charater.
    // A negative index indicates a new line.

    // > 0 1+index of the glyph to use
    // < 0 -(1+index) of the glyph and new line
    // = 0 end

    // build glyph run
    for (size_t i = 0; i < txtlen; i++) {

        if (txtutf8[i] == '\n') { // special handle new lines
            nextAsNewLine = true;
            continue;
        }

        // --------------------------------------------------------------------
        // UTF8-Bytes to Decimal
        // NB:
        // -'Unsigned int' needs to have at least 3 bytes for encoding utf8 in decimal.
        // -(Following variables are "unisgned" so that always zeros are shifted and not ones ...)
        // -! so far: THERE IS NO COMPLETE CHECK FOR INVALID UTF8 BYTE SEQUENCES ... (only slowing down performance)
        // - Therefore ASSUMING well formed utf8 encoding ...

        unsigned char byte = txtutf8[i];
        // If byte >= 0 -> ASCII-Byte: 0XXXXXXX = 0...127
        if (byte < 128) { 
            idx = static_cast<unsigned int>(byte);
        }
        else { // ... else if byte >= 128 => UTF8-Byte: 1XXXXXXX 
            // Supporting UTF8 for up to 3 bytes:
            if (byte >= (unsigned char)(0b11100000)) {                       // => >224 - 1110XXXX -> start 3-Byte UTF8, 2 bytes are following
                folBytes = 2;
                idx = (unsigned int)(byte & (unsigned char)(0b00001111));    // => consider only last 4 bits
                idx = (idx << 12);                                           // => 2*6 Bits are following
                continue;
            }
            else if (byte >= (unsigned char)(0b11000000)) {                  // => >192 - 110XXXXX -> start 2-Byte UTF8, 1 byte is following
                folBytes = 1;
                idx = (unsigned int)(byte & (unsigned char)(0b00011111));    // => consider only last 5 bits
                idx = (idx << 6);                                            // => 1*6 Bits are following
                continue;
            }
            else if (byte >= (unsigned char)(0b10000000)) {                  // => >128 - 10XXXXXX -> "following" 1-2 bytes
                folBytes--;
                tmpIdx = (unsigned int)(byte & (unsigned char)(0b00111111)); // => consider only last 6 bits
                idx    = (idx | (tmpIdx << (folBytes*6)));                   // => shift tmpIdx depending on following byte and 'merge' (|) with idx
                if (folBytes > 0)  continue;                                 // => else idx is complete
            }
        }

        // Check if glyph info is available
        if (idx > (unsigned int)this->glyphIdcs.size()) {
            //vislib::sys::Log::DefaultLog.WriteWarn("[SDFFont] [buildUpGlyphRun] Glyph index greater than available: \"%i\" > max. Index = \"%i\".\n", idx, this->idxCnt);
            continue;
        }
        if (this->glyphIdcs[idx] == nullptr) {
            //vislib::sys::Log::DefaultLog.WriteWarn("[SDFFont] [buildUpGlyphRun] Glyph info not available for: \"%i\".\n", idx);
            continue;
        }

        // --------------------------------------------------------------------

        // add glyph to run
        if (txtutf8[i] == ' ') { // the only special white-space
            glyphrun[pos++] = static_cast<int>(1 + idx);
            lineLength += this->glyphIdcs[idx]->xadvance;
            // no test for soft break here!
            if (!knowLastWhite || blackspace) {
                knowLastWhite  = true;
                blackspace     = false;
                lastWhiteGlyph = pos - 1;
            }
            lastWhiteSpace = i;
        }
        else if (nextAsNewLine) {
            nextAsNewLine   = false;
            glyphrun[pos++] = -static_cast<int>(1 + idx);
            knowLastWhite   = false;
            blackspace      = true;
            lineLength      = this->glyphIdcs[idx]->xadvance;
        }
        else {
            blackspace      = true;
            glyphrun[pos++] = static_cast<int>(1 + idx);
            lineLength     += this->glyphIdcs[idx]->xadvance;
            // test for soft break
            if (lineLength > maxWidth) {
                // soft break
                if (knowLastWhite) {
                    i             = lastWhiteSpace;
                    pos           = lastWhiteGlyph + 1;
                    lineLength    = 0.0f;
                    knowLastWhite = false;
                    nextAsNewLine = true;
                }
                else {
                    // last word to long
                    glyphrun[pos - 1] = -glyphrun[pos - 1];
                    lineLength        = this->glyphIdcs[idx]->xadvance;
                }
            }
        }
    }

    return glyphrun;
}


/*
* SDFFont::drawGlyphs
*/
void SDFFont::drawGlyphs(const float col[4], int* run, float x, float y, float z, float size, bool flipY, Alignment align) const {

    // Data buffers
    unsigned glyphCnt = 0;
    int *tmpRun = run;
    while ((*tmpRun) != 0) {
        tmpRun++;
        glyphCnt++;
    }
    unsigned int posCnt = glyphCnt * 18; // 2 Triangles * 3 Vertices * 3 Coordinates
    unsigned int texCnt = glyphCnt * 12; // 2 Triangles * 3 Vertices * 2 Coordinates

    GLfloat *posData = new GLfloat[posCnt];   
    GLfloat *texData = new GLfloat[texCnt]; 

    float gx = x;
    float gy = y;
    float gz = z;

    float sy = (flipY) ? (size) : (-size);
    float kern = 0.0f;
    unsigned int lastGlyphId = 0;

    // Billboard stuff
    // -> Setting fixed rotation point depending on alignment.
    vislib::math::Vector<GLfloat, 4> billboardRotPoint;

    if (this->billboard) {
        float deltaY = 0.0f;
        switch (align) {
        case ALIGN_LEFT_MIDDLE:
            deltaY = static_cast<float>(this->lineCount(run, false)) * sy * 0.5f;
            break;
        case ALIGN_LEFT_BOTTOM:
            deltaY = static_cast<float>(this->lineCount(run, false)) * sy;
            break;
        case ALIGN_CENTER_MIDDLE:
            deltaY = static_cast<float>(this->lineCount(run, false)) * sy * 0.5f;
            break;
        case ALIGN_CENTER_BOTTOM:
            deltaY = static_cast<float>(this->lineCount(run, false)) * sy;
            break;
        case ALIGN_RIGHT_MIDDLE:
            deltaY = static_cast<float>(this->lineCount(run, false)) * sy * 0.5f;
            break;
        case ALIGN_RIGHT_BOTTOM:
            deltaY = static_cast<float>(this->lineCount(run, false)) * sy;
            break;
        default:
            break;
        }
        billboardRotPoint.Set(gx, gy + deltaY, gz, 1.0f);

        GLfloat modelViewMatrix_column[16];
        glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix_column);
        vislib::math::ShallowMatrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewMatrix(&modelViewMatrix_column[0]);

        // Apply model view matrix ONLY to rotation point ...
        billboardRotPoint = modelViewMatrix * billboardRotPoint;
        billboardRotPoint.SetY(billboardRotPoint.Y() - deltaY);
        gx = billboardRotPoint.X();
        gy = billboardRotPoint.Y();
        gz = billboardRotPoint.Z();
    }

    // Adjustment for first line
    if ((align == ALIGN_CENTER_BOTTOM) || (align == ALIGN_CENTER_MIDDLE) || (align == ALIGN_CENTER_TOP)) {
        gx -= this->lineWidth(run, false) * size * 0.5f;
    }
    else if ((align == ALIGN_RIGHT_BOTTOM) || (align == ALIGN_RIGHT_MIDDLE) || (align == ALIGN_RIGHT_TOP)) {
        gx -= this->lineWidth(run, false) * size;
    }

    unsigned int glyphIter = 0;
    while ((*run) != 0) {

        // Run contains only available character indices (=> glyph is always != nullptr)
        SDFGlyphInfo *glyph = this->glyphIdcs[((*run) < 0) ? (-1 - (*run)) : ((*run) - 1)];

        // Adjust positions if character indicates a new line
        if ((*run) < 0) {
            gx = (this->billboard)?(billboardRotPoint.X()):(x);
            if ((align == ALIGN_CENTER_BOTTOM) || (align == ALIGN_CENTER_MIDDLE) || (align == ALIGN_CENTER_TOP)) {
                gx -= this->lineWidth(run, false) * size * 0.5f;
            }
            else if ((align == ALIGN_RIGHT_BOTTOM) || (align == ALIGN_RIGHT_MIDDLE) || (align == ALIGN_RIGHT_TOP)) {
                gx -= this->lineWidth(run, false) * size;
            }
            gy += (sy);
        }
        
        // ____________________________________
        //
        //          t3=p3-----------p2=t2
        //             |      /\     |
        //          height   /  \    |
        //             |    /----\   |
        //             |   /      \  |
        //     -----t0=p0---width---p1=t1
        //     |       |                   
        //  yoffset    |              
        //     |       |-kern-|     
        //     X---xoffset----|
        // (gx,gy,gz)
        //
        // ____________________________________

        // Get kerning
        kern = 0.0f;
        for (unsigned int i = 0; i < glyph->kernCnt; i++) {
            if (lastGlyphId == glyph->kerns[i].previous) {
                kern = glyph->kerns[i].xamount;
                break;
            }
        }

        // Temp position values
        float tmpP01x = size * (glyph->xoffset + kern) + gx;
        float tmpP23x = tmpP01x + (size * glyph->width);
        float tmpP03y = sy * (glyph->yoffset) + gy;
        float tmpP12y = tmpP03y + (sy * glyph->height);

        // Set position data:
        posData[glyphIter * 18 + 0]  = tmpP01x;   // p0-x
        posData[glyphIter * 18 + 1]  = tmpP03y;   // p0-y
        posData[glyphIter * 18 + 2]  = gz;        // p0-z

        posData[glyphIter * 18 + 3] = tmpP01x;    // p1-x
        posData[glyphIter * 18 + 4] = tmpP12y;    // p1-y
        posData[glyphIter * 18 + 5] = gz;         // p1-z

        posData[glyphIter * 18 + 6] = tmpP23x;    // p2-x
        posData[glyphIter * 18 + 7] = tmpP12y;    // p2-y
        posData[glyphIter * 18 + 8] = gz;         // p2-z

        posData[glyphIter * 18 + 9]  = tmpP01x;   // p0-x
        posData[glyphIter * 18 + 10] = tmpP03y;   // p0-y
        posData[glyphIter * 18 + 11] = gz;        // p0-z

        posData[glyphIter * 18 + 12] = tmpP23x;   // p2-x
        posData[glyphIter * 18 + 13] = tmpP12y;   // p2-y
        posData[glyphIter * 18 + 14] = gz;        // p2-z

        posData[glyphIter * 18 + 15] = tmpP23x;   // p3-x
        posData[glyphIter * 18 + 16] = tmpP03y;   // p3-y
        posData[glyphIter * 18 + 17] = gz;        // p3-z

        // Change rotation of quad positions for flipped y axis from CCW to CW.
        if (flipY) {
            posData[glyphIter * 18 + 3] = tmpP23x;   // p2-x
            posData[glyphIter * 18 + 6] = tmpP01x;   // p1-x
            posData[glyphIter * 18 + 13] = tmpP03y;  // p3-y
            posData[glyphIter * 18 + 16] = tmpP12y;  // p2-y
        }
        
        // Set texture data
        texData[glyphIter * 12 + 0] = glyph->texX0; // t0-x
        texData[glyphIter * 12 + 1] = glyph->texY0; // t0-y

        texData[glyphIter * 12 + 2] = glyph->texX0; // t1-x
        texData[glyphIter * 12 + 3] = glyph->texY1; // t1-y

        texData[glyphIter * 12 + 4] = glyph->texX1; // t2-x
        texData[glyphIter * 12 + 5] = glyph->texY1; // t2-y

        texData[glyphIter * 12 + 6] = glyph->texX0; // t0-x
        texData[glyphIter * 12 + 7] = glyph->texY0; // t0-y

        texData[glyphIter * 12 + 8]  = glyph->texX1; // t2-x
        texData[glyphIter * 12 + 9]  = glyph->texY1; // t2-y

        texData[glyphIter * 12 + 10] = glyph->texX1; // t3-x
        texData[glyphIter * 12 + 11] = glyph->texY0; // t3-y

        // Change rotation of texture coord for flipped y axis from CCW to CW.
        if (flipY) {
            texData[glyphIter * 12 + 2]  = glyph->texX1; // t2-x
            texData[glyphIter * 12 + 4]  = glyph->texX0; // t1-x
            texData[glyphIter * 12 + 9]  = glyph->texY0; // t3-y
            texData[glyphIter * 12 + 11] = glyph->texY1; // t2-y
        }

        // Update info for next character
        glyphIter++;
        gx += ((glyph->xadvance + kern) * size);
        lastGlyphId = glyph->id;
        run++;
    }

    if (this->GetBatchDrawMode()) {
        // Copy new data to batch cache ...
        for (unsigned int i = 0; i < posCnt; ++i) {
            this->posBatchCache.push_back(posData[i]);
        }
        for (unsigned int i = 0; i < texCnt; ++i) {
            this->texBatchCache.push_back(texData[i]);
        }
        for (unsigned int i = 0; i < (glyphCnt * 6); ++i) {
            this->colBatchCache.push_back(col[0]);
            this->colBatchCache.push_back(col[1]);
            this->colBatchCache.push_back(col[2]);
            this->colBatchCache.push_back(col[3]);
        }
    } 
    else {
        // ... or draw glyphs instantly.
        for (unsigned int i = 0; i < (unsigned int)this->vbos.size(); i++) {
            glBindBuffer(GL_ARRAY_BUFFER, this->vbos[i].handle);
            if (this->vbos[i].index == (GLuint)VBOAttrib::POSITION) {
                glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)posCnt * sizeof(GLfloat), posData, GL_STATIC_DRAW);
            }
            else if (this->vbos[i].index == (GLuint)VBOAttrib::TEXTURE) {
                glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)texCnt * sizeof(GLfloat), texData, GL_STATIC_DRAW);
            }
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }

        // Draw data buffers
        this->render(glyphCnt, &col);
    } 

    ARY_SAFE_DELETE(posData);
    ARY_SAFE_DELETE(texData);
}


/*
 * SDFFont::render
 */
void SDFFont::render(unsigned int gc, const float *col[4]) const {

    // Check texture
    if (!this->texture.IsValid()) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [render] Texture is not valid. \n");
        return;
    }

    // Check if per vertex color should be used
    const vislib::graphics::gl::GLSLShader *usedShader = &this->shader;
    if (col == nullptr) {
        usedShader = &this->shadervertcol;
    }

    // Check shaders
    if (!usedShader->IsValidHandle(usedShader->ProgramHandle())) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [render] Shader handle is not valid. \n");
        return;
    }

    // Get current matrices
    GLfloat projMatrix_column[16];
    glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
    vislib::math::ShallowMatrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> projMatrix(&projMatrix_column[0]);

    // modelviewprojection matrix
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewProjMatrix;
    if (this->billboard) {
        // Only projection matrix has to be applied for billboard
        modelViewProjMatrix = projMatrix;
    }
    else {
        GLfloat modelViewMatrix_column[16];
        glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix_column);
        vislib::math::ShallowMatrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewMatrix(&modelViewMatrix_column[0]);

        // Calculate model view projection matrix and apply rotation
        modelViewProjMatrix = projMatrix * modelViewMatrix * static_cast<vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR>>(this->rotation);
    }

    // Store/Set blending
    GLint blendSrc;
    GLint blendDst;
    glGetIntegerv(GL_BLEND_SRC, &blendSrc);
    glGetIntegerv(GL_BLEND_DST, &blendDst);
    bool blendEnabled = glIsEnabled(GL_BLEND);
    if (!blendEnabled) {
        glEnable(GL_BLEND);
    }
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // dFdx()/dFdx() in fragment shader:
    glHint(GL_FRAGMENT_SHADER_DERIVATIVE_HINT, GL_NICEST);

    glBindVertexArray(this->vaoHandle);

    glEnable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, this->texture.GetId()); // instead of this->texture.Bind() => because draw() is CONST

    glUseProgram(usedShader->ProgramHandle()); // instead of usedShader->Enable() => because draw() is CONST

    // Vertex shader
    glUniformMatrix4fv(usedShader->ParameterLocation("mvpMat"), 1, GL_FALSE, modelViewProjMatrix.PeekComponents());

    // Set global color, if given
    if (col != nullptr) {
        glUniform4fv(usedShader->ParameterLocation("inColor"), 1, (*col));
    }

    // Fragment shader
    glUniform1i(usedShader->ParameterLocation("fontTex"), 0);
    glUniform1i(usedShader->ParameterLocation("renderType"), (int)(this->renderType));

    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)gc * 6); // 2 triangles per glyph -> 6 vertices

    glUseProgram(0); // instead of usedShader->Disable() => because draw() is CONST
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);

    // Reset blending
    if (!blendEnabled) {
        glDisable(GL_BLEND);
    }
    glBlendFunc(blendSrc, blendDst);
}


/*
* SDFFont::loadFont
*/
bool SDFFont::loadFont(megamol::core::CoreInstance *core) {

    this->initialised = false;

    this->ResetRotation();
    this->SetBatchDrawMode(false);
    this->ClearBatchDrawCache();
    this->SetBillboardMode(false);

    if (core == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadFont] Pointer to MegaMol CoreInstance is NULL. \n");
        return false;
    }

    // (1) Load buffers --------------------------------------------------------
    ///vislib::sys::Log::DefaultLog.WriteInfo("[SDFFont] [loadFont] Loading OGL BUFFERS ... \n");
    if (!this->loadFontBuffers()) {
        vislib::sys::Log::DefaultLog.WriteWarn("[SDFFont] [loadFont] Failed to load buffers. \n");
        return false;
    }

    // (2) Load font information -----------------------------------------------
    vislib::StringA infoFile = this->fontFileName;
    infoFile.Append(".fnt");
    ///vislib::sys::Log::DefaultLog.WriteInfo("[SDFFont] [loadFont] Loading FONT INFO ... \n");
    if (!this->loadFontInfo(ResourceWrapper::getFileName(core->Configuration(), infoFile))) {
        vislib::sys::Log::DefaultLog.WriteWarn("[SDFFont] [loadFont] Failed to load font info file: \"%s\". \n", infoFile.PeekBuffer());
        return false;
    }

    // (3) Load font texture --------------------------------------------------------
    vislib::StringA textureFile = this->fontFileName;
    textureFile.Append(".png");
    ///vislib::sys::Log::DefaultLog.WriteInfo("[SDFFont] [loadFont] Loading FONT TEXTURE ... \n");
    if (!this->loadFontTexture(ResourceWrapper::getFileName(core->Configuration(), textureFile))) {
        vislib::sys::Log::DefaultLog.WriteWarn("[SDFFont] [loadFont] Failed to load font texture: \"%s\". \n", textureFile.PeekBuffer());
        return false;
    }

    // (4) Load shaders --------------------------------------------------------
    ///vislib::sys::Log::DefaultLog.WriteInfo("[SDFFont] [loadFont] Loading SHADERS ... \n");
    if (!this->loadFontShader(core)) {
        vislib::sys::Log::DefaultLog.WriteWarn("[SDFFont] [loadFont] Failed to load font shaders. \n");
        return false;
    }

    this->initialised = true;
    return true;
}


/*
* SDFFont::translateFontName
*/
vislib::StringA SDFFont::translateFontName(FontName fn) {

    vislib::StringA fileName = "";
    switch (fn) {
        case(SDFFont::FontName::EVOLVENTA_SANS): fileName = "Evolventa-SansSerif"; break;
        case(SDFFont::FontName::ROBOTO_SANS):    fileName = "Roboto-SansSerif";    break;
        case(SDFFont::FontName::VOLLKORN_SERIF): fileName = "Vollkorn-Serif";      break;
        case(SDFFont::FontName::UBUNTU_MONO):    fileName = "Ubuntu-Mono";         break;
        default: break;
    }
    return fileName;
}


/*
* SDFFont::loadFontBuffers
*/
bool SDFFont::loadFontBuffers() {

    // Reset 
    if (glIsVertexArray(this->vaoHandle)) {
        glDeleteVertexArrays(1, &this->vaoHandle);
    }
    for (unsigned int i = 0; i < (unsigned int)this->vbos.size(); i++) {
        glDeleteBuffers(1, &this->vbos[i].handle);
    }
    this->vbos.clear();

    // Declare data buffers ---------------------------------------------------

    // Init vbos
    SDFVBO newVBO;
    newVBO.handle = 0; // Default init

    // VBO for position data
    newVBO.name  = "inPos";
    newVBO.index = (GLuint)VBOAttrib::POSITION;
    newVBO.dim   = 3;
    this->vbos.push_back(newVBO);

    // VBO for texture data
    newVBO.name  = "inTexCoord";
    newVBO.index = (GLuint)VBOAttrib::TEXTURE;
    newVBO.dim   = 2;
    this->vbos.push_back(newVBO);

    // VBO for texture data
    newVBO.name  = "inColor";
    newVBO.index = (GLuint)VBOAttrib::COLOR;
    newVBO.dim   = 4;
    this->vbos.push_back(newVBO);

    // ------------------------------------------------------------------------

    // Create Vertex Array Object 
    glGenVertexArrays(1, &this->vaoHandle);
    glBindVertexArray(this->vaoHandle);

    for (unsigned int i = 0; i < (unsigned int)this->vbos.size(); i++) {
        glGenBuffers(1, &this->vbos[i].handle);
        glBindBuffer(GL_ARRAY_BUFFER, this->vbos[i].handle);
        // Create empty buffer
        glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);
        // Bind buffer to vertex attribute
        glEnableVertexAttribArray(this->vbos[i].index); 
        glVertexAttribPointer(this->vbos[i].index, this->vbos[i].dim, GL_FLOAT, GL_FALSE, 0, (GLubyte *)nullptr);
    }
   
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    for (unsigned int i = 0; i < (unsigned int)this->vbos.size(); i++) {
        glDisableVertexAttribArray(this->vbos[i].index);
    }

    return true;
}


/*
* SDFFont::loadFontInfo
*
* Bitmap Font file format: http://www.angelcode.com/products/fnont/doc/file_format.html
*
*/
bool SDFFont::loadFontInfo(vislib::StringA filename) {

    // Reset font info
    for (unsigned int i = 0; i < this->glyphs.size(); i++) {
        ARY_SAFE_DELETE(this->glyphs[i].kerns);
    }
    for (size_t i = 0; i < this->glyphIdcs.size(); i++) {
        this->glyphIdcs[i] = nullptr;
    }

    this->glyphs.clear();
    this->glyphIdcs.clear();

    // Load file
    vislib::sys::ASCIIFileBuffer file;
    if (!file.LoadFile(filename)) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadfontCharacters] Could not load file as ascii buffer: \"%s\". \n", filename.PeekBuffer());
        return false;
    }

    float texWidth    = 0.0f;
    float texHeight   = 0.0f;
    float lineHeight  = 0.0f;
    
    int    idx;
    float  width;
    float  height;

    unsigned int maxId = 0;;
    std::vector<SDFGlyphKerning> tmpKerns;
    tmpKerns.clear();

    // Read info file line by line
    size_t lineCnt = 0;
    vislib::StringA line;
    while (lineCnt < file.Count()) {
        line = static_cast<vislib::StringA>(file.Line(lineCnt));
        // (1) Parse common info line
        if (line.StartsWith("common ")) { 

            idx = line.Find("scaleW=", 0);
            texWidth = (float)std::atof(line.Substring(idx + 7, 4));

            idx = line.Find("scaleH=", 0);
            texHeight = (float)std::atof(line.Substring(idx + 7, 4));

            idx = line.Find("lineHeight=", 0);
            lineHeight = (float)std::atof(line.Substring(idx + 11, 4));
        }
        // (2) Parse character info
        else if (line.StartsWith("char ")) { 
            SDFGlyphInfo newChar;

            idx = line.Find("id=", 0);
            newChar.id = (unsigned int)std::atoi(line.Substring(idx + 3, 5)); 

            if (maxId < newChar.id) {
                maxId = newChar.id;
            }

            idx = line.Find("x=", 0);
            newChar.texX0 = (float)std::atof(line.Substring(idx + 2, 4)) / texWidth;

            idx = line.Find("y=", 0);
            newChar.texY0 = (float)std::atof(line.Substring(idx + 2, 4)) / texHeight;

            idx = line.Find("width=", 0);
            width = (float)std::atof(line.Substring(idx + 6, 4));

            idx = line.Find("height=", 0);
            height = (float)std::atof(line.Substring(idx + 7, 4));

            newChar.width  = width / lineHeight;
            newChar.height = height / lineHeight;

            idx = line.Find("xoffset=", 0);
            newChar.xoffset = (float)std::atof(line.Substring(idx + 8, 4)) / lineHeight;

            idx = line.Find("yoffset=", 0);
            newChar.yoffset  = (float)std::atof(line.Substring(idx + 8, 4)) / lineHeight;

            idx = line.Find("xadvance=", 0);
            newChar.xadvance = (float)std::atof(line.Substring(idx + 9, 4)) / lineHeight;

            newChar.kernCnt = 0;
            newChar.kerns   = nullptr;

            newChar.texX1 = newChar.texX0 + width / texWidth;
            newChar.texY1 = newChar.texY0 + height / texHeight;

            this->glyphs.push_back(newChar);
        }
        // (3) Parse kerning info
        else if (line.StartsWith("kerning ")) { 

            SDFGlyphKerning newKern;

            idx = line.Find("first=", 0);
            newKern.previous = (unsigned int)std::atoi(line.Substring(idx+6, 4));

            idx = line.Find("second=", 0);
            newKern.current = (unsigned int)std::atoi(line.Substring(idx+7, 4));

            idx = line.Find("amount=", 0);
            newKern.xamount = (float)std::atof(line.Substring(idx+7, 4)) / lineHeight;

            tmpKerns.push_back(newKern);
        }
        // Proceed with next line ...
        lineCnt++;
    }
    //Clear ascii file buffer
    file.Clear();

    // Init index pointer array 
    maxId++;
    for (unsigned int i = 0; i < maxId; i++) {
        this->glyphIdcs.push_back(nullptr);
    }
    // Set pointers to available glyph info
    for (unsigned int i = 0; i < (unsigned int)this->glyphs.size(); i++) {
        // Filling character index array --------------------------------------
        if (this->glyphs[i].id > (unsigned int)this->glyphIdcs.size()) {
            vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadFontInfo] Character is out of range: \"%i\". \n", this->glyphs[i].id);
            return false;
        }
        this->glyphIdcs[this->glyphs[i].id] = &this->glyphs[i];
        // Assigning glyphKrns -------------------------------------------------
        // Get count of glyphKrns for current glyph
        for (unsigned int j = 0; j < (unsigned int)tmpKerns.size(); j++) {
            if (this->glyphs[i].id == tmpKerns[j].current) {
                this->glyphs[i].kernCnt++;
            }
        }
        unsigned int c = 0;
        this->glyphs[i].kerns = new SDFGlyphKerning[this->glyphs[i].kernCnt];
        for (unsigned int j = 0; j < (unsigned int)tmpKerns.size(); j++) {
            if (this->glyphs[i].id == tmpKerns[j].current) {
                this->glyphs[i].kerns[c] = tmpKerns[j];
                c++;
            }
        }
    }

    return true;
}


/*
* SDFFont::loadTexture
*/
bool SDFFont::loadFontTexture(vislib::StringA filename) {

    // Reset font texture
    this->texture.Release();

    static vislib::graphics::BitmapImage img;
    static sg::graphics::PngBitmapCodec  pbc;
    pbc.Image() = &img;
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    BYTE *buf = nullptr;
    size_t size = 0;

    if ((size = this->loadFile(filename, &buf)) <= 0) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadTexture] Could not find texture: \"%s\". \n", filename.PeekBuffer());
        ARY_SAFE_DELETE(buf);
        return false;
    }

    if (pbc.Load((void*)buf, size)) {

        img.Convert(vislib::graphics::BitmapImage::TemplateByteRGBA);

        if (this->texture.Create(img.Width(), img.Height(), false, img.PeekDataAs<BYTE>(), GL_RGBA) != GL_NO_ERROR) {
            vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadTexture] Could not load texture: \"%s\". \n", filename.PeekBuffer());
            ARY_SAFE_DELETE(buf);
            return false;
        }

        this->texture.Bind();
        this->texture.SetFilter(GL_LINEAR, GL_LINEAR);
        this->texture.SetWrap(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);
        ARY_SAFE_DELETE(buf);
    }
    else {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadTexture] Could not read texture: \"%s\". \n", filename.PeekBuffer());
        ARY_SAFE_DELETE(buf);
        return false;
    }

    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    return true;
}


/*
* SDFFont::loadFontShader
*/
bool SDFFont::loadFontShader(megamol::core::CoreInstance *core) {

    if (core == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadFontShader] Pointer to MegaMol CoreInstance is NULL. \n");
        return false;
    }

    vislib::graphics::gl::ShaderSource vs, fs;

    StringA shaderNamespace  = "sdffont";

    StringA vertShaderName = shaderNamespace;
    vertShaderName.Append("::vertex");

    StringA fragShaderName = shaderNamespace;
    fragShaderName.Append("::fragment");

    StringA versSnipName = vertShaderName;
    versSnipName.Append("::version");

    StringA mainSnipName = vertShaderName;
    mainSnipName.Append("::main");

    // Create array of shaders to loop over
    vislib::graphics::gl::GLSLShader *shaderPtr[2];
    shaderPtr[0] = &this->shader;
    shaderPtr[1] = &this->shadervertcol;

    // Defining used vbo attributes for each shader
    std::vector<unsigned int> attribLoc[2];
    for (unsigned int i = 0; i < (unsigned int)this->vbos.size(); i++) {
        GLuint index = this->vbos[i].index;
        if (index == (GLuint)VBOAttrib::POSITION) {
            attribLoc[0].push_back(i);
            attribLoc[1].push_back(i);
        }
        else if (index == (GLuint)VBOAttrib::TEXTURE) {
            attribLoc[0].push_back(i);
            attribLoc[1].push_back(i);
        }
        else if (index == (GLuint)VBOAttrib::COLOR) {
            attribLoc[1].push_back(i);
        }
    }

    // Loop over all shaders in array
    for (unsigned int i = 0; i < 2; ++i) {
        // Reset shader
        shaderPtr[i]->Release();

        // Load shader
        try {

            vs.Clear();
            if (!core->ShaderSourceFactory().MakeShaderSource(vertShaderName, vs)) {
                vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadShader] Unable to make shader source for vertex shader. \n");
                return false;
            }

            fs.Clear();
            if (!core->ShaderSourceFactory().MakeShaderSource(fragShaderName, fs)) {
                vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadShader] Unable to make shader source for fragment shader. \n");
                return false;
            }

            // Choose right color snippet for current shader
            StringA colSnipName = vertShaderName;
            if (&this->shader == shaderPtr[i]) {
                colSnipName.Append("::globalColor");
            }
            else if (&this->shadervertcol == shaderPtr[i]) {
                colSnipName.Append("::vertexColor");
            }

            // Putting vertex shader together
            vs.Clear();
            vs.Append(core->ShaderSourceFactory().MakeShaderSnippet(versSnipName)); /// 1)
            vs.Append(core->ShaderSourceFactory().MakeShaderSnippet(colSnipName));  /// 2)
            vs.Append(core->ShaderSourceFactory().MakeShaderSnippet(mainSnipName)); /// 3)
            //VLTRACE(vislib::Trace::LEVEL_VL_INFO, "\n----- Vertex shader using '%s': ----- \n%s\n", colSnipName.PeekBuffer(), vs.WholeCode().PeekBuffer());

            // Compiling shaders
            if (!shaderPtr[i]->Compile(vs.Code(), vs.Count(), fs.Code(), fs.Count())) {
                vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadShader] Unable to compile \"%s\"-shader: Unknown error. \n", shaderNamespace.PeekBuffer());
                return false;
            }

            // Bind vertex shader attributes (before linking shaders!)
            for (unsigned int j = 0; j < attribLoc[i].size(); j++) {
                glBindAttribLocation(shaderPtr[i]->ProgramHandle(), this->vbos[attribLoc[i][j]].index, this->vbos[attribLoc[i][j]].name.PeekBuffer());
            }

            // Linking shaders
            if (!shaderPtr[i]->Link()) {
                vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadShader] Unable to link \"%s\"-shader: Unknown error. \n", shaderNamespace.PeekBuffer());
                return false;
            }
        }
        catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
            vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadShader] Unable to compile \"%s\"-shader (@%s): %s. \n", shaderNamespace.PeekBuffer(),
                vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()), ce.GetMsgA());
            return false;
        }
        catch (vislib::Exception e) {
            vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadShader] Unable to compile \"%s\"-shader: %s. \n", shaderNamespace.PeekBuffer(), e.GetMsgA());
            return false;
        }
        catch (...) {
            vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadShader] Unable to compile \"%s\"-shader: Unknown exception. \n", shaderNamespace.PeekBuffer());
            return false;
        }
    }

    return true;
}


/*
* SDFFont::loadFile
*/
size_t SDFFont::loadFile(vislib::StringA filename, BYTE **outData) {

    // Reset out data
    *outData = nullptr;

    vislib::StringW name = static_cast<vislib::StringW>(filename);
    if (name.IsEmpty()) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadFile] Unable to load file: No name given. \n");
        return false;
    }
    if (!vislib::sys::File::Exists(name)) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadFile] Unable to load not existing file: \"%s\". \n", filename.PeekBuffer());
        return false;
    }

    size_t size = static_cast<size_t>(vislib::sys::File::GetSize(name));
    if (size < 1) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadFile] Unable to load empty file: \"%s\". \n", filename.PeekBuffer());
        return false;
    }

    vislib::sys::FastFile f;
    if (!f.Open(name, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadFile] Unable to open file: \"%s\". \n", filename.PeekBuffer());
        return false;
    }

    *outData = new BYTE[size];
    size_t num = static_cast<size_t>(f.Read(*outData, size));
    if (num != size) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadFile] Unable to read whole file: \"%s\". \n", filename.PeekBuffer());
        ARY_SAFE_DELETE(*outData);
        return false;
    }

    return num;
}
