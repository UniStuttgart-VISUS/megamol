/**
 * MegaMol
 * Copyright (c) 2006, MegaMol Dev Team
 * All rights reserved.
 */
// This implementation is based on "vislib/graphics/OutlinetFont.h"

#include "mmcore_gl/utility/SDFFont.h"

#include <fstream>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>

#include "mmcore/utility/ResourceWrapper.h"
#include "mmcore_gl/utility/ShaderFactory.h"

using namespace megamol::core::utility;
using megamol::core::utility::log::Log;


SDFFont::SDFFont(PresetFontName fn) : SDFFont(fn, 1.0f, RenderMode::RENDERMODE_FILL, false) {}
SDFFont::SDFFont(PresetFontName fn, SDFFont::RenderMode render_mode) : SDFFont(fn, 1.0f, render_mode, false) {}
SDFFont::SDFFont(PresetFontName fn, float size) : SDFFont(fn, size, RenderMode::RENDERMODE_FILL, false) {}
SDFFont::SDFFont(PresetFontName fn, bool flipY) : SDFFont(fn, 1.0f, RenderMode::RENDERMODE_FILL, flipY) {}
SDFFont::SDFFont(PresetFontName fn, SDFFont::RenderMode render_mode, bool flipY)
        : SDFFont(fn, 1.0f, render_mode, flipY) {}
SDFFont::SDFFont(PresetFontName fn, float size, bool flipY) : SDFFont(fn, size, RenderMode::RENDERMODE_FILL, flipY) {}
SDFFont::SDFFont(PresetFontName fn, float size, SDFFont::RenderMode render_mode)
        : SDFFont(fn, size, render_mode, false) {}
SDFFont::SDFFont(PresetFontName fn, float size, SDFFont::RenderMode render_mode, bool flipY)
        : initialised(false)
        , fontFileName(this->presetFontNameToString(fn))
        , renderMode(render_mode)
        , billboardMode(false)
        , batchDrawMode(false)
        , smoothMode(true)
        , rotation()
        , outlineColor(0.0f, 0.0f, 0.0f)
        , outlineThickness(0.1f)
        , globalSize(size)
        , globalFlipY(flipY)
        , texture(nullptr)
        , vaoHandle(GLuint(0))
        , vbos()
        , posBatchCache()
        , texBatchCache()
        , colBatchCache()
        , glyphs()
        , glyphIdcs()
        , glyphKrns() {
    this->SetSize(size);
    this->SetFlipY(flipY);
}

SDFFont::SDFFont(std::string fn) : SDFFont(fn, 1.0f, RenderMode::RENDERMODE_FILL, false) {}
SDFFont::SDFFont(std::string fn, SDFFont::RenderMode render_mode) : SDFFont(fn, 1.0f, render_mode, false) {}
SDFFont::SDFFont(std::string fn, float size) : SDFFont(fn, size, RenderMode::RENDERMODE_FILL, false) {}
SDFFont::SDFFont(std::string fn, bool flipY) : SDFFont(fn, 1.0f, RenderMode::RENDERMODE_FILL, flipY) {}
SDFFont::SDFFont(std::string fn, SDFFont::RenderMode render_mode, bool flipY) : SDFFont(fn, 1.0f, render_mode, flipY) {}
SDFFont::SDFFont(std::string fn, float size, bool flipY) : SDFFont(fn, size, RenderMode::RENDERMODE_FILL, flipY) {}
SDFFont::SDFFont(std::string fn, float size, SDFFont::RenderMode render_mode) : SDFFont(fn, size, render_mode, false) {}
SDFFont::SDFFont(std::string fn, float size, SDFFont::RenderMode render_mode, bool flipY)
        : initialised(false)
        , fontFileName(fn)
        , renderMode(render_mode)
        , billboardMode(false)
        , batchDrawMode(false)
        , smoothMode(true)
        , rotation()
        , outlineColor(0.0f, 0.0f, 0.0f)
        , outlineThickness(0.1f)
        , globalSize(size)
        , globalFlipY(flipY)
        , texture(nullptr)
        , vaoHandle(GLuint(0))
        , vbos()
        , posBatchCache()
        , texBatchCache()
        , colBatchCache()
        , glyphs()
        , glyphIdcs()
        , glyphKrns() {
    this->SetSize(size);
    this->SetFlipY(flipY);
}

megamol::core::utility::SDFFont::SDFFont(const SDFFont& src)
        : initialised(src.initialised)
        , fontFileName(src.fontFileName)
        , renderMode(src.renderMode)
        , billboardMode(src.billboardMode)
        , batchDrawMode(src.batchDrawMode)
        , smoothMode(src.smoothMode)
        , rotation(src.rotation)
        , outlineColor(src.outlineColor)
        , outlineThickness(src.outlineThickness)
        , globalSize(src.globalSize)
        , globalFlipY(src.globalFlipY)
        , shaderglobcol(src.shaderglobcol)
        , shadervertcol(src.shadervertcol)
        , texture(src.texture)
        , vaoHandle(src.vaoHandle)
        , vbos(src.vbos)
        , posBatchCache(src.posBatchCache)
        , texBatchCache(src.texBatchCache)
        , colBatchCache(src.colBatchCache)
        , glyphs(src.glyphs)
        , glyphIdcs(src.glyphIdcs)
        , glyphKrns(src.glyphKrns) {}


SDFFont::~SDFFont(void) {

    if (this->initialised) {
        this->Deinitialise();
    }
}


bool SDFFont::Initialise(
    megamol::core::CoreInstance* core_instance_ptr, megamol::frontend_resources::RuntimeConfig const& runtimeConf) {

    if (!this->initialised) {
        this->initialised = this->loadFont(core_instance_ptr, runtimeConf);
    }
    return this->initialised;
}


unsigned int SDFFont::BlockLines(float maxWidth, float size, const char* txt) const {

    return this->lineCount(this->buildGlyphRun(txt, maxWidth / size), true);
}


void SDFFont::DrawString(const glm::mat4& mvm, const glm::mat4& pm, const float col[4], float x, float y, float size,
    bool flipY, const char* txt, SDFFont::Alignment align) const {

    if (!this->initialised || (this->renderMode == RenderMode::RENDERMODE_NONE))
        return;

    int* run = this->buildGlyphRun(txt, FLT_MAX);

    if ((align == ALIGN_CENTER_MIDDLE) || (align == ALIGN_LEFT_MIDDLE) || (align == ALIGN_RIGHT_MIDDLE)) {
        y += static_cast<float>(this->lineCount(run, false)) * 0.5f * size * (flipY ? -1.0f : 1.0f);

    } else if ((align == ALIGN_CENTER_BOTTOM) || (align == ALIGN_LEFT_BOTTOM) || (align == ALIGN_RIGHT_BOTTOM)) {
        y += static_cast<float>(this->lineCount(run, false)) * size * (flipY ? -1.0f : 1.0f);
    }

    this->drawGlyphs(mvm, pm, col, run, x, y, 0.0f, size, flipY, align);

    ARY_SAFE_DELETE(run);
}


void SDFFont::DrawString(const glm::mat4& mvm, const glm::mat4& pm, const float col[4], float x, float y, float w,
    float h, float size, bool flipY, const char* txt, SDFFont::Alignment align) const {

    if (!this->initialised || (this->renderMode == RenderMode::RENDERMODE_NONE))
        return;

    int* run = this->buildGlyphRun(txt, w / size);

    if (flipY)
        y += h;

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

    this->drawGlyphs(mvm, pm, col, run, x, y, 0.0f, size, flipY, align);

    ARY_SAFE_DELETE(run);
}


void SDFFont::DrawString(const glm::mat4& mvm, const glm::mat4& pm, const float col[4], float x, float y, float z,
    float w, float h, float size, bool flipY, const char* txt, SDFFont::Alignment align) const {

    if (!this->initialised || (this->renderMode == RenderMode::RENDERMODE_NONE))
        return;

    int* run = this->buildGlyphRun(txt, w / size);

    if (flipY)
        y += h;

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

    this->drawGlyphs(mvm, pm, col, run, x, y, z, size, flipY, align);

    ARY_SAFE_DELETE(run);
}


void SDFFont::DrawString(const glm::mat4& mvm, const glm::mat4& pm, const float col[4], float x, float y, float z,
    float size, bool flipY, const char* txt, Alignment align) const {

    if (!this->initialised || (this->renderMode == RenderMode::RENDERMODE_NONE))
        return;

    int* run = this->buildGlyphRun(txt, FLT_MAX);

    if ((align == ALIGN_CENTER_MIDDLE) || (align == ALIGN_LEFT_MIDDLE) || (align == ALIGN_RIGHT_MIDDLE)) {
        y += static_cast<float>(this->lineCount(run, false)) * 0.5f * size * (flipY ? -1.0f : 1.0f);
    } else if ((align == ALIGN_CENTER_BOTTOM) || (align == ALIGN_LEFT_BOTTOM) || (align == ALIGN_RIGHT_BOTTOM)) {
        y += static_cast<float>(this->lineCount(run, false)) * size * (flipY ? -1.0f : 1.0f);
    }

    this->drawGlyphs(mvm, pm, col, run, x, y, z, size, flipY, align);

    ARY_SAFE_DELETE(run);
}


float SDFFont::LineWidth(float size, const char* txt) const {

    int* run = this->buildGlyphRun(txt, FLT_MAX);
    int* i = run;
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


void SDFFont::BatchDrawString(const glm::mat4& mvm, const glm::mat4& pm, const float col[4]) const {

    if (this->posBatchCache.empty())
        return;

    // Bind glyph data in batch cache
    for (unsigned int i = 0; i < (unsigned int)this->vbos.size(); i++) {
        glBindBuffer(GL_ARRAY_BUFFER, this->vbos[i].handle);
        if (this->vbos[i].index == (GLuint)VBOAttrib::POSITION) {
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)this->posBatchCache.size() * sizeof(GLfloat),
                &this->posBatchCache.front(), GL_STATIC_DRAW);
        } else if (this->vbos[i].index == (GLuint)VBOAttrib::TEXTURE) {
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)this->texBatchCache.size() * sizeof(GLfloat),
                &this->texBatchCache.front(), GL_STATIC_DRAW);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    // Draw batch cache
    unsigned int glyphCnt =
        ((unsigned int)this->posBatchCache.size() / 18); // 18 = 2 Triangles * 3 Vertices * 3 Coordinates
    this->render(mvm, pm, glyphCnt, &col);
}


void SDFFont::BatchDrawString(const glm::mat4& mvm, const glm::mat4& pm) const {

    if (this->posBatchCache.empty())
        return;

    // Bind glyph data in batch cache
    for (unsigned int i = 0; i < (unsigned int)this->vbos.size(); i++) {
        glBindBuffer(GL_ARRAY_BUFFER, this->vbos[i].handle);
        if (this->vbos[i].index == (GLuint)VBOAttrib::POSITION) {
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)this->posBatchCache.size() * sizeof(GLfloat),
                &this->posBatchCache.front(), GL_STATIC_DRAW);
        } else if (this->vbos[i].index == (GLuint)VBOAttrib::TEXTURE) {
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)this->texBatchCache.size() * sizeof(GLfloat),
                &this->texBatchCache.front(), GL_STATIC_DRAW);
        } else if (this->vbos[i].index == (GLuint)VBOAttrib::COLOR) {
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)this->colBatchCache.size() * sizeof(GLfloat),
                &this->colBatchCache.front(), GL_STATIC_DRAW);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    // Draw batch cache
    unsigned int glyphCnt =
        ((unsigned int)this->posBatchCache.size() / 18); // 18 = 2 Triangles * 3 Vertices * 3 Coordinates
    this->render(mvm, pm, glyphCnt, nullptr);
}


void SDFFont::Deinitialise(void) {

    this->ClearBatchDrawCache();
    this->texture.reset();
    this->shaderglobcol.reset();
    this->shadervertcol.reset();

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
            delete[] this->glyphs[i].kerns;
        }
    }

    // Set glyph indey pointer to null
    for (size_t i = 0; i < this->glyphIdcs.size(); i++) {
        this->glyphIdcs[i] = nullptr;
    }

    this->initialised = false;
}


int SDFFont::lineCount(int* run, bool deleterun) const {
    if ((run == nullptr) || (run[0] == 0))
        return 0;
    int i = 1;
    for (int j = 0; run[j] != 0; j++) {
        if (run[j] < 0)
            i++;
    }
    if (deleterun)
        ARY_SAFE_DELETE(run);

    return i;
}


float SDFFont::lineWidth(int*& run, bool iterate) const {

    int* i = run;
    float len = 0.0f;
    while (*i != 0) {
        len += this->glyphIdcs[((*i) < 0) ? (-1 - (*i)) : ((*i) - 1)]->xadvance; // No check -> requires valid run!
        i++;
        if (*i < 0)
            break;
    }
    if (iterate)
        run = i;

    return len;
}


int* SDFFont::buildGlyphRun(const char* txt, float maxWidth) const {

    vislib::StringA txtutf8(txt);
    size_t txtlen = static_cast<size_t>(vislib::CharTraitsA::SafeStringLength(txtutf8));
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

    int* glyphrun = new int[txtlen + 1];
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
        } else { // ... else if byte >= 128 => UTF8-Byte: 1XXXXXXX
            // Supporting UTF8 for up to 3 bytes:
            if (byte >= (unsigned char)(0b11100000)) { // => >224 - 1110XXXX -> start 3-Byte UTF8, 2 bytes are following
                folBytes = 2;
                idx = (unsigned int)(byte & (unsigned char)(0b00001111)); // => consider only last 4 bits
                idx = (idx << 12);                                        // => 2*6 Bits are following
                continue;
            } else if (byte >=
                       (unsigned char)(0b11000000)) { // => >192 - 110XXXXX -> start 2-Byte UTF8, 1 byte is following
                folBytes = 1;
                idx = (unsigned int)(byte & (unsigned char)(0b00011111)); // => consider only last 5 bits
                idx = (idx << 6);                                         // => 1*6 Bits are following
                continue;
            } else if (byte >= (unsigned char)(0b10000000)) { // => >128 - 10XXXXXX -> "following" 1-2 bytes
                folBytes--;
                tmpIdx = (unsigned int)(byte & (unsigned char)(0b00111111)); // => consider only last 6 bits
                idx = (idx | (tmpIdx << (folBytes *
                                         6))); // => shift tmpIdx depending on following byte and 'merge' (|) with idx
                if (folBytes > 0)
                    continue; // => else idx is complete
            }
        }
        // Check if glyph info is available
        if (idx > (unsigned int)this->glyphIdcs.size()) {
            /// megamol::core::utility::log::Log::DefaultLog.WriteWarn("[SDFFont] Glyph index greater than available: \"%i\" > max. Index = \"%i\".\n", idx, this->idxCnt);
            continue;
        }
        if (this->glyphIdcs[idx] == nullptr) {
            /// megamol::core::utility::log::Log::DefaultLog.WriteWarn("[SDFFont] Glyph info not available for: \"%i\".\n", idx);
            continue;
        }
        // --------------------------------------------------------------------

        // add glyph to run
        if (txtutf8[i] == ' ') { // the only special white-space
            glyphrun[pos++] = static_cast<int>(1 + idx);
            lineLength += this->glyphIdcs[idx]->xadvance;
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
            lineLength = this->glyphIdcs[idx]->xadvance;
        } else {
            blackspace = true;
            glyphrun[pos++] = static_cast<int>(1 + idx);
            lineLength += this->glyphIdcs[idx]->xadvance;
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
                    lineLength = this->glyphIdcs[idx]->xadvance;
                }
            }
        }
    }

    return glyphrun;
}


void SDFFont::drawGlyphs(const glm::mat4& mvm, const glm::mat4& pm, const float col[4], int* run, float x, float y,
    float z, float size, bool flipY, Alignment align) const {

    // Data buffers
    unsigned glyphCnt = 0;
    int* tmpRun = run;
    while ((*tmpRun) != 0) {
        tmpRun++;
        glyphCnt++;
    }
    unsigned int posCnt = glyphCnt * 18; // 2 Triangles * 3 Vertices * 3 Coordinates
    unsigned int texCnt = glyphCnt * 12; // 2 Triangles * 3 Vertices * 2 Coordinates

    float* posData = new float[posCnt];
    float* texData = new float[texCnt];

    float gx = x;
    float gy = y;
    float gz = z;

    float sy = (flipY) ? (size) : (-size);
    float kern = 0.0f;
    unsigned int lastGlyphId = 0;

    // Billboard stuff
    // -> Setting fixed rotation point depending on alignment.
    glm::vec4 billboardRotPoint(0.0f, 0.0f, 0.0f, 1.0f);
    if (this->billboardMode) {
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
        billboardRotPoint = glm::vec4(gx, gy + deltaY, gz, 1.0f);

        // Apply model view matrix ONLY to rotation point ...
        billboardRotPoint = mvm * billboardRotPoint;
        billboardRotPoint.y = (billboardRotPoint.y - deltaY);
        gx = billboardRotPoint.x;
        gy = billboardRotPoint.y;
        gz = billboardRotPoint.z;
    }

    // Adjustment for first line
    if ((align == ALIGN_CENTER_BOTTOM) || (align == ALIGN_CENTER_MIDDLE) || (align == ALIGN_CENTER_TOP)) {
        gx -= this->lineWidth(run, false) * size * 0.5f;
    } else if ((align == ALIGN_RIGHT_BOTTOM) || (align == ALIGN_RIGHT_MIDDLE) || (align == ALIGN_RIGHT_TOP)) {
        gx -= this->lineWidth(run, false) * size;
    }

    unsigned int glyphIter = 0;
    while ((*run) != 0) {

        // Run contains only available character indices (=> glyph is always != nullptr)
        SDFGlyphInfo* glyph = this->glyphIdcs[((*run) < 0) ? (-1 - (*run)) : ((*run) - 1)];

        // Adjust positions if character indicates a new line
        if ((*run) < 0) {
            gx = (this->billboardMode) ? (billboardRotPoint.x) : (x);
            if ((align == ALIGN_CENTER_BOTTOM) || (align == ALIGN_CENTER_MIDDLE) || (align == ALIGN_CENTER_TOP)) {
                gx -= this->lineWidth(run, false) * size * 0.5f;
            } else if ((align == ALIGN_RIGHT_BOTTOM) || (align == ALIGN_RIGHT_MIDDLE) || (align == ALIGN_RIGHT_TOP)) {
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

        glm::vec4 p0(tmpP01x, tmpP03y, gz, 0.0f);
        glm::vec4 p1(tmpP01x, tmpP12y, gz, 0.0f);
        glm::vec4 p2(tmpP23x, tmpP12y, gz, 0.0f);
        glm::vec4 p3(tmpP23x, tmpP03y, gz, 0.0f);

        /// Apply rotation
        glm::mat4 rotation_matrix = glm::toMat4(this->rotation);
        p0 = rotation_matrix * p0;
        p1 = rotation_matrix * p1;
        p2 = rotation_matrix * p2;
        p3 = rotation_matrix * p3;

        // Set position data:
        posData[glyphIter * 18 + 0] = p0.x;
        posData[glyphIter * 18 + 1] = p0.y;
        posData[glyphIter * 18 + 2] = p0.z;

        posData[glyphIter * 18 + 3] = p1.x;
        posData[glyphIter * 18 + 4] = p1.y;
        posData[glyphIter * 18 + 5] = p1.z;

        posData[glyphIter * 18 + 6] = p2.x;
        posData[glyphIter * 18 + 7] = p2.y;
        posData[glyphIter * 18 + 8] = p2.z;

        posData[glyphIter * 18 + 9] = p0.x;
        posData[glyphIter * 18 + 10] = p0.y;
        posData[glyphIter * 18 + 11] = p0.z;

        posData[glyphIter * 18 + 12] = p2.x;
        posData[glyphIter * 18 + 13] = p2.y;
        posData[glyphIter * 18 + 14] = p2.z;

        posData[glyphIter * 18 + 15] = p3.x;
        posData[glyphIter * 18 + 16] = p3.y;
        posData[glyphIter * 18 + 17] = p3.z;

        // Change rotation of quad positions for flipped y axis from CCW to CW.
        if (flipY) {
            posData[glyphIter * 18 + 3] = p2.x;  // p2-x
            posData[glyphIter * 18 + 6] = p1.x;  // p1-x
            posData[glyphIter * 18 + 13] = p3.y; // p3-y
            posData[glyphIter * 18 + 16] = p2.y; // p2-y
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

        texData[glyphIter * 12 + 8] = glyph->texX1; // t2-x
        texData[glyphIter * 12 + 9] = glyph->texY1; // t2-y

        texData[glyphIter * 12 + 10] = glyph->texX1; // t3-x
        texData[glyphIter * 12 + 11] = glyph->texY0; // t3-y

        // Change rotation of texture coord for flipped y axis from CCW to CW.
        if (flipY) {
            texData[glyphIter * 12 + 2] = glyph->texX1;  // t2-x
            texData[glyphIter * 12 + 4] = glyph->texX0;  // t1-x
            texData[glyphIter * 12 + 9] = glyph->texY0;  // t3-y
            texData[glyphIter * 12 + 11] = glyph->texY1; // t2-y
        }

        // Update info for next character
        glyphIter++;
        gx += ((glyph->xadvance + kern) * size);
        lastGlyphId = glyph->id;
        run++;
    }

    if (this->batchDrawMode) {
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
    } else {
        // ... or draw glyphs instantly.
        for (unsigned int i = 0; i < (unsigned int)this->vbos.size(); i++) {
            glBindBuffer(GL_ARRAY_BUFFER, this->vbos[i].handle);
            if (this->vbos[i].index == (GLuint)VBOAttrib::POSITION) {
                glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)posCnt * sizeof(GLfloat), posData, GL_STATIC_DRAW);
            } else if (this->vbos[i].index == (GLuint)VBOAttrib::TEXTURE) {
                glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)texCnt * sizeof(GLfloat), texData, GL_STATIC_DRAW);
            }
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }

        // Draw data buffers
        this->render(mvm, pm, glyphCnt, &col);
    }

    ARY_SAFE_DELETE(posData);
    ARY_SAFE_DELETE(texData);
}


void SDFFont::render(
    const glm::mat4& mvm, const glm::mat4& pm, unsigned int glyph_count, const float* color_ptr[4]) const {

    // Check texture
    if (this->texture->getName() == 0) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[SDFFont] Texture is not valid. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }

    // Check if per vertex color should be used
    const std::shared_ptr<glowl::GLSLProgram>& usedShader = (color_ptr == nullptr) ? shadervertcol : shaderglobcol;

    // Check shaders
    if (usedShader == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[SDFFont] Shader handle is not valid. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }

    glm::mat4 shader_matrix;
    if (this->billboardMode) {
        // Only projection matrix has to be applied for billboard
        shader_matrix = pm;
    } else {
        shader_matrix = pm * mvm;
    }

    // Set blending
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    // dFdx()/dFdx() in fragment shader:
    glHint(GL_FRAGMENT_SHADER_DERIVATIVE_HINT, GL_NICEST);

    glEnable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE0);
    this->texture->bindTexture();

    glBindVertexArray(this->vaoHandle);

    usedShader->use();

    // Vertex shader uniforms
    usedShader->setUniform("mvpMat", shader_matrix);
    // Set global color, if given
    if (color_ptr != nullptr) {
        glUniform4fv(usedShader->getUniformLocation("inColor"), 1, *color_ptr);
    }

    // Fragment shader uniforms
    usedShader->setUniform("fontTex", 0);
    usedShader->setUniform("renderMode", static_cast<int>(this->renderMode));
    if (this->renderMode == RenderMode::RENDERMODE_OUTLINE) {
        usedShader->setUniform("outlineColor", this->outlineColor);
        usedShader->setUniform("outlineThickness", this->outlineThickness);
    }
    usedShader->setUniform("smoothMode", static_cast<int>(this->smoothMode));

    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)glyph_count * 6); // 2 triangles per glyph -> 6 vertices

    glUseProgram(0); // instead of usedShader->Disable() => because draw() is CONST
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_BLEND);
}


bool SDFFont::loadFont(
    megamol::core::CoreInstance* core_instance_ptr, megamol::frontend_resources::RuntimeConfig const& runtimeConf) {

    this->ResetRotation();
    this->SetBatchDrawMode(false);
    this->ClearBatchDrawCache();
    this->SetBillboardMode(false);
    this->SetSmoothMode(true);

    if (core_instance_ptr == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[SDFFont] Pointer to MegaMol CoreInstance is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
            __LINE__);
        return false;
    }

    // (1) Load buffers --------------------------------------------------------
    if (!this->loadFontBuffers()) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[SDFFont] Failed to load buffers. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // (2) Load font information -----------------------------------------------
    std::string infoFile = this->fontFileName;
    infoFile.append(".fnt");
    vislib::StringW info_filename =
        ResourceWrapper::getFileName(core_instance_ptr->Configuration(), vislib::StringA(infoFile.c_str()))
            .PeekBuffer();
    if (!this->loadFontInfo(info_filename)) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[SDFFont] Failed to load font info file: \"%s\". [%s, %s, line %d]\n", infoFile.c_str(), __FILE__,
            __FUNCTION__, __LINE__);
        return false;
    }

    // (3) Load font texture --------------------------------------------------------
    std::string textureFile = this->fontFileName;
    textureFile.append(".png");
    auto texture_filename = static_cast<std::wstring>(
        ResourceWrapper::getFileName(core_instance_ptr->Configuration(), vislib::StringA(textureFile.c_str()))
            .PeekBuffer());
    if (!megamol::core_gl::utility::RenderUtils::LoadTextureFromFile(
            this->texture, megamol::core::utility::WChar2Utf8String(texture_filename))) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[SDFFont] Failed to load font texture: \"%s\". [%s, %s, line %d]\n", textureFile.c_str(), __FILE__,
            __FUNCTION__, __LINE__);
        return false;
    }

    // (4) Load shaders --------------------------------------------------------
    if (!this->loadFontShader(runtimeConf)) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[SDFFont] Failed to load font shaders. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


std::string SDFFont::presetFontNameToString(PresetFontName fn) const {

    std::string fileName;
    switch (fn) {
    case (PRESET_EVOLVENTA_SANS):
        fileName = "Evolventa-SansSerif";
        break;
    case (PRESET_ROBOTO_SANS):
        fileName = "Roboto-SansSerif";
        break;
    case (PRESET_VOLLKORN_SERIF):
        fileName = "Vollkorn-Serif";
        break;
    case (PRESET_UBUNTU_MONO):
        fileName = "Ubuntu-Mono";
        break;
    default:
        break;
    }
    return fileName;
}


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
    newVBO.name = "inPos";
    newVBO.index = (GLuint)VBOAttrib::POSITION;
    newVBO.dim = 3;
    this->vbos.push_back(newVBO);

    // VBO for texture data
    newVBO.name = "inTexCoord";
    newVBO.index = (GLuint)VBOAttrib::TEXTURE;
    newVBO.dim = 2;
    this->vbos.push_back(newVBO);

    // VBO for texture data
    newVBO.name = "inColor";
    newVBO.index = (GLuint)VBOAttrib::COLOR;
    newVBO.dim = 4;
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
        glVertexAttribPointer(this->vbos[i].index, this->vbos[i].dim, GL_FLOAT, GL_FALSE, 0, (GLubyte*)nullptr);
    }

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    for (unsigned int i = 0; i < (unsigned int)this->vbos.size(); i++) {
        glDisableVertexAttribArray(this->vbos[i].index);
    }

    return true;
}


// Bitmap Font file format: http://www.angelcode.com/products/fnont/doc/file_format.html
bool SDFFont::loadFontInfo(vislib::StringW filename) {

    // Reset font info
    for (unsigned int i = 0; i < this->glyphs.size(); i++) {
        ARY_SAFE_DELETE(this->glyphs[i].kerns);
    }
    for (size_t i = 0; i < this->glyphIdcs.size(); i++) {
        this->glyphIdcs[i] = nullptr;
    }

    this->glyphs.clear();
    this->glyphIdcs.clear();

    std::ifstream input_file(this->to_string(filename.PeekBuffer()));
    if (!input_file.is_open() || !input_file.good()) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[SDFFont] Unable to open font file \"%s\": Bad file. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
            __LINE__);
        return false;
    }

    float texWidth = 0.0f;
    float texHeight = 0.0f;
    float lineHeight = 0.0f;
    size_t idx;
    float width;
    float height;
    unsigned int maxId = 0;
    ;
    std::vector<SDFGlyphKerning> tmpKerns;
    tmpKerns.clear();

    // Read info file line by line
    /// NB: Assuming file of correct format - there are no sanity checks for parsing
    std::string line;
    while (std::getline(input_file, line)) {
        // (1) Parse common info line
        if (line.rfind("common ", 0) == 0) { // starts with

            idx = line.find("scaleW=", 0);
            texWidth = (float)std::atof(line.substr(idx + 7, 4).c_str());

            idx = line.find("scaleH=", 0);
            texHeight = (float)std::atof(line.substr(idx + 7, 4).c_str());

            idx = line.find("lineHeight=", 0);
            lineHeight = (float)std::atof(line.substr(idx + 11, 4).c_str());
        }
        // (2) Parse character info
        else if (line.rfind("char ", 0) == 0) {
            SDFGlyphInfo newChar;

            idx = line.find("id=", 0);
            newChar.id = (unsigned int)std::atoi(line.substr(idx + 3, 5).c_str());

            if (maxId < newChar.id) {
                maxId = newChar.id;
            }

            idx = line.find("x=", 0);
            newChar.texX0 = (float)std::atof(line.substr(idx + 2, 4).c_str()) / texWidth;

            idx = line.find("y=", 0);
            newChar.texY0 = (float)std::atof(line.substr(idx + 2, 4).c_str()) / texHeight;

            idx = line.find("width=", 0);
            width = (float)std::atof(line.substr(idx + 6, 4).c_str());

            idx = line.find("height=", 0);
            height = (float)std::atof(line.substr(idx + 7, 4).c_str());

            newChar.width = width / lineHeight;
            newChar.height = height / lineHeight;

            idx = line.find("xoffset=", 0);
            newChar.xoffset = (float)std::atof(line.substr(idx + 8, 4).c_str()) / lineHeight;

            idx = line.find("yoffset=", 0);
            newChar.yoffset = (float)std::atof(line.substr(idx + 8, 4).c_str()) / lineHeight;

            idx = line.find("xadvance=", 0);
            newChar.xadvance = (float)std::atof(line.substr(idx + 9, 4).c_str()) / lineHeight;

            newChar.kernCnt = 0;
            newChar.kerns = nullptr;

            newChar.texX1 = newChar.texX0 + width / texWidth;
            newChar.texY1 = newChar.texY0 + height / texHeight;

            this->glyphs.push_back(newChar);
        }
        // (3) Parse kerning info
        else if (line.rfind("kerning ", 0) == 0) {

            SDFGlyphKerning newKern;

            idx = line.find("first=", 0);
            newKern.previous = (unsigned int)std::atoi(line.substr(idx + 6, 4).c_str());

            idx = line.find("second=", 0);
            newKern.current = (unsigned int)std::atoi(line.substr(idx + 7, 4).c_str());

            idx = line.find("amount=", 0);
            newKern.xamount = (float)std::atof(line.substr(idx + 7, 4).c_str()) / lineHeight;

            tmpKerns.push_back(newKern);
        }
    }
    // Close file
    input_file.close();

    // Init index pointer array
    maxId++;
    for (unsigned int i = 0; i < maxId; i++) {
        this->glyphIdcs.push_back(nullptr);
    }
    // Set pointers to available glyph info
    for (unsigned int i = 0; i < (unsigned int)this->glyphs.size(); i++) {
        // Filling character index array --------------------------------------
        if (this->glyphs[i].id > (unsigned int)this->glyphIdcs.size()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[SDFFont] Character is out of range: \"%i\". [%s, %s, line %d]\n", this->glyphs[i].id, __FILE__,
                __FUNCTION__, __LINE__);
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


bool SDFFont::loadFontShader(megamol::frontend_resources::RuntimeConfig const& runtimeConf) {

    auto const shader_options = core::utility::make_path_shader_options(runtimeConf);

    try {
        auto shader_options_globcol = shader_options;
        shader_options_globcol.addDefinition("GLOBAL_COLOR");

        this->shaderglobcol = core::utility::make_shared_glowl_shader(
            "globcol", shader_options_globcol, "core/sdffont/sdffont.vert.glsl", "core/sdffont/sdffont.frag.glsl");
        this->shadervertcol = core::utility::make_shared_glowl_shader(
            "vertcol", shader_options, "core/sdffont/sdffont.vert.glsl", "core/sdffont/sdffont.frag.glsl");

    } catch (std::exception& e) {
        Log::DefaultLog.WriteError(("SimplestSphereRenderer: " + std::string(e.what())).c_str());
        return false;
    }

    return true;
}
