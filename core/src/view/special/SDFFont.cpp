/*
 * SDFFont.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "mmcore/view/special/SDFFont.h"

#include "mmcore/misc/PngBitmapCodec.h"

#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/math/Vector.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/File.h"
#include "vislib/sys/FastFile.h"
#include "vislib/CharTraits.h"
#include "vislib/memutils.h"
#include "vislib/UTF8Encoder.h"
#include "vislib/math/ShallowMatrix.h"
#include "vislib/math/Matrix.h"
#include "vislib/sys/ASCIIFileBuffer.h"


using namespace vislib;

using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::view;
using namespace megamol::core::view::special;


/* PUBLIC ********************************************************************/


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const BitmapFont bmf) : AbstractFont(),
    renderType(SDFFont::RENDERTYPE_FILL), texture(), shader(), characters(), vbos() {

    this->loadFont(bmf);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const BitmapFont bmf,  SDFFont::RenderType render) : AbstractFont(), 
    font(bmf), renderType(render), texture(), shader(), characters(), vbos() {

    this->loadFont(bmf);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const BitmapFont bmf, float size)  : AbstractFont(), 
    font(bmf), renderType(SDFFont::RENDERTYPE_FILL), texture(), shader(), characters(), vbos() {

    this->SetSize(size);

    this->loadFont(bmf);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const BitmapFont bmf, bool flipY) : AbstractFont(), 
    font(bmf), renderType(SDFFont::RENDERTYPE_FILL), texture(), shader(), characters(), vbos() {

    this->SetFlipY(flipY);

    this->loadFont(bmf);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const BitmapFont bmf, SDFFont::RenderType render, bool flipY) : AbstractFont(),
    font(bmf), renderType(render), texture(), shader(), characters(), vbos() {

    this->SetFlipY(flipY);

    this->loadFont(bmf);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const BitmapFont bmf, float size, bool flipY) : AbstractFont(), 
    font(bmf), renderType(SDFFont::RENDERTYPE_FILL), texture(), shader(), characters(), vbos() {

    this->SetSize(size);
    this->SetFlipY(flipY);

    this->loadFont(bmf);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const BitmapFont bmf, float size, SDFFont::RenderType render) : AbstractFont(), 
    font(bmf), renderType(render), texture(), shader(), characters(), vbos() {

    this->SetSize(size);

    this->loadFont(bmf);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const BitmapFont bmf, float size, SDFFont::RenderType render, bool flipY) : AbstractFont(),
        font(bmf), renderType(render), texture(), shader(), characters(), vbos() {

    this->SetSize(size);
    this->SetFlipY(flipY);

    this->loadFont(bmf);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src) : AbstractFont(),
    font(src.font), renderType(src.renderType), texture(), shader(), characters(), vbos() {

    this->SetSize(src.GetSize());
    this->SetFlipY(src.IsFlipY());

    this->loadFont(src.font);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src, SDFFont::RenderType render) : AbstractFont(), 
    font(src.font), renderType(render), texture(), shader(), characters(), vbos() {

    this->SetSize(src.GetSize());
    this->SetFlipY(src.IsFlipY());

    this->loadFont(src.font);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src, float size) : AbstractFont(),
        font(src.font), renderType(src.renderType), texture(), shader(), characters(), vbos() {

    this->SetSize(size);
    this->SetFlipY(src.IsFlipY());

    this->loadFont(src.font);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src, bool flipY) : AbstractFont(),
        font(src.font), renderType(src.renderType), texture(), shader(), characters(), vbos() {

    this->SetSize(src.GetSize());
    this->SetFlipY(flipY);

    this->loadFont(src.font);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src, SDFFont::RenderType render, bool flipY) : AbstractFont(),
        font(src.font), renderType(render), texture(), shader(), characters(), vbos() {

    this->SetSize(src.GetSize());
    this->SetFlipY(flipY);

    this->loadFont(src.font);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src, float size, bool flipY) : AbstractFont(), 
    font(src.font), renderType(src.renderType), texture(), shader(), characters(), vbos() {

    this->SetSize(size);
    this->SetFlipY(flipY);

    this->loadFont(src.font);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src, float size,  SDFFont::RenderType render) : AbstractFont(), 
    font(src.font),  renderType(render), texture(), shader(), characters(), vbos() {

    this->SetSize(size);
    this->SetFlipY(src.IsFlipY());

    this->loadFont(src.font);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src, float size, SDFFont::RenderType render, bool flipY) : AbstractFont(),
        font(src.font), renderType(render), texture(), shader(), characters(), vbos() {

    this->SetSize(size);
    this->SetFlipY(flipY);

    this->loadFont(src.font);
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

    return this->lineCount(maxWidth, size, txt);
}


/*
 * SDFFont::BlockLines
 */
unsigned int SDFFont::BlockLines(float maxWidth, float size, const wchar_t *txt) const {

    return this->lineCount(maxWidth, size, txt);
}


/*
 * SDFFont::DrawString
 */
void SDFFont::DrawString(float x, float y, float size, bool flipY, const char *txt, AbstractFont::Alignment align) const {

    //TEMP
    this->draw(txt, x, y, 0.0f, size, flipY, align);

    // TODO

    /*
    int *run = this->buildGlyphRun(txt, FLT_MAX);

    if ((align == ALIGN_CENTER_MIDDLE) || (align == ALIGN_LEFT_MIDDLE) || (align == ALIGN_RIGHT_MIDDLE)) {
        y += static_cast<float>(this->lineCount(run, false)) * 0.5f * size * (flipY ? 1.0f : -1.0f);

    } else if ((align == ALIGN_CENTER_BOTTOM) || (align == ALIGN_LEFT_BOTTOM) || (align == ALIGN_RIGHT_BOTTOM)) {
        y += static_cast<float>(this->lineCount(run, false)) * size * (flipY ? 1.0f : -1.0f);
    }

    if ((this->renderType == RENDERTYPE_FILL) || (this->renderType == RENDERTYPE_FILL_AND_OUTLINE)) {
        this->drawFilled(run, x, y, 0.0f, size, flipY, align);
    }
    if ((this->renderType == RENDERTYPE_OUTLINE) || (this->renderType == RENDERTYPE_FILL_AND_OUTLINE)) {
        this->drawOutline(run, x, y, 0.0f, size, flipY, align);
    }

    delete[] run;
    */
}


/*
* SDFFont::DrawString
*/
void SDFFont::DrawString(float x, float y, float size, bool flipY, const wchar_t *txt, AbstractFont::Alignment align) const {


}


/*
 * SDFFont::DrawString
 */
void SDFFont::DrawString(float x, float y, float w, float h, float size, bool flipY, const char *txt, AbstractFont::Alignment align) const {

    //TEMP
    this->draw(txt, x, y, 0.0f, size, flipY, align);

    // TODO

    /*
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
#endif // !_WIN32
    }

    if ((this->renderType == RENDERTYPE_FILL) || (this->renderType == RENDERTYPE_FILL_AND_OUTLINE)) {
        this->drawFilled(run, x, y, 0.0f, size, flipY, align);
    }
    if ((this->renderType == RENDERTYPE_OUTLINE) || (this->renderType == RENDERTYPE_FILL_AND_OUTLINE)) {
        this->drawOutline(run, x, y, 0.0f, size, flipY, align);
    }

    delete[] run;
    */
}


/*
 * SDFFont::DrawString
 */
void SDFFont::DrawString(float x, float y, float w, float h, float size,  bool flipY, const wchar_t *txt, AbstractFont::Alignment align) const {


}


/*
* SDFFont::DrawString
*/
void SDFFont::DrawString(float x, float y, float z, float size, bool flipY, const char * txt, Alignment align) const {

    //TEMP
    this->draw(txt, x, y, z, size, flipY, align);

    // TODO

    /*
    int *run = this->buildGlyphRun(txt, FLT_MAX);

    if ((align == ALIGN_CENTER_MIDDLE) || (align == ALIGN_LEFT_MIDDLE) || (align == ALIGN_RIGHT_MIDDLE)) {
        y += static_cast<float>(this->lineCount(run, false)) * 0.5f * size * (flipY ? 1.0f : -1.0f);
    }
    else if ((align == ALIGN_CENTER_BOTTOM) || (align == ALIGN_LEFT_BOTTOM) || (align == ALIGN_RIGHT_BOTTOM)) {
        y += static_cast<float>(this->lineCount(run, false)) * size * (flipY ? 1.0f : -1.0f);
    }

    if ((this->renderType == RENDERTYPE_FILL) || (this->renderType == RENDERTYPE_FILL_AND_OUTLINE)) {
        this->drawFilled(run, x, y, z, size, flipY, align);
    }
    if ((this->renderType == RENDERTYPE_OUTLINE) || (this->renderType == RENDERTYPE_FILL_AND_OUTLINE)) {
        this->drawOutline(run, x, y, z, size, flipY, align);
    }

    delete[] run;
    */
}


/*
* SDFFont::DrawString
*/
void SDFFont::DrawString(float x, float y, float z, float size, bool flipY, const wchar_t * txt, Alignment align) const {


}


/*
 * SDFFont::LineWidth
 */
float SDFFont::LineWidth(float size, const char *txt) const {

    float w = 0.0f;

    // TODO

    return w;
}


/*
 * SDFFont::LineWidth
 */
float SDFFont::LineWidth(float size, const wchar_t *txt) const {

    float w = 0.0f;

    // TODO

    return w;
}


/* PRIVATE ********************************************************************/


/*
 * SDFFont::initialise
 */
bool SDFFont::initialise(void) {

    // unused so far ....

    return true;
}


/*
 * SDFFont::deinitialise
 */
void SDFFont::deinitialise(void) {

    // Texture
    this->texture.Release();
    // Shader
    this->shader.Release();
    // VAO
    glDeleteVertexArrays(1, &this->vaoHandle);
    // VBOs
    for (unsigned int i = 0; i < (unsigned int)this->vbos.size(); i++) {
        glDeleteBuffers(1, &this->vbos[i].handle);
    }
    this->vbos.clear();
}


/*
* SDFFont::lineCount
*/
unsigned int SDFFont::lineCount(float maxWidth, float size, const char *txt) const {

    unsigned int i = 0;

    // TODO

    return i;
}


/*
* SDFFont::lineCount
*/
unsigned int SDFFont::lineCount(float maxWidth, float size, const wchar_t *txt) const {

    unsigned int i = 0;

    // TODO

    return i;
}


/*
* SDFFont::draw
*/
void SDFFont::draw(const char *txt, float x, float y, float z, float size, bool flipY, Alignment align) const {

    // Check texture
    if (!this->texture.IsValid()) {
        //vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [draw] Texture is not valid. \n");
        return;
    }
    // Check shader
    if (!this->shader.IsValidHandle(this->shader.ProgramHandle())) {
        //vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [draw] Shader handle is not valid. \n");
        return;
    }

    // ------------------------------------------------------------------------

    // Loop over all characters in the string
    SDFFontCharacter * charInfo = NULL;
    unsigned int       charCnt = (unsigned int)this->indices.size();
    unsigned int       charIdx = 0;

    unsigned int textCnt = 0;
    const char *text = txt;
    while (*text != '\0') {
        text++;
        textCnt++;
    }

    GLfloat *posData   = new GLfloat[textCnt * 12];
    GLfloat *texData   = new GLfloat[textCnt * 8];
    GLfloat *transData = new GLfloat[textCnt * 16];

    // Assumption: String MUST be '\0' terminated!
    text = txt;
    textCnt = 0;
    while (*text != '\0') {
        charIdx = (unsigned int)(*text);

        // (TEMP: Replace newline wiwth space)
        if ((*text) == '\n') {
            charIdx = (unsigned int)(' ');
        }

        charInfo = NULL;
        if (charIdx < charCnt) {
            charInfo = this->indices[charIdx];
        }
        if (charInfo != NULL) {

            // Positions
            posData[textCnt * 12 + 0] = charInfo->posData[0];
            posData[textCnt * 12 + 1] = charInfo->posData[1];
            posData[textCnt * 12 + 2] = charInfo->posData[2];

            posData[textCnt * 12 + 3] = charInfo->posData[3];
            posData[textCnt * 12 + 4] = charInfo->posData[4];
            posData[textCnt * 12 + 5] = charInfo->posData[5];

            posData[textCnt * 12 + 6] = charInfo->posData[6];
            posData[textCnt * 12 + 7] = charInfo->posData[7];
            posData[textCnt * 12 + 8] = charInfo->posData[8];

            posData[textCnt * 12 + 9] = charInfo->posData[9];
            posData[textCnt * 12 + 10] = charInfo->posData[10];
            posData[textCnt * 12 + 11] = charInfo->posData[11];

            // Texture  
            texData[textCnt * 8 + 0] = charInfo->texData[0];
            texData[textCnt * 8 + 1] = charInfo->texData[1];

            texData[textCnt * 8 + 2] = charInfo->texData[2];
            texData[textCnt * 8 + 3] = charInfo->texData[3];

            texData[textCnt * 8 + 4] = charInfo->texData[4];
            texData[textCnt * 8 + 5] = charInfo->texData[5];

            texData[textCnt * 8 + 6] = charInfo->texData[6];
            texData[textCnt * 8 + 7] = charInfo->texData[7];

            // Transformation
            transData[textCnt * 16 + 0] = (float)textCnt * size;
            transData[textCnt * 16 + 1] = 0.0f;
            transData[textCnt * 16 + 2] = 0.0f;
            transData[textCnt * 16 + 3] = size;

            transData[textCnt * 16 + 4] = (float)textCnt * size;
            transData[textCnt * 16 + 5] = 0.0f;
            transData[textCnt * 16 + 6] = 0.0f;
            transData[textCnt * 16 + 7] = size;

            transData[textCnt * 16 + 8] = (float)textCnt * size;
            transData[textCnt * 16 + 9] = 0.0f;
            transData[textCnt * 16 + 10] = 0.0f;
            transData[textCnt * 16 + 11] = size;

            transData[textCnt * 16 + 12] = (float)textCnt * size;
            transData[textCnt * 16 + 13] = 0.0f;
            transData[textCnt * 16 + 14] = 0.0f;
            transData[textCnt * 16 + 15] = size;

            textCnt++;
        }
        text++;
    }

    for (unsigned int i = 0; i < (unsigned int)this->vbos.size(); i++) {
        glBindBuffer(GL_ARRAY_BUFFER, this->vbos[i].handle);
        if (this->vbos[i].index == (GLuint)VBOAttrib::POSITION) {
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)textCnt * 12 * sizeof(GLfloat), posData, GL_STATIC_DRAW);
        }
        else if (this->vbos[i].index == (GLuint)VBOAttrib::TEXTURE) {
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)textCnt * 8 * sizeof(GLfloat), texData, GL_STATIC_DRAW);
        }
        else if (this->vbos[i].index == (GLuint)VBOAttrib::TRANSFORM) {
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)textCnt * 16 * sizeof(GLfloat), transData, GL_STATIC_DRAW);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    // ------------------------------------------------------------------------
    // Draw

    // Get current matrices 
    GLfloat modelViewMatrix_column[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix_column);
    vislib::math::ShallowMatrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewMatrix(&modelViewMatrix_column[0]);
    GLfloat projMatrix_column[16];
    glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
    vislib::math::ShallowMatrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> projMatrix(&projMatrix_column[0]);
    // Compute modelviewprojection matrix
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewProjMatrix = projMatrix * modelViewMatrix;

    // Get current color
    GLfloat color[4];
    glGetFloatv(GL_CURRENT_COLOR, color);

    // Store opengl states
    bool depthEnabled = glIsEnabled(GL_DEPTH_TEST);
    if (!depthEnabled) {
        glEnable(GL_DEPTH_TEST);
    }
    bool blendEnabled = glIsEnabled(GL_BLEND);
    if (!blendEnabled) {
        glEnable(GL_BLEND);
    }
    GLint blendSrc;
    GLint blendDst;
    glGetIntegerv(GL_BLEND_SRC, &blendSrc);
    glGetIntegerv(GL_BLEND_DST, &blendDst);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glBindVertexArray(this->vaoHandle);

    glEnable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, this->texture.GetId()); // instead of this->texture.Bind() => because function is CONST

    glUseProgram(this->shader.ProgramHandle()); // instead of this->shader.Enable() => because function is CONST

    // Set shader variables
    glUniformMatrix4fv(this->shader.ParameterLocation("mvpMat"), 1, GL_FALSE, modelViewProjMatrix.PeekComponents());
    glUniform4fv(this->shader.ParameterLocation("color"), 1, color);
    glUniform1i(this->shader.ParameterLocation("fontTex"), 0);

    glDrawArrays(GL_QUADS, 0, (GLsizei)textCnt * 4);

    glUseProgram(0); // instead of this->shader.Disable() => because function is CONST
    glBindVertexArray(0);
    glDisable(GL_TEXTURE_2D);

    // Reset opengl states
    if (!depthEnabled) {
        glDisable(GL_DEPTH_TEST);
    }
    if (!blendEnabled) {
        glDisable(GL_BLEND);
    }
    glBlendFunc(blendSrc, blendDst);

    delete[] posData;
    delete[] texData;
    delete[] transData;



    // TODO (with SSBO)

    /*
    float gx = x;
    float gy = y;
    float sy = flipY ? -size : size;

    if ((align == ALIGN_CENTER_BOTTOM) || (align == ALIGN_CENTER_MIDDLE) || (align == ALIGN_CENTER_TOP)) {
        gx -= this->lineWidth(run, false) * size * 0.5f;
    }
    else if ((align == ALIGN_RIGHT_BOTTOM) || (align == ALIGN_RIGHT_MIDDLE) || (align == ALIGN_RIGHT_TOP)) {
        gx -= this->lineWidth(run, false) * size;
    }


    glEnableClientState(GL_VERTEX_ARRAY);
    glDisable(GL_CULL_FACE);

    while (*run != 0) {
        const SDFGlyphInfo &glyph = this->data.glyph[
            (*run < 0) ? (-1 - *run) : (*run - 1)];

        if (*run < 0) {
            gx = x;

            if ((align == ALIGN_CENTER_BOTTOM) || (align == ALIGN_CENTER_MIDDLE) || (align == ALIGN_CENTER_TOP)) {
                gx -= this->lineWidth(run, false) * size * 0.5f;
            }
            else if ((align == ALIGN_RIGHT_BOTTOM) || (align == ALIGN_RIGHT_MIDDLE) || (align == ALIGN_RIGHT_TOP)) {
                gx -= this->lineWidth(run, false) * size;
            }

            gy += sy;
        }

        glPushMatrix();
        glTranslatef(gx, gy, z);
        glScalef(size, sy, 1.0f);
        glVertexPointer(2, GL_FLOAT, 0, glyph.points);
        glDrawElements(GL_TRIANGLES, glyph.triCount, GL_UNSIGNED_SHORT, glyph.tris);
        glPopMatrix();

        gx += glyph.width * size;

        run++;
    }

    glDisableClientState(GL_VERTEX_ARRAY);
    */
}


/*
* SDFFont::draw
*/
void SDFFont::draw(const wchar_t *txt, float x, float y, float z, float size, bool flipY, Alignment align) const {

    // TODO

}


/*
* SDFFont::loadFont
*/
bool SDFFont::loadFont(BitmapFont bmf) {

    // Convert BitmapFont to string
    vislib::StringA fontName = "";
    switch (bmf) {
        case  (BitmapFont::BMFONT_EVOLVENTA): fontName = "evolventa"; break;
        default: break;
    }

    // Folder holding font data
    vislib::StringA folder = ".\\fonts\\";

    // 1) Load buffers --------------------------------------------------------
    if (!this->loadFontBuffers()) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadFont] Failed to load buffers. \n");
        return false;
    }

    // 2) Load font information -----------------------------------------------
    vislib::StringA infoFile = folder;
    infoFile.Append(fontName);
    infoFile.Append(".fnt");
    if (!this->loadFontInfo(infoFile)) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadFont] Failed to load font info file. \n");
        return false;
    }

    // 3) Load texture --------------------------------------------------------
    vislib::StringA textureFile = folder;
    textureFile.Append(fontName);
    textureFile.Append(".png");
    if (!this->loadFontTexture(textureFile)) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadFont] Failed to loda font texture. \n");
        return false;
    }

    // 4) Load shaders --------------------------------------------------------
    vislib::StringA vertShaderFile = folder;
    vertShaderFile.Append("vertex.shader");
    vislib::StringA fragShaderFile = folder;
    fragShaderFile.Append("fragment.shader");
    if (!this->loadFontShader(vertShaderFile, fragShaderFile)) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadFont] Failed to load font shaders. \n");
        return false;
    }  

    return true;
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

    // Create Vertex Array Object 
    glGenVertexArrays(1, &this->vaoHandle);
    glBindVertexArray(this->vaoHandle);

    // Init vbos
    SDFVBO newVBO;

    // VBO for position data
    newVBO.name   = "inVertPos";
    newVBO.index  = (GLuint)VBOAttrib::POSITION;
    newVBO.dim    = 3;
    this->vbos.push_back(newVBO);
    newVBO.handle = 0;

    // VBO for texture data
    newVBO.name   = "inVertTexCoord";
    newVBO.index  = (GLuint)VBOAttrib::TEXTURE;
    newVBO.dim    = 2;
    this->vbos.push_back(newVBO);
    newVBO.handle = 0;

    // VBO for transformation data
    newVBO.name   = "inVertOffset";
    newVBO.index  = (GLuint)VBOAttrib::TRANSFORM;
    newVBO.dim    = 4;
    this->vbos.push_back(newVBO);
    newVBO.handle = 0;

    for (unsigned int i = 0; i < (unsigned int)this->vbos.size(); i++) {
        glGenBuffers(1, &this->vbos[i].handle);
        glBindBuffer(GL_ARRAY_BUFFER, this->vbos[i].handle);
        // Create empty buffer
        glBufferData(GL_ARRAY_BUFFER, 0, NULL, GL_STATIC_DRAW);
        // Bind buffer to vertex attribute
        glEnableVertexAttribArray(this->vbos[i].index); 
        glVertexAttribPointer(this->vbos[i].index, this->vbos[i].dim, GL_FLOAT, GL_FALSE, 0, (GLubyte *)NULL);
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
* Bitmap Font file format: http://www.angelcode.com/products/bmfont/doc/file_format.html
*
*/
bool SDFFont::loadFontInfo(vislib::StringA filename) {

    // Reset font info
    this->characters.clear();

    // Load file
    vislib::sys::ASCIIFileBuffer file;
    if (!file.LoadFile(filename)) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadfontCharacters] Could not load file as ascii buffer: \"%s\". \n", filename.PeekBuffer());
        return false;
    }

    float texWidth  = 0.0f;
    float texHeight = 0.0f;
    float fontSize  = 0.0f;
    
    // Read info file line by line
    int idx = 0;
    int second = 0;
    SIZE_T lineCnt = 0;
    vislib::StringA line;
    while (lineCnt < file.Count()) {
        line = static_cast<vislib::StringA>(file.Line(lineCnt));
        if (line.StartsWith("info ")) { // Parse info line
            idx = line.Find("size=", 0);
            fontSize = (float)std::atof(line.Substring(idx + 5, 4));
        }
        else if (line.StartsWith("common ")) { // Parse common info line
            idx = line.Find("scaleW=", 0);
            texWidth = (float)std::atof(line.Substring(idx + 7, 4));
            idx = line.Find("scaleH=", 0);
            texHeight = (float)std::atof(line.Substring(idx + 7, 4));
        }
        else if (line.StartsWith("char ")) { // Parse character info
            SDFFontCharacter newChar;

            idx = line.Find("id=", 0);
            newChar.id = (int)std::atoi(line.Substring(idx+3, 4)); 
            idx = line.Find("x=", 0);
            newChar.texX = (int)std::atoi(line.Substring(idx+2, 4));
            idx = line.Find("y=", 0);
            newChar.texY = (int)std::atoi(line.Substring(idx+2, 4));
            idx = line.Find("width=", 0);
            newChar.width = (int)std::atoi(line.Substring(idx+6, 4));
            idx = line.Find("height=", 0);
            newChar.height = (int)std::atoi(line.Substring(idx+7, 4));
            idx = line.Find("xoffset=", 0);
            newChar.xoffset = (int)std::atoi(line.Substring(idx+8, 4));
            idx = line.Find("yoffset=", 0);
            newChar.yoffset  = (int)std::atoi(line.Substring(idx+8, 4));
            idx = line.Find("xadvance=", 0);
            newChar.xadvance = (int)std::atoi(line.Substring(idx+9, 4));
            newChar.kernings.clear();
            newChar.posData.Clear();
            newChar.texData.Clear();

            this->characters.push_back(newChar);
        }
        else if (line.StartsWith("kerning ")) { // Parse kerning info

            idx = line.Find("second=", 0);
            second = (int)std::atoi(line.Substring(idx+7, 4));

            SDFFontKerning newKern;

            idx = line.Find("first=", 0);
            newKern.previous = (int)std::atoi(line.Substring(idx+6, 4));
            idx = line.Find("amount=", 0);
            newKern.amount = (int)std::atoi(line.Substring(idx+7, 4));

            // Assumption: Character data is already read
            for (unsigned int i = 0; i < (unsigned int)this->characters.size(); i++) {
                if (this->characters[i].id == second) {
                    this->characters[i].kernings.push_back(newKern);
                }
            }
        }
        // Proceed with next line ...
        lineCnt++;
    }
    //Clear ascii file buffer
    file.Clear();

    // Building character index array -----------------------------------------
    this->indices.clear();
    // Get largest character id
    int maxCharId = 0;
    for (unsigned int i = 0; i < (unsigned int)this->characters.size(); i++) {
        if (this->characters[i].id > maxCharId) {
            maxCharId = this->characters[i].id;
        }
    }
    maxCharId++; // [0,maxCharId], otherwise indices are in [0,maxCharId-1]
    // Init font info indices
    for (int i = 0; i < maxCharId; i++) {
        this->indices.push_back(NULL);
    }
    // Set pointer to font info for available characters
    for (unsigned int i = 0; i < (unsigned int)this->characters.size(); i++) {
        this->indices[this->characters[i].id] = &this->characters[i];
    }

    // Generating position and texture data -----------------------------------
    GLfloat tx0, tx1, ty0, ty1;
    vislib::math::Vector<GLfloat, 3> p0, p1, p2, p3;
    float width, height;
    for (unsigned int i = 0; i < (unsigned int)this->characters.size(); i++) {
        this->characters[i].posData.Clear();
        this->characters[i].texData.Clear();

        width  = (float)this->characters[i].width;
        height = (float)this->characters[i].height;

        // Position data
        p0 = vislib::math::Vector<GLfloat, 3>(0.0f, 0.0f, 0.0f);
        p1 = p0;
        p1.SetX(p1.X() + (width / fontSize));
        p2 = p1;
        p2.SetY(p2.Y() + (height / fontSize));
        p3 = p0;
        p3.SetY(p3.Y() + (height / fontSize));
        this->characters[i].posData.Add(p0.X()); this->characters[i].posData.Add(p0.Y()); this->characters[i].posData.Add(p0.Z());
        this->characters[i].posData.Add(p1.X()); this->characters[i].posData.Add(p1.Y()); this->characters[i].posData.Add(p1.Z());
        this->characters[i].posData.Add(p2.X()); this->characters[i].posData.Add(p2.Y()); this->characters[i].posData.Add(p2.Z());
        this->characters[i].posData.Add(p3.X()); this->characters[i].posData.Add(p3.Y()); this->characters[i].posData.Add(p3.Z());

        // Texture data
        tx0 = (GLfloat)this->characters[i].texX / texWidth;
        ty0 = (GLfloat)this->characters[i].texY / texHeight;
        tx1 = tx0 + (GLfloat)width / texWidth;
        ty1 = ty0 + (GLfloat)height / texHeight;
        this->characters[i].texData.Add(tx0); this->characters[i].texData.Add(ty1);
        this->characters[i].texData.Add(tx1); this->characters[i].texData.Add(ty1);
        this->characters[i].texData.Add(tx1); this->characters[i].texData.Add(ty0);
        this->characters[i].texData.Add(tx0); this->characters[i].texData.Add(ty0);
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
    void   *buf  = NULL;
    SIZE_T  size = 0;

    if ((size = this->loadFile(filename, &buf)) <= 0) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadTexture] Could not find texture: \"%s\". \n", filename.PeekBuffer());
        ARY_SAFE_DELETE(buf);
        return false;
    }

    if (pbc.Load(buf, size)) {
        // (Using template with minimum channels containing alpha)
        img.Convert(vislib::graphics::BitmapImage::TemplateByteGrayAlpha); 
        // (Red channel is Gray value - Green channel is alpha value from png)
        if (this->texture.Create(img.Width(), img.Height(), false, img.PeekDataAs<BYTE>(), GL_RG) != GL_NO_ERROR) { 
            vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadTexture] Could not load texture: \"%s\". \n", filename.PeekBuffer());
            ARY_SAFE_DELETE(buf);
            return false;
        }
        this->texture.SetFilter(GL_LINEAR, GL_LINEAR);
        this->texture.SetWrap(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
        ARY_SAFE_DELETE(buf);
    }
    else {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadTexture] Could not read texture: \"%s\". \n", filename.PeekBuffer());
        ARY_SAFE_DELETE(buf);
        return false;
    }

    return true;
}


/*
* SDFFont::loadFontShader
*/
bool SDFFont::loadFontShader(vislib::StringA vert, vislib::StringA frag) {

    // Reset shader
    this->shader.Release();

    const char *shaderName = "SDFFont";
    SIZE_T size = 0;

    void *vertBuf = NULL;
    if ((size = this->loadFile(vert, &vertBuf)) <= 0) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadShader] Could not find vertex shader: \"%s\". \n", vert.PeekBuffer());
        ARY_SAFE_DELETE(vertBuf);
        return false;
    }
    // Terminating buffer with '\0' is mandatory for being able to compile shader
    ((char *)vertBuf)[size-1] = '\0';

    void *fragBuf = NULL;
    if ((size = this->loadFile(frag, &fragBuf)) <= 0) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadShader] Could not find fragment shader: \"%s\". \n", frag.PeekBuffer());
        ARY_SAFE_DELETE(fragBuf);
        return false;
    }
    // Terminating buffer with '\0' is mandatory for being able to compile shader
    ((char *)fragBuf)[size-1] = '\0';

    try {
        // Compiling shaders
        if (!this->shader.Compile((const char **)(&vertBuf), (SIZE_T)1, (const char **)(&fragBuf), (SIZE_T)1)) {
            vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadShader] Unable to compile \"%s\"-shader: Unknown error. \n", shaderName);
            ARY_SAFE_DELETE(vertBuf);
            ARY_SAFE_DELETE(fragBuf);
            return false;
        }

        // Bind vertex shader attributes (before linking shaders!)
        for (unsigned int i = 0; i < (unsigned int)this->vbos.size(); i++) {
            glBindAttribLocation(this->shader.ProgramHandle(), this->vbos[i].index, this->vbos[i].name.PeekBuffer());
        }

        // Linking shaders
        if (!this->shader.Link()) {
            vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadShader] Unable to link \"%s\"-shader: Unknown error. \n", shaderName);
            ARY_SAFE_DELETE(vertBuf);
            ARY_SAFE_DELETE(fragBuf);
            return false;
        }

        ARY_SAFE_DELETE(vertBuf);
        ARY_SAFE_DELETE(fragBuf);
    }
    catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadShader] Unable to compile \"%s\"-shader (@%s): %s. \n", shaderName,
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()), ce.GetMsgA());
        return false;
    }
    catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadShader] Unable to compile \"%s\"-shader: %s. \n", shaderName, e.GetMsgA());
        return false;
    }
    catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadShader] Unable to compile \"%s\"-shader: Unknown exception. \n", shaderName);
        return false;
    }

    return true;
}


/*
* SDFFont::loadFile
*/
SIZE_T SDFFont::loadFile(vislib::StringA filename, void **outData) {

    // Reset out data
    *outData = NULL;


    vislib::StringW name = static_cast<vislib::StringW>(filename);
    if (name.IsEmpty()) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadFile] Unable to load file: No name given. \n");
        return false;
    }
    if (!vislib::sys::File::Exists(name)) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadFile] Unable to load not existing file: \"%s\". \n", filename.PeekBuffer());
        return false;
    }

    SIZE_T size = static_cast<SIZE_T>(vislib::sys::File::GetSize(name));
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
    SIZE_T num = static_cast<SIZE_T>(f.Read(*outData, size));
    if (num != size) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadFile] Unable to read whole file: \"%s\". \n", filename.PeekBuffer());
        ARY_SAFE_DELETE(*outData);
        return false;
    }

    return num;
}

