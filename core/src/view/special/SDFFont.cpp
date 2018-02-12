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
    renderType(SDFFont::RENDERTYPE_FILL), texture(), shader(), fontCharacters(), fontKernings() {

    this->loadFont(bmf);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const BitmapFont bmf,  SDFFont::RenderType render) : AbstractFont(), 
    font(bmf), renderType(render), texture(), shader(), fontCharacters(), fontKernings() {

    this->loadFont(bmf);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const BitmapFont bmf, float size)  : AbstractFont(), 
    font(bmf), renderType(SDFFont::RENDERTYPE_FILL), texture(), shader(), fontCharacters(), fontKernings() {

    this->SetSize(size);

    this->loadFont(bmf);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const BitmapFont bmf, bool flipY) : AbstractFont(), 
    font(bmf), renderType(SDFFont::RENDERTYPE_FILL), texture(), shader(), fontCharacters(), fontKernings() {

    this->SetFlipY(flipY);

    this->loadFont(bmf);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const BitmapFont bmf, SDFFont::RenderType render, bool flipY) : AbstractFont(),
    font(bmf), renderType(render), texture(), shader(), fontCharacters(), fontKernings() {

    this->SetFlipY(flipY);

    this->loadFont(bmf);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const BitmapFont bmf, float size, bool flipY) : AbstractFont(), 
    font(bmf), renderType(SDFFont::RENDERTYPE_FILL), texture(), shader(), fontCharacters(), fontKernings() {

    this->SetSize(size);
    this->SetFlipY(flipY);

    this->loadFont(bmf);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const BitmapFont bmf, float size, SDFFont::RenderType render) : AbstractFont(), 
    font(bmf), renderType(render), texture(), shader(), fontCharacters(), fontKernings() {

    this->SetSize(size);

    this->loadFont(bmf);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const BitmapFont bmf, float size, SDFFont::RenderType render, bool flipY) : AbstractFont(),
        font(bmf), renderType(render), texture(), shader(), fontCharacters(), fontKernings() {

    this->SetSize(size);
    this->SetFlipY(flipY);

    this->loadFont(bmf);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src) : AbstractFont(),
    font(src.font), renderType(src.renderType), texture(), shader(), fontCharacters(), fontKernings() {

    this->SetSize(src.GetSize());
    this->SetFlipY(src.IsFlipY());

    this->loadFont(src.font);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src, SDFFont::RenderType render) : AbstractFont(), 
    font(src.font), renderType(render), texture(), shader(), fontCharacters(), fontKernings() {

    this->SetSize(src.GetSize());
    this->SetFlipY(src.IsFlipY());

    this->loadFont(src.font);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src, float size) : AbstractFont(),
        font(src.font), renderType(src.renderType), texture(), shader(), fontCharacters(), fontKernings() {

    this->SetSize(size);
    this->SetFlipY(src.IsFlipY());

    this->loadFont(src.font);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src, bool flipY) : AbstractFont(),
        font(src.font), renderType(src.renderType), texture(), shader(), fontCharacters(), fontKernings() {

    this->SetSize(src.GetSize());
    this->SetFlipY(flipY);

    this->loadFont(src.font);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src, SDFFont::RenderType render, bool flipY) : AbstractFont(),
        font(src.font), renderType(render), texture(), shader(), fontCharacters(), fontKernings() {

    this->SetSize(src.GetSize());
    this->SetFlipY(flipY);

    this->loadFont(src.font);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src, float size, bool flipY) : AbstractFont(), 
    font(src.font), renderType(src.renderType), texture(), shader(), fontCharacters(), fontKernings() {

    this->SetSize(size);
    this->SetFlipY(flipY);

    this->loadFont(src.font);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src, float size,  SDFFont::RenderType render) : AbstractFont(), 
    font(src.font),  renderType(render), texture(), shader(), fontCharacters(), fontKernings() {

    this->SetSize(size);
    this->SetFlipY(src.IsFlipY());

    this->loadFont(src.font);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src, float size, SDFFont::RenderType render, bool flipY) : AbstractFont(),
        font(src.font), renderType(render), texture(), shader(), fontCharacters(), fontKernings() {

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

    return this->lineCount(maxWidth, size, vislib::StringA(txt));
}


/*
 * SDFFont::BlockLines
 */
unsigned int SDFFont::BlockLines(float maxWidth, float size, const wchar_t *txt) const {

    return this->lineCount(maxWidth, size, static_cast<vislib::StringA>(vislib::StringW(txt)));
}


/*
 * SDFFont::DrawString
 */
void SDFFont::DrawString(float x, float y, float size, bool flipY, const char *txt, AbstractFont::Alignment align) const {

    //TEMP
    this->draw(vislib::StringA(txt), x, y, 0.0f, size, flipY, align);

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

    this->DrawString(x, y, size, flipY, (static_cast<vislib::StringA>(vislib::StringW(txt))).PeekBuffer(), align);
}


/*
 * SDFFont::DrawString
 */
void SDFFont::DrawString(float x, float y, float w, float h, float size, bool flipY, const char *txt, AbstractFont::Alignment align) const {

    //TEMP
    this->draw(vislib::StringA(txt), x, y, 0.0f, size, flipY, align);

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

    this->DrawString(x, y, w, h, size, flipY, (static_cast<vislib::StringA>(vislib::StringW(txt))).PeekBuffer(), align);
}


/*
* SDFFont::DrawString
*/
void SDFFont::DrawString(float x, float y, float z, float size, bool flipY, const char * txt, Alignment align) const {

    //TEMP
    this->draw(vislib::StringA(txt), x, y, z, size, flipY, align);

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

    this->DrawString(x, y, z, size, flipY, (static_cast<vislib::StringA>(vislib::StringW(txt))).PeekBuffer(), align);
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

    return this->LineWidth(size, (static_cast<vislib::StringA>(vislib::StringW(txt))).PeekBuffer());
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
    if (this->texture.IsValid()) {
        this->texture.Release();
    }

    // Shader
    this->shader.Release();

    // VAO
    if (glIsVertexArray(this->vaoHandle)) {
        glDeleteVertexArrays(1, &this->vaoHandle);
    }

    // VBOs
    if (glIsBuffer(this->vboHandles[0])) {
        glDeleteBuffers(1, &this->vboHandles[0]);
    }
    if (glIsBuffer(this->vboHandles[1])) {
        glDeleteBuffers(1, &this->vboHandles[1]);
    }
}


/*
* SDFFont::lineCount
*/
unsigned int SDFFont::lineCount(float maxWidth, float size, vislib::StringA txt) const {

    unsigned int i = 0;

    // TODO

    return i;
}


/*
* SDFFont::draw
*/
void SDFFont::draw(vislib::StringA txt, float x, float y, float z, float size, bool flipY, Alignment align) const {

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
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }

    // Enable buffers, texture and shader
    glBindVertexArray(this->vaoHandle);
    glEnable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE0);

    glBindTexture(GL_TEXTURE_2D, this->texture.GetId()); // = this->texture.Bind(); => can't be used because draw function has to be CONST

    glUseProgram(this->shader.ProgramHandle()); // = this->shader.Enable(); => can't be used because draw function has to be CONST

    // Set shader variables
    glUniformMatrix4fv(this->shader.ParameterLocation("mvpMat"), 1, GL_FALSE, modelViewProjMatrix.PeekComponents());
    glUniform4fv(this->shader.ParameterLocation("color"), 1, color);
    glUniform1i(this->shader.ParameterLocation("fontTex"), 0);

    // Draw ...
    glDrawArrays(GL_QUADS, 0, 4);

    // Disable buffers, texture and shader
    glUseProgram(0); // =  this->shader.Disable(); => can't be used because draw function has to be CONST
    glBindVertexArray(0);
    glDisable(GL_TEXTURE_2D);

    // Reset opengl states
    if (!depthEnabled) {
        glDisable(GL_DEPTH_TEST);
    }
    if (!blendEnabled) {
        glDisable(GL_BLEND);
    }

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


    // 1) Load font information -----------------------------------------------
    vislib::StringA infoFile = folder;
    infoFile.Append(fontName);
    infoFile.Append(".fnt");
    if (!this->loadFontInfo(infoFile)) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadFont] Failed to load font info file. \n");
        return false;
    }

    // 2) Load texture --------------------------------------------------------
    vislib::StringA textureFile = folder;
    textureFile.Append(fontName);
    textureFile.Append(".png");
    if (!this->loadFontTexture(textureFile)) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadFont] Failed to loda font texture. \n");
        return false;
    }

    // 3) Load shaders --------------------------------------------------------
    vislib::StringA vertShaderFile = folder;
    vertShaderFile.Append("vertex.shader");
    vislib::StringA fragShaderFile = folder;
    fragShaderFile.Append("fragment.shader");
    if (!this->loadShader(vertShaderFile, fragShaderFile)) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadFont] Failed to load font shaders. \n");
        return false;
    }  

    // 4) Load buffers --------------------------------------------------------
    if (!this->loadBuffers()) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadFont] Failed to load buffers. \n");
        return false;
    }


    //DEBUG Print generall OpenGL information ---------------------------------
    /*
    const GLubyte *renderer    = glGetString(GL_RENDERER);
    const GLubyte *vendor      = glGetString(GL_VENDOR);
    const GLubyte *version     = glGetString(GL_VERSION);
    const GLubyte *glslVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);
    GLint major, minor;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
    GLint nExtensions;
    glGetIntegerv(GL_NUM_EXTENSIONS, &nExtensions);
    vislib::sys::Log::DefaultLog.WriteInfo("[SDFFont] [OpenGL Info] GL Vendor : %s\n", vendor);
    vislib::sys::Log::DefaultLog.WriteInfo("[SDFFont] [OpenGL Info] GL Renderer : %s\n", renderer);
    vislib::sys::Log::DefaultLog.WriteInfo("[SDFFont] [OpenGL Info] GL Version (string) : %s\n", version);
    vislib::sys::Log::DefaultLog.WriteInfo("[SDFFont] [OpenGL Info] GL Version (integer) : %d.%d\n", major, minor);
    vislib::sys::Log::DefaultLog.WriteInfo("[SDFFont] [OpenGL Info] GLSL Version : %s\n", glslVersion);
    // Available Extensions
    //for (int i = 0; i < nExtensions; i++) {
    //    vislib::sys::Log::DefaultLog.WriteInfo("[SDFFont] [OpenGL Info] %s\n", glGetStringi(GL_EXTENSIONS, i));
    //}
    */

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
    this->fontCharacters.clear();
    this->fontKernings.clear();

    // Check file
    void   *buf = NULL;
    SIZE_T  size = 0;
    if ((size = this->loadFile(filename, &buf)) <= 0) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadfontCharacters] Could not find font information: \"%s\". \n", filename.PeekBuffer());
        ARY_SAFE_DELETE(buf);
        return false;
    }
    ARY_SAFE_DELETE(buf);

    // Load file
    vislib::sys::ASCIIFileBuffer file;
    if (!file.LoadFile(filename)) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadfontCharacters] Could not load file as ascii buffer: \"%s\". \n", filename.PeekBuffer());
        return false;
    }

    SIZE_T lineCnt = 0;
    vislib::StringA line;
    // Read info file line by line
    while (lineCnt < file.Count()) {
        line = static_cast<vislib::StringA>(file.Line(lineCnt));

         if (line.StartsWith("char ")) {
            SDFFontCharacter newChar;
            int idx = line.Find("id=", 0);
            newChar.charId   = (int)std::atoi(line.Substring(idx+3, 6)); // first parameter of substring is start (beginning with 0), second parameter is range
            idx = line.Find("x=", 0);
            newChar.texX     = (int)std::atoi(line.Substring(idx+2, 4));
            idx = line.Find("y=", 0);
            newChar.texY     = (int)std::atoi(line.Substring(idx+2, 4));
            idx = line.Find("width=", 0);
            newChar.width    = (int)std::atoi(line.Substring(idx+6, 4));
            idx = line.Find("height=", 0);
            newChar.height   = (int)std::atoi(line.Substring(idx+7, 4));
            idx = line.Find("xoffset=", 0);
            newChar.xoffset  = (int)std::atoi(line.Substring(idx+8, 4));
            idx = line.Find("yoffset=", 0);
            newChar.yoffset  = (int)std::atoi(line.Substring(idx+8, 4));
            idx = line.Find("xadvance=", 0);
            newChar.xadvance = (int)std::atoi(line.Substring(idx+9, 4));
            this->fontCharacters.push_back(newChar);
            //DEBUG std::cout << newChar.charId << " | " << newChar.texX << " | " << newChar.texY << " | " << newChar.width << " | " << newChar.height << " | " << newChar.xoffset << " | " << newChar.yoffset << " | " << newChar.xadvance << " | " << std::endl;
        }
        else if (line.StartsWith("kerning ")) {
            SDFFontKerning newKern;
            int idx = line.Find("first=", 0); 
            newKern.first = (int)std::atoi(line.Substring(idx+6, 4));
            idx = line.Find("second=", 0);
            newKern.second = (int)std::atoi(line.Substring(idx+7, 4));
            idx = line.Find("amount=", 0);
            newKern.amount = (int)std::atoi(line.Substring(idx+7, 4));
            this->fontKernings.push_back(newKern);
            //DEBUG std::cout << newKern.first << " | " << newKern.second << " | " << newKern.amount << " | " << std::endl;
        }
        // Proceed with next line ...
        lineCnt++;
    }
    //Clear ascii file buffer
    file.Clear();

    return true;
}


/*
* SDFFont::loadTexture
*/
bool SDFFont::loadFontTexture(vislib::StringA filename) {

    // Reset font texture
    if (this->texture.IsValid()) {
        this->texture.Release();
    }

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
        img.Convert(vislib::graphics::BitmapImage::TemplateByteGrayAlpha); // Using template with minimum channels containing alpha
        if (this->texture.Create(img.Width(), img.Height(), false, img.PeekDataAs<BYTE>(), GL_RG) != GL_NO_ERROR) { // Red channel is Gray value - Green channel is alpha value from png
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
* SDFFont::loadShader
*/
bool SDFFont::loadShader(vislib::StringA vert, vislib::StringA frag) {

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

        // Vertex shader attributes
        glBindAttribLocation(this->shader.ProgramHandle(), 0, "inVertPos");
        glBindAttribLocation(this->shader.ProgramHandle(), 1, "inVertTexCoord");
        // Fragment shader attributes
        glBindFragDataLocation(this->shader.ProgramHandle(), 0, "outFragColor");

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
* SDFFont::loadBuffers
*/
bool SDFFont::loadBuffers() {

    float positionData[] = {
        1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
        -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f  };

    float texData[] = {
        1.0f, 0.0f,
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f };

    // Create Vertex Array Object 
    glGenVertexArrays(1, &this->vaoHandle);
    glBindVertexArray(this->vaoHandle);

    // Create the Vertex Buffer Objects
    glGenBuffers(2, this->vboHandles);

    // Populate the position buffer
    GLuint positionBufferHandle = vboHandles[0];
    glBindBuffer(GL_ARRAY_BUFFER, positionBufferHandle);
    glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float), positionData, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); // Vertex position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLubyte *)NULL);

    // Populate the tex coord buffer
    GLuint textureBufferHandle = vboHandles[1];
    glBindBuffer(GL_ARRAY_BUFFER, textureBufferHandle);
    glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(float), texData, GL_STATIC_DRAW);
    glEnableVertexAttribArray(1); // Vertex color
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (GLubyte *)NULL);

    glBindVertexArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(1);

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

