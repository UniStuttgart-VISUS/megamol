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


using namespace vislib;

using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::view;
using namespace megamol::core::view::special;


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const BitmapFont bmf) : AbstractFont(),
    renderType(SDFFont::RENDERTYPE_FILL), texture(), shader(), fontInfo() {

    this->loadFont(bmf);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const BitmapFont bmf,  SDFFont::RenderType render) : AbstractFont(), 
    font(bmf), renderType(render), texture(), shader(), fontInfo() {

    this->loadFont(bmf);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const BitmapFont bmf, float size)  : AbstractFont(), 
    font(bmf), renderType(SDFFont::RENDERTYPE_FILL), texture(), shader(), fontInfo() {

    this->SetSize(size);

    this->loadFont(bmf);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const BitmapFont bmf, bool flipY) : AbstractFont(), 
    font(bmf), renderType(SDFFont::RENDERTYPE_FILL), texture(), shader(), fontInfo() {

    this->SetFlipY(flipY);

    this->loadFont(bmf);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const BitmapFont bmf, SDFFont::RenderType render, bool flipY) : AbstractFont(),
    font(bmf), renderType(render), texture(), shader(), fontInfo() {

    this->SetFlipY(flipY);

    this->loadFont(bmf);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const BitmapFont bmf, float size, bool flipY) : AbstractFont(), 
    font(bmf), renderType(SDFFont::RENDERTYPE_FILL), texture(), shader(), fontInfo() {

    this->SetSize(size);
    this->SetFlipY(flipY);

    this->loadFont(bmf);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const BitmapFont bmf, float size, SDFFont::RenderType render) : AbstractFont(), 
    font(bmf), renderType(render), texture(), shader(), fontInfo() {

    this->SetSize(size);

    this->loadFont(bmf);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const BitmapFont bmf, float size, SDFFont::RenderType render, bool flipY) : AbstractFont(),
        font(bmf), renderType(render), texture(), shader(), fontInfo() {

    this->SetSize(size);
    this->SetFlipY(flipY);

    this->loadFont(bmf);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src) : AbstractFont(),
    font(src.font), renderType(src.renderType), texture(), shader(), fontInfo() {

    this->SetSize(src.GetSize());
    this->SetFlipY(src.IsFlipY());

    this->loadFont(src.font);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src, SDFFont::RenderType render) : AbstractFont(), 
    font(src.font), renderType(render), texture(), shader(), fontInfo() {

    this->SetSize(src.GetSize());
    this->SetFlipY(src.IsFlipY());

    this->loadFont(src.font);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src, float size) : AbstractFont(),
        font(src.font), renderType(src.renderType), texture(), shader(), fontInfo() {

    this->SetSize(size);
    this->SetFlipY(src.IsFlipY());

    this->loadFont(src.font);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src, bool flipY) : AbstractFont(),
        font(src.font), renderType(src.renderType), texture(), shader(), fontInfo() {

    this->SetSize(src.GetSize());
    this->SetFlipY(flipY);

    this->loadFont(src.font);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src, SDFFont::RenderType render, bool flipY) : AbstractFont(),
        font(src.font), renderType(render), texture(), shader(), fontInfo() {

    this->SetSize(src.GetSize());
    this->SetFlipY(flipY);

    this->loadFont(src.font);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src, float size, bool flipY) : AbstractFont(), 
    font(src.font), renderType(src.renderType), texture(), shader(), fontInfo() {

    this->SetSize(size);
    this->SetFlipY(flipY);

    this->loadFont(src.font);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src, float size,  SDFFont::RenderType render) : AbstractFont(), 
    font(src.font),  renderType(render), texture(), shader(), fontInfo() {

    this->SetSize(size);
    this->SetFlipY(src.IsFlipY());

    this->loadFont(src.font);
}


/*
 * SDFFont::SDFFont
 */
SDFFont::SDFFont(const SDFFont& src, float size, SDFFont::RenderType render, bool flipY) : AbstractFont(),
        font(src.font), renderType(render), texture(), shader(), fontInfo() {

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

    return this->lineCount(vislib::StringA(txt));
}


/*
 * SDFFont::BlockLines
 */
unsigned int SDFFont::BlockLines(float maxWidth, float size, const wchar_t *txt) const {

    return this->lineCount(vislib::StringW(txt));
}


/*
 * SDFFont::DrawString
 */
void SDFFont::DrawString(float x, float y, float size, bool flipY, const char *txt, AbstractFont::Alignment align) const {






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



    return w;
}


/*
 * SDFFont::LineWidth
 */
float SDFFont::LineWidth(float size, const wchar_t *txt) const {

    float w = 0.0f;



    return w;
}


/*
 * SDFFont::lineCount
 */
unsigned int SDFFont::lineCount(vislib::StringA txt) const {

    unsigned int i = 0;

    


    return i;
   
}

/*
* SDFFont::lineCount
*/
unsigned int SDFFont::lineCount(vislib::StringW txt) const {

    unsigned int i = 0;




    return i;

}

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

    this->shader.Release();
    this->texture.Release();
}


/*
* SDFFont::draw
*/
void SDFFont::draw(vislib::StringA *txt, float x, float y, float z, float size, bool flipY, Alignment align) const {








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

    vislib::StringA filename = ".\\";
    filename.Append(fontName);

    vislib::StringA infoFile = filename;
    infoFile.Append(".fnt");

    vislib::StringA textureFile = filename;
    textureFile.Append(".png");

    if (!this->loadFontInfo(infoFile)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "[SDFFont] [loadFont] ...\n");
        return false;
    }
    else {
        if (!this->loadFontTexture(textureFile)) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "[SDFFont] [loadFont] ...\n");
            return false;
        }
        else {
            if (!this->loadShader()) {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "[SDFFont] [loadFont] ...\n");
                return false;
            }
        }
    }



    return true;
}


/*
* SDFFont::loadFontInfo
*/
bool SDFFont::loadFontInfo(vislib::StringA filename) {

    // Reset font info
    this->fontInfo.Clear();







    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "[SDFFont] [loadFontInfo] ...\n");
    return false;


    return true;

}


/*
* SDFFont::loadShader
*/
bool SDFFont::loadShader() {

    // Reset shader
    this->shader.Release();


    const char *shaderName = "SDFFont Shader";

    vislib::StringA vertShader = "\
void main(void) { \
gl_Position = VertexPosition; \
} \
";

    vislib::StringA fragShader = "\
void main(void) { \
    FragColor = vec4(0.0, 0.5, 0.0, 1.0); \
} \
";

    try {
        const char *vertStr   = vertShader.PeekBuffer();
        const char **vertCode = &(vertStr);
        const char *fragStr   = fragShader.PeekBuffer();
        const char **fragCode = &(fragStr);

        if (!this->shader.Create(vertCode, vertShader.Length(), fragCode, fragShader.Length())) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "[SDFFont] [loadShader]Unable to create %s: Unknown error\n", shaderName);
            return false;
        }
    }
    catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "[SDFFont] [loadShader]Unable to compile %s shader (@%s): %s\n", shaderName,
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()), ce.GetMsgA());
        return false;
    }
    catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "[SDFFont] [loadShader]Unable to compile %s shader: %s\n", shaderName, e.GetMsgA());
        return false;
    }
    catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "[SDFFont] [loadShader] Unable to compile %s shader: Unknown exception\n", shaderName);
        return false;
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
    void *buf = NULL;
    SIZE_T size = 0;

    // Loading file
    vislib::StringW name = static_cast<vislib::StringW>(filename);
    if (name.IsEmpty()) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadFile] Unable to load file: No name given.\n");
        return false;
    }
    if (!vislib::sys::File::Exists(name)) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadFile] Unable to load file \"%s\": Not existing.\n", name.PeekBuffer());
        return false;
    }
    size = static_cast<SIZE_T>(vislib::sys::File::GetSize(name));
    if (size < 1) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadFile] Unable to load file \"%s\": File is empty.\n", name.PeekBuffer());
        return false;
    }
    vislib::sys::FastFile f;
    if (!f.Open(name, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadFile] Unable to load file \"%s\": Cannot open file.\n", name.PeekBuffer());
        return false;
    }
    buf = new BYTE[size];
    SIZE_T num = static_cast<SIZE_T>(f.Read(buf, size));
    if (num != size) {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadFile] Unable to load file \"%s\": Cannot read whole file.\n", name.PeekBuffer());
        ARY_SAFE_DELETE(buf);
        return false;
    }

    if ((size = num) > 0) {
        if (pbc.Load(buf, size)) {
            img.Convert(vislib::graphics::BitmapImage::TemplateByteRGBA);
            if (this->texture.Create(img.Width(), img.Height(), false, img.PeekDataAs<BYTE>(), GL_RGBA) != GL_NO_ERROR) {
                vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadTexture] Could not load \"%s\" texture.", filename.PeekBuffer());
                ARY_SAFE_DELETE(buf);
                return false;
            }
            this->texture.Bind();
            glBindTexture(GL_TEXTURE_2D, 0);
            this->texture.SetFilter(GL_LINEAR, GL_LINEAR);
            this->texture.SetWrap(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
            ARY_SAFE_DELETE(buf);
        }
        else {
            vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadTexture] Could not read \"%s\" texture.", filename.PeekBuffer());
            return false;
        }
    }
    else {
        vislib::sys::Log::DefaultLog.WriteError("[SDFFont] [loadTexture] Could not find \"%s\" texture.", filename.PeekBuffer());
        return false;
    }

    return true;
}

