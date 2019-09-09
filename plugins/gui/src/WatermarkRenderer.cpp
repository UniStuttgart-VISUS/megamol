/*
 * WatermarkRenderer.cpp
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "WatermarkRenderer.h"


using namespace megamol::core;
using namespace megamol::gui;


WatermarkRenderer::WatermarkRenderer(void)
    : view::Renderer3DModule_2()
    , paramAlpha("alpha", "The alpha value for the watermarks.")
    , paramScaleAll("relativeScaleAll", "The relative scale factor for all images.")
    , paramImgTopLeft("imageTopLeft", "The image file name for the top left watermark.")
    , paramScaleTopLeft("scaleTopLeft", "The scale factor for the top left watermark.")
    , paramImgTopRight("imageTopRight", "The image file name for the top right watermark.")
    , paramScaleTopRight("scaleTopRight", "The scale factor for the top right watermark.")
    , paramImgBottomLeft("imageBottomLeft", "The image file name for the bottom left watermark.")
    , paramScaleBottomLeft("scaleBottomLeft", "The scale factor for the botttom left watermark.")
    , paramImgBottomRight("imageBottomRight", "The image file name for the bottom right watermark.")
    , paramScaleBottomRight("scaleBottomRight", "The scale factor for the bottom right watermark.")
    , paramImgCenter("imageCenter", "The image file name for the center watermark.")
    , paramScaleCenter("scaleCenter", "The scale factor for the center watermark.")
    , textureTopLeft()
    , textureTopRight()
    , textureBottomLeft()
    , textureBottomRight()
    , textureCenter()
    , sizeTopLeft()
    , sizeTopRight()
    , sizeBottomLeft()
    , sizeBottomRight()
    , sizeCenter()
    , shader()
    , vaoHandle()
    , vbos() {

    // Init image file name params
    this->paramImgTopLeft.SetParameter(new param::FilePathParam(""));
    this->MakeSlotAvailable(&this->paramImgTopLeft);

    this->paramImgTopRight.SetParameter(new param::FilePathParam(""));
    this->MakeSlotAvailable(&this->paramImgTopRight);

    this->paramImgBottomLeft.SetParameter(new param::FilePathParam(""));
    this->MakeSlotAvailable(&this->paramImgBottomLeft);

    this->paramImgBottomRight.SetParameter(new param::FilePathParam(""));
    this->MakeSlotAvailable(&this->paramImgBottomRight);

    this->paramImgCenter.SetParameter(new param::FilePathParam(""));
    this->MakeSlotAvailable(&this->paramImgCenter);

    // Init scale params
    this->paramScaleAll.SetParameter(new param::FloatParam(1.0));
    this->MakeSlotAvailable(&this->paramScaleAll);

    this->paramScaleTopLeft.SetParameter(new param::FloatParam(1.0, 0.0000001f));
    this->MakeSlotAvailable(&this->paramScaleTopLeft);

    this->paramScaleTopRight.SetParameter(new param::FloatParam(1.0, 0.0000001f));
    this->MakeSlotAvailable(&this->paramScaleTopRight);

    this->paramScaleBottomLeft.SetParameter(new param::FloatParam(1.0, 0.0000001f));
    this->MakeSlotAvailable(&this->paramScaleBottomLeft);

    this->paramScaleBottomRight.SetParameter(new param::FloatParam(1.0, 0.0000001f));
    this->MakeSlotAvailable(&this->paramScaleBottomRight);

    this->paramScaleCenter.SetParameter(new param::FloatParam(1.0, 0.0000001f));
    this->MakeSlotAvailable(&this->paramScaleCenter);

    // Init alpha param
    this->paramAlpha.SetParameter(new param::FloatParam(1.0f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->paramAlpha);
}


WatermarkRenderer::~WatermarkRenderer(void) { this->Release(); }


void WatermarkRenderer::release(void) {

    // Textures
    this->textureBottomLeft.Release();
    this->textureBottomRight.Release();
    this->textureTopLeft.Release();
    this->textureTopRight.Release();
    this->textureCenter.Release();
    // Shader
    this->shader.Release();
    // VBOs
    for (unsigned int i = 0; i < (unsigned int)this->vbos.size(); i++) {
        glDeleteBuffers(1, &this->vbos[i].handle);
    }
    this->vbos.clear();
    // VAO
    glDeleteVertexArrays(1, &this->vaoHandle);
}


bool WatermarkRenderer::create(void) {

    if (!this->loadBuffers()) {
        return false;
    }
    if (!this->loadShaders()) {
        return false;
    }

    return true;
}


bool WatermarkRenderer::GetExtents(megamol::core::view::CallRender3D_2& call) {

    auto cr3d_in = &call;
    if (cr3d_in == nullptr) return false;

    // Propagate changes from outgoing CallRender3D (cr3d_out) to incoming CallRender3D (cr3d_in).
    view::CallRender3D_2* cr3d_out = this->chainRenderSlot.CallAs<view::CallRender3D_2>();
    if ((cr3d_out != nullptr) && (*cr3d_out)(core::view::AbstractCallRender::FnGetExtents)) {
        unsigned int timeFramesCount = cr3d_out->TimeFramesCount();
        cr3d_in->SetTimeFramesCount((timeFramesCount > 0) ? (timeFramesCount) : (1));
        cr3d_in->SetTime(cr3d_out->Time());
        cr3d_in->AccessBoundingBoxes() = cr3d_out->AccessBoundingBoxes();
    }

    return true;
}


bool WatermarkRenderer::Render(megamol::core::view::CallRender3D_2& call) {

    auto cr3d_in = &call;
    if (cr3d_in == nullptr) return false;

    // Update parameters ------------------------------------------------------
    if (this->paramImgTopLeft.IsDirty()) {
        this->paramImgTopLeft.ResetDirty();
        this->loadTexture(WatermarkRenderer::TOP_LEFT,
            std::string(this->paramImgTopLeft.Param<param::FilePathParam>()->Value().PeekBuffer()));
    }
    if (this->paramImgTopRight.IsDirty()) {
        this->paramImgTopRight.ResetDirty();
        this->loadTexture(WatermarkRenderer::TOP_RIGHT,
            std::string(this->paramImgTopRight.Param<param::FilePathParam>()->Value().PeekBuffer()));
    }
    if (this->paramImgBottomLeft.IsDirty()) {
        this->paramImgBottomLeft.ResetDirty();
        this->loadTexture(WatermarkRenderer::BOTTOM_LEFT,
            std::string(this->paramImgBottomLeft.Param<param::FilePathParam>()->Value().PeekBuffer()));
    }
    if (this->paramImgBottomRight.IsDirty()) {
        this->paramImgBottomRight.ResetDirty();
        this->loadTexture(WatermarkRenderer::BOTTOM_RIGHT,
            std::string(this->paramImgBottomRight.Param<param::FilePathParam>()->Value().PeekBuffer()));
    }
    if (this->paramImgCenter.IsDirty()) {
        this->paramImgCenter.ResetDirty();
        this->loadTexture(WatermarkRenderer::CENTER,
            std::string(this->paramImgCenter.Param<param::FilePathParam>()->Value().PeekBuffer()));
    }
    if (this->paramScaleAll.IsDirty()) {
        this->paramScaleAll.ResetDirty();
        float scaleAll = this->paramScaleAll.Param<param::FloatParam>()->Value();
        this->paramScaleTopLeft.Param<param::FloatParam>()->SetValue(
            this->paramScaleTopLeft.Param<param::FloatParam>()->Value() + scaleAll, false);
        this->paramScaleTopRight.Param<param::FloatParam>()->SetValue(
            this->paramScaleTopRight.Param<param::FloatParam>()->Value() + scaleAll, false);
        this->paramScaleBottomLeft.Param<param::FloatParam>()->SetValue(
            this->paramScaleBottomLeft.Param<param::FloatParam>()->Value() + scaleAll, false);
        this->paramScaleBottomRight.Param<param::FloatParam>()->SetValue(
            this->paramScaleBottomRight.Param<param::FloatParam>()->Value() + scaleAll, false);
        this->paramScaleCenter.Param<param::FloatParam>()->SetValue(
            this->paramScaleCenter.Param<param::FloatParam>()->Value() + scaleAll, false);
    }

    // First call render function of outgoing renderer ------------------------

    auto cr3d_out = this->chainRenderSlot.CallAs<view::CallRender3D_2>();
    if (cr3d_out != nullptr) {
        *cr3d_out = *cr3d_in;
        (*cr3d_out)(core::view::AbstractCallRender::FnRender);
    }

    // ...then draw watermarks ------------------------------------------------

    //// Camera
    view::Camera_2 cam;
    cr3d_out->GetCamera(cam);
    cam_type::snapshot_type snapshot;
    cam_type::matrix_type viewTemp, projTemp;
    cam.calc_matrices(snapshot, viewTemp, projTemp, thecam::snapshot_content::all);

    // Viewport
    glm::vec4 viewport;
    if (!cam.image_tile().empty()) {
        viewport = glm::vec4(
            cam.image_tile().left(), cam.image_tile().bottom(), cam.image_tile().width(), cam.image_tile().height());
    } else {
        viewport = glm::vec4(0.0f, 0.0f, cam.resolution_gate().width(), cam.resolution_gate().height());
    }
    float vpWidth = viewport[2];
    float vpHeight = viewport[3];

    // Store/Set opengl states
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDisable(GL_LIGHTING);
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);

    GLint blendSrc;
    GLint blendDst;
    glGetIntegerv(GL_BLEND_SRC, &blendSrc);
    glGetIntegerv(GL_BLEND_DST, &blendDst);
    bool blendEnabled = glIsEnabled(GL_BLEND);
    if (!blendEnabled) {
        glEnable(GL_BLEND);
    }
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Draw
    glBindVertexArray(this->vaoHandle);
    this->renderWatermark(WatermarkRenderer::TOP_LEFT, vpHeight, vpWidth);
    this->renderWatermark(WatermarkRenderer::TOP_RIGHT, vpHeight, vpWidth);
    this->renderWatermark(WatermarkRenderer::BOTTOM_LEFT, vpHeight, vpWidth);
    this->renderWatermark(WatermarkRenderer::BOTTOM_RIGHT, vpHeight, vpWidth);
    this->renderWatermark(WatermarkRenderer::CENTER, vpHeight, vpWidth);
    glBindVertexArray(0);

    // Reset opengl states
    if (!blendEnabled) {
        glDisable(GL_BLEND);
    }
    glBlendFunc(blendSrc, blendDst);

    return true;
}


bool WatermarkRenderer::renderWatermark(WatermarkRenderer::corner cor, float vpH, float vpW) {

    // Set watermark dimensions
    float imageWidth;
    float imageHeight;
    float left;
    float top;
    float bottom;
    float right;
    float scale;
    vislib::graphics::gl::OpenGLTexture2D* tex = nullptr;

    switch (cor) {
    case (WatermarkRenderer::TOP_LEFT):
        tex = &this->textureTopLeft;
        scale = this->paramScaleTopLeft.Param<param::FloatParam>()->Value();
        imageWidth = vpW * scale;
        imageHeight = imageWidth * (this->sizeTopLeft.y / this->sizeTopLeft.x);
        left = 0.0f;
        right = imageWidth;
        top = vpH;
        bottom = vpH - imageHeight;
        break;
    case (WatermarkRenderer::TOP_RIGHT):
        tex = &this->textureTopRight;
        scale = this->paramScaleTopRight.Param<param::FloatParam>()->Value();
        imageWidth = vpW * scale;
        imageHeight = imageWidth * (this->sizeTopRight.y / this->sizeTopRight.x);
        left = vpW - imageWidth;
        right = vpW;
        top = vpH;
        bottom = vpH - imageHeight;
        break;
    case (WatermarkRenderer::BOTTOM_LEFT):
        tex = &this->textureBottomLeft;
        scale = this->paramScaleBottomLeft.Param<param::FloatParam>()->Value();
        imageWidth = vpW * scale;
        imageHeight = imageWidth * (this->sizeBottomLeft.y / this->sizeBottomLeft.x);
        left = 0.0f;
        right = imageWidth;
        top = imageHeight;
        bottom = 0.0f;
        break;
    case (WatermarkRenderer::BOTTOM_RIGHT):
        tex = &this->textureBottomRight;
        scale = this->paramScaleBottomRight.Param<param::FloatParam>()->Value();
        imageWidth = vpW * scale;
        imageHeight = imageWidth * (this->sizeBottomRight.y / this->sizeBottomRight.x);
        left = vpW - imageWidth;
        right = vpW;
        top = imageHeight;
        bottom = 0.0f;
        break;
    case (WatermarkRenderer::CENTER):
        tex = &this->textureCenter;
        scale = this->paramScaleCenter.Param<param::FloatParam>()->Value();
        imageWidth = vpW * scale;
        imageHeight = imageWidth * (this->sizeCenter.y / this->sizeCenter.x);
        left = vpW / 2.0f - imageWidth / 2.0f;
        right = vpW / 2.0f + imageWidth / 2.0f;
        top = vpH / 2.0f + imageHeight / 2.0f;
        bottom = vpH / 2.0f - imageHeight / 2.0f;
        break;
    default:
        return false;
    }

    // Return if no valid texture is loaded (e.g. no filename is given)
    if (!tex->IsValid()) {
        return false;
    }

    // Scale coords to default viewport in [-1,1]
    /// XXX Move to shader?
    left = (left / vpW) * 2.0f - 1.0f;
    right = (right / vpW) * 2.0f - 1.0f;
    top = (top / vpH) * 2.0f - 1.0f;
    bottom = (bottom / vpH) * 2.0f - 1.0f;

    const size_t vertCnt = 6;
    std::vector<GLfloat> vertData;
    vertData.resize(vertCnt * this->vbos[VBOAttribs::POSITION].dim);
    std::vector<GLfloat> texData;
    texData.resize(vertCnt * this->vbos[VBOAttribs::TEXTURE].dim);

    // Triangle One (Front face => CCW)
    vertData[0] = left;   // pos X
    vertData[1] = bottom; // pos Y
    vertData[2] = -1.0f;  // pos Z
    texData[0] = 0.0f;    // tex X
    texData[1] = 1.0f;    // tex Y

    vertData[3] = right;  // pos X
    vertData[4] = bottom; // pos Y
    vertData[5] = -1.0f;  // pos Z
    texData[2] = 1.0f;    // tex X
    texData[3] = 1.0f;    // tex Y

    vertData[6] = left;  // pos X
    vertData[7] = top;   // pos Y
    vertData[8] = -1.0f; // pos Z
    texData[4] = 0.0f;   // tex X
    texData[5] = 0.0f;   // tex Y

    // Triangle Two
    vertData[9] = right;   // pos X
    vertData[10] = bottom; // pos Y
    vertData[11] = -1.0f;  // pos Z
    texData[6] = 1.0f;     // tex X
    texData[7] = 1.0f;     // tex Y

    vertData[12] = right; // pos X
    vertData[13] = top;   // pos Y
    vertData[14] = -1.0f; // pos Z
    texData[8] = 1.0f;    // tex X
    texData[9] = 0.0f;    // tex Y

    vertData[15] = left;  // pos X
    vertData[16] = top;   // pos Y
    vertData[17] = -1.0f; // pos Z
    texData[10] = 0.0f;   // tex X
    texData[11] = 0.0f;   // tex Y

    for (unsigned int i = 0; i < (unsigned int)this->vbos.size(); i++) {
        glBindBuffer(GL_ARRAY_BUFFER, this->vbos[i].handle);
        if (this->vbos[i].index == (GLuint)VBOAttribs::POSITION) {
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)vertCnt * this->vbos[i].dim * sizeof(GLfloat), vertData.data(),
                GL_STATIC_DRAW);
        } else if (this->vbos[i].index == (GLuint)VBOAttribs::TEXTURE) {
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)vertCnt * this->vbos[i].dim * sizeof(GLfloat), texData.data(),
                GL_STATIC_DRAW);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    glEnable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE0);
    tex->Bind();

    this->shader.Enable();

    glUniform1i(this->shader.ParameterLocation("tex"), 0);
    glUniform1f(this->shader.ParameterLocation("alpha"),
        static_cast<GLfloat>(this->paramAlpha.Param<param::FloatParam>()->Value()));

    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)vertCnt);

    this->shader.Disable();

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);

    return true;
}


bool WatermarkRenderer::loadBuffers(void) {

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
    VBOData newVBO;

    // VBO for position data
    newVBO.name = "inVertPos";
    newVBO.index = (GLuint)VBOAttribs::POSITION;
    newVBO.dim = 3;
    newVBO.handle = 0; // default init
    this->vbos.push_back(newVBO);

    // VBO for texture coordinates
    newVBO.name = "inTexCoord";
    newVBO.index = (GLuint)VBOAttribs::TEXTURE;
    newVBO.dim = 2;
    newVBO.handle = 0; // default init
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


bool WatermarkRenderer::loadTexture(WatermarkRenderer::corner cor, std::string filename) {

    vislib::graphics::gl::OpenGLTexture2D* tex = nullptr;
    glm::vec2* texSize = nullptr;

    switch (cor) {
    case (WatermarkRenderer::TOP_LEFT):
        tex = &this->textureTopLeft;
        texSize = &this->sizeTopLeft;
        break;
    case (WatermarkRenderer::TOP_RIGHT):
        tex = &this->textureTopRight;
        texSize = &this->sizeTopRight;
        break;
    case (WatermarkRenderer::BOTTOM_LEFT):
        tex = &this->textureBottomLeft;
        texSize = &this->sizeBottomLeft;
        break;
    case (WatermarkRenderer::BOTTOM_RIGHT):
        tex = &this->textureBottomRight;
        texSize = &this->sizeBottomRight;
        break;
    case (WatermarkRenderer::CENTER):
        tex = &this->textureCenter;
        texSize = &this->sizeCenter;
        break;
    default:
        return false;
    }

    if (tex->IsValid()) {
        tex->Release();
    }

    static vislib::graphics::BitmapImage img;
    static sg::graphics::PngBitmapCodec pbc;
    pbc.Image() = &img;
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    void* buf = nullptr;
    SIZE_T size = 0;

    if ((size = this->loadFile(filename, &buf)) > 0) {
        if (pbc.Load(buf, size)) {
            img.Convert(vislib::graphics::BitmapImage::TemplateByteRGBA);
            texSize->x = static_cast<float>(img.Width());
            texSize->y = static_cast<float>(img.Height());
            if (tex->Create(img.Width(), img.Height(), false, img.PeekDataAs<BYTE>(), GL_RGBA) != GL_NO_ERROR) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "[WatermarkRenderer] Could not load \"%s\" texture.", filename.c_str());
                ARY_SAFE_DELETE(buf);
                return false;
            }
            tex->Bind();
            glGenerateMipmap(GL_TEXTURE_2D);
            tex->SetFilter(GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR);
            tex->SetWrap(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
            glBindTexture(GL_TEXTURE_2D, 0);

            ARY_SAFE_DELETE(buf);
        } else {
            vislib::sys::Log::DefaultLog.WriteError(
                "[WatermarkRenderer] Could not read \"%s\" texture.", filename.c_str());
            ARY_SAFE_DELETE(buf);
            return false;
        }
    } else {
        ARY_SAFE_DELETE(buf);
        return false;
    }

    return true;
}


bool WatermarkRenderer::loadShaders(void) {

    // Reset shader
    this->shader.Release();

    vislib::graphics::gl::ShaderSource vert, frag;
    try {
        if (!megamol::core::Module::instance()->ShaderSourceFactory().MakeShaderSource("watermark::vertex", vert)) {
            return false;
        }
        if (!megamol::core::Module::instance()->ShaderSourceFactory().MakeShaderSource("watermark::fragment", frag)) {
            return false;
        }
        if (!this->shader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            vislib::sys::Log::DefaultLog.WriteError("[WatermarkRenderer] Unable to compile shader: Unknown error\n");
            return false;
        }
    } catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteError("[WatermarkRenderer] Unable to compile shader (@%s): %s\n",
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA());
        return false;
    } catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteError("[WatermarkRenderer] Unable to compile shader: %s\n", e.GetMsgA());
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("[WatermarkRenderer] Unable to compile shader: Unknown exception\n");
        return false;
    }

    return true;
}


SIZE_T WatermarkRenderer::loadFile(std::string name, void** outData) {

    *outData = nullptr;

    vislib::StringW filename = static_cast<vislib::StringW>(name.c_str());
    if (filename.IsEmpty()) {
        vislib::sys::Log::DefaultLog.WriteError("[WatermarkRenderer] Unable to load file: No filename given\n");
        return 0;
    }

    if (!vislib::sys::File::Exists(filename)) {
        vislib::sys::Log::DefaultLog.WriteError(
            "[WatermarkRenderer] Unable to load file \"%s\": Not existing\n", name.c_str());
        return 0;
    }

    SIZE_T size = static_cast<SIZE_T>(vislib::sys::File::GetSize(filename));
    if (size < 1) {
        vislib::sys::Log::DefaultLog.WriteError(
            "[WatermarkRenderer] Unable to load file \"%s\": File is empty\n", name.c_str());
        return 0;
    }

    vislib::sys::FastFile f;
    if (!f.Open(filename, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        vislib::sys::Log::DefaultLog.WriteError(
            "[WatermarkRenderer] Unable to load file \"%s\": Cannot open file\n", name.c_str());
        return 0;
    }

    *outData = new BYTE[size];
    SIZE_T num = static_cast<SIZE_T>(f.Read(*outData, size));
    if (num != size) {
        vislib::sys::Log::DefaultLog.WriteError(
            "[WatermarkRenderer] Unable to load file \"%s\": Cannot read whole file\n", name.c_str());
        ARY_SAFE_DELETE(*outData);
        return 0;
    }

    return num;
}
