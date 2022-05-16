/*
 * TriSoupRenderer.cpp
 *
 * Copyright (C) 2010 by Sebastian Grottel
 * Copyright (C) 2008-2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "ProteinViewRenderer.h"
#include "geometry_calls_gl/CallTriMeshDataGL.h"
#include "image_calls/Image2DCall.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore_gl/view/CallRender3DGL.h"
#include "vislib_gl/graphics/gl/IncludeAllGL.h"

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/utility/sys/MemmappedFile.h"
#include "vislib/math/ShallowPoint.h"
#include "vislib/math/Vector.h"
#include "vislib/math/mathfunctions.h"
#include <filesystem>

using namespace megamol;
using namespace megamol::molsurfmapcluster;

using namespace megamol::core;


/*
 * ProteinViewRenderer::ProteinViewRenderer
 */
ProteinViewRenderer::ProteinViewRenderer(void)
        : Renderer3DModuleGL()
        , getDataSlot("getData", "The slot to fetch the tri-mesh data")
        , getImageDataSlot("getImageData", "The slot to fetch the image data")
        , showVertices("showVertices", "Flag whether to show the verices of the object")
        , lighting("lighting", "Flag whether or not use lighting for the surface")
        , windRule("windingrule", "The triangle edge winding rule")
        , nameSlot("name", "The name of the displayed protein mesh")
        , colorSlot("color", "The triangle color (if no colors are read from file)")
        , theFont(megamol::core::utility::SDFFont::PRESET_ROBOTO_SANS)
        , texVa(0)
        , lastHash(0) {

    this->getDataSlot.SetCompatibleCall<megamol::geocalls_gl::CallTriMeshDataGLDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->getImageDataSlot.SetCompatibleCall<image_calls::Image2DCallDescription>();
    this->MakeSlotAvailable(&this->getImageDataSlot);

    this->nameSlot.SetParameter(new param::StringParam(""));
    this->MakeSlotAvailable(&this->nameSlot);

    this->showVertices.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->showVertices);

    this->lighting.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->lighting);

    param::EnumParam* ep = new param::EnumParam(0);
    ep = new param::EnumParam(0);
    ep->SetTypePair(0, "Counter-Clock Wise");
    ep->SetTypePair(1, "Clock Wise");
    this->windRule << ep;
    this->MakeSlotAvailable(&this->windRule);

    this->colorSlot.SetParameter(new param::ColorParam(1.0f, 1.0f, 1.0f, 1.0f));
    this->MakeSlotAvailable(&this->colorSlot);
}


/*
 * ProteinViewRenderer::~ProteinViewRenderer
 */
ProteinViewRenderer::~ProteinViewRenderer(void) {
    this->Release();
}


/*
 * ProteinViewRenderer::create
 */
bool ProteinViewRenderer::create(void) {
    // Initialise font
    if (!this->theFont.Initialise(this->GetCoreInstance())) {
        core::utility::log::Log::DefaultLog.WriteError("Couldn't initialize the font.");
        return false;
    }

    vislib_gl::graphics::gl::ShaderSource texVertShader;
    vislib_gl::graphics::gl::ShaderSource texFragShader;

    auto ssf = std::make_shared<core_gl::utility::ShaderSourceFactory>(instance()->Configuration().ShaderDirectories());
    if (!ssf->MakeShaderSource("molsurfMapOverlay::vertex", texVertShader)) {
        core::utility::log::Log::DefaultLog.WriteMsg(
            core::utility::log::Log::LEVEL_ERROR, "Unable to load vertex shader source for texture Vertex Shader");
        return false;
    }
    if (!ssf->MakeShaderSource("molsurfMapOverlay::fragment", texFragShader)) {
        core::utility::log::Log::DefaultLog.WriteMsg(
            core::utility::log::Log::LEVEL_ERROR, "Unable to load fragment shader source for texture Fragment Shader");
        return false;
    }

    try {
        if (!this->textureShader.Create(
                texVertShader.Code(), texVertShader.Count(), texFragShader.Code(), texFragShader.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        core::utility::log::Log::DefaultLog.WriteError("Unable to create shader: %s\n", e.GetMsgA());
        return false;
    }

    texVertShader.Clear();
    texFragShader.Clear();

    const float size = 1.0f;
    std::vector<float> texVerts = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, size, 0.0f, 0.0f, 1.0f, size, 0.0f, 0.0f, 1.0f,
        0.0f, size, size, 0.0f, 1.0f, 1.0f};

    this->texBuffer = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, texVerts, GL_STATIC_DRAW);

    glGenVertexArrays(1, &this->texVa);
    glBindVertexArray(this->texVa);

    this->texBuffer->bind();
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), 0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

    return true;
}


/*
 * ProteinViewRenderer::GetExtents
 */
bool ProteinViewRenderer::GetExtents(core_gl::view::CallRender3DGL& call) {

    megamol::geocalls_gl::CallTriMeshDataGL* ctmd = this->getDataSlot.CallAs<megamol::geocalls_gl::CallTriMeshDataGL>();
    if (ctmd == NULL)
        return false;
    ctmd->SetFrameID(static_cast<int>(call.Time()));
    if (!(*ctmd)(1))
        return false;

    call.SetTimeFramesCount(ctmd->FrameCount());
    call.AccessBoundingBoxes().Clear();
    call.AccessBoundingBoxes() = ctmd->AccessBoundingBoxes();

    return true;
}


/*
 * ProteinViewRenderer::release
 */
void ProteinViewRenderer::release(void) {
    // intentionally empty
}


/*
 * TriSoupRenderer::Render
 */
bool ProteinViewRenderer::Render(core_gl::view::CallRender3DGL& call) {
    megamol::geocalls_gl::CallTriMeshDataGL* ctmd = this->getDataSlot.CallAs<megamol::geocalls_gl::CallTriMeshDataGL>();
    if (ctmd == NULL)
        return false;

    ctmd->SetFrameID(static_cast<int>(call.Time()));
    if (!(*ctmd)(1))
        return false;

    ctmd->SetFrameID(static_cast<int>(call.Time())); // necessary?
    if (!(*ctmd)(0))
        return false;

    core::view::Camera cam = call.GetCamera();
    glm::mat4 proj = cam.getProjectionMatrix();
    glm::mat4 view = cam.getViewMatrix();
    glm::mat4 mvp = proj * view;
    glm::vec3 viewdir = cam.getPose().direction;

    auto viewport = call.GetViewResolution();
    this->m_viewport = {viewport.x, viewport.y};

    glm::mat4 ortho = glm::ortho(
        0.0f, static_cast<float>(this->m_viewport.x), 0.0f, static_cast<float>(this->m_viewport.y), -1.0f, 1.0f);

#if 1
    // lighting setup
    //this->GetLights();
    glm::vec4 lightPos = {0.0f, 0.0f, 0.0f, 1.0f};
    // TODO read lights correctly
    //if (this->lightMap.size() != 1) {
    //    // megamol::core::utility::log::Log::DefaultLog.WriteWarn(
    //    //    "[TriSoupRenderer] Only one single point light source is supported by this renderer");
    //}
    //for (auto light : this->lightMap) {
    //    if (light.second.lightType != core::view::light::POINTLIGHT) {
    //        // megamol::core::utility::log::Log::DefaultLog.WriteWarn(
    //        //    "[TriSoupRenderer] Only single point light source is supported by this renderer");
    //    } else {
    //        auto lPos = light.second.pl_position;
    //        // light.second.lightColor;
    //        // light.second.lightIntensity;
    //        if (lPos.size() == 3) {
    //            lightPos[0] = lPos[0];
    //            lightPos[1] = lPos[1];
    //            lightPos[2] = lPos[2];
    //        }
    //        if (lPos.size() == 4) {
    //            lightPos[0] = lPos[0];
    //            lightPos[1] = lPos[1];
    //            lightPos[2] = lPos[2];
    //            lightPos[3] = lPos[3];
    //        }
    //        break;
    //    }
    //}
    glm::vec4 zeros(0.f);
    glm::vec4 ambient(0.2f, 0.2f, 0.2f, 1.f);
    glm::vec4 diffuse(1.f, 1.f, 1.f, 0.f);
    glm::vec4 specular(0.f, 0.f, 0.f, 0.f);

    ::glMatrixMode(GL_PROJECTION);
    ::glPushMatrix();
    ::glLoadMatrixf(glm::value_ptr(proj));

    ::glMatrixMode(GL_MODELVIEW);
    ::glPushMatrix();
    ::glLoadMatrixf(glm::value_ptr(view));

    bool normals = false;
    bool colors = false;
    bool textures = false;
    ::glEnable(GL_DEPTH_TEST);
    bool doLighting = this->lighting.Param<param::BoolParam>()->Value();
    if (doLighting) {
        ::glEnable(GL_LIGHTING);
        ::glEnable(GL_LIGHT0);
        ::glLightfv(GL_LIGHT0, GL_POSITION, glm::value_ptr(lightPos));
        ::glLightfv(GL_LIGHT0, GL_AMBIENT, glm::value_ptr(ambient));
        ::glLightfv(GL_LIGHT0, GL_DIFFUSE, glm::value_ptr(diffuse));
        ::glLightfv(GL_LIGHT0, GL_SPECULAR, glm::value_ptr(specular));
    } else {
        ::glLightfv(GL_LIGHT0, GL_POSITION, glm::value_ptr(zeros));
        ::glLightfv(GL_LIGHT0, GL_AMBIENT, glm::value_ptr(zeros));
        ::glLightfv(GL_LIGHT0, GL_DIFFUSE, glm::value_ptr(zeros));
        ::glLightfv(GL_LIGHT0, GL_SPECULAR, glm::value_ptr(zeros));
        ::glDisable(GL_LIGHT0);
        ::glDisable(GL_LIGHTING);
    }
    ::glDisable(GL_BLEND);
    ::glEnableClientState(GL_VERTEX_ARRAY);
    ::glDisableClientState(GL_NORMAL_ARRAY);
    ::glDisableClientState(GL_COLOR_ARRAY);
    ::glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    ::glEnable(GL_COLOR_MATERIAL);
    ::glEnable(GL_TEXTURE_2D);
    ::glBindTexture(GL_TEXTURE_2D, 0);
    ::glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
    ::glEnable(GL_NORMALIZE);

    // GLsizei** oldbind = new GLsizei*[16];
    // for (int i = 0; i < 16; i++) {
    //    oldbind[i] = new GLsizei();
    //}
    ////glGetVertexAttribPointerv(0, GL_VERTEX_ARRAY_POINTER, (GLvoid**)oldbind);
    // glGetPointerv(GL_VERTEX_ARRAY_POINTER, (GLvoid**)oldbind);
    glVertexAttribPointer(0, 0, GL_FLOAT, GL_FALSE, 0, nullptr);
    // glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &oldbind);
    // glBindBuffer(GL_ARRAY_BUFFER, 0);
    // glBindVertexArray(0);


    GLint cfm;
    ::glGetIntegerv(GL_CULL_FACE_MODE, &cfm);
    GLint pm[2];
    ::glGetIntegerv(GL_POLYGON_MODE, pm);
    GLint twr;
    ::glGetIntegerv(GL_FRONT_FACE, &twr);

    if (this->windRule.Param<param::EnumParam>()->Value() == 0) {
        ::glFrontFace(GL_CCW);
    } else {
        ::glFrontFace(GL_CW);
    }

    int fpm, bpm;
    fpm = GL_FILL;
    bpm = GL_FILL;

    ::glPolygonMode(GL_FRONT, fpm);
    ::glPolygonMode(GL_BACK, bpm);
    ::glDisable(GL_CULL_FACE);

    //::glColor3f(1.0f, 1.0f, 1.0f);
    float r, g, b;
    this->colorSlot.ResetDirty();
    auto curcol = this->colorSlot.Param<param::ColorParam>()->Value();
    ::glColor3fv(curcol.data());

    for (unsigned int i = 0; i < ctmd->Count(); i++) {
        const megamol::geocalls_gl::CallTriMeshDataGL::Mesh& obj = ctmd->Objects()[i];

        switch (obj.GetVertexDataType()) {
        case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_FLOAT:
            ::glVertexPointer(3, GL_FLOAT, 0, obj.GetVertexPointerFloat());
            break;
        case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_DOUBLE:
            ::glVertexPointer(3, GL_DOUBLE, 0, obj.GetVertexPointerDouble());
            break;
        default:
            continue;
        }

        if (obj.HasNormalPointer() != NULL) {
            if (!normals) {
                ::glEnableClientState(GL_NORMAL_ARRAY);
                normals = true;
            }
            switch (obj.GetNormalDataType()) {
            case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_FLOAT:
                ::glNormalPointer(GL_FLOAT, 0, obj.GetNormalPointerFloat());
                break;
            case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_DOUBLE:
                ::glNormalPointer(GL_DOUBLE, 0, obj.GetNormalPointerDouble());
                break;
            default:
                continue;
            }
        } else if (normals) {
            ::glDisableClientState(GL_NORMAL_ARRAY);
            normals = false;
        }

        if (obj.HasColourPointer() != NULL) {
            if (!colors) {
                ::glEnableClientState(GL_COLOR_ARRAY);
                colors = true;
            }
            switch (obj.GetColourDataType()) {
            case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_BYTE:
                ::glColorPointer(3, GL_UNSIGNED_BYTE, 0, obj.GetColourPointerByte());
                break;
            case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_FLOAT:
                ::glColorPointer(3, GL_FLOAT, 0, obj.GetColourPointerFloat());
                break;
            case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_DOUBLE:
                ::glColorPointer(3, GL_DOUBLE, 0, obj.GetColourPointerDouble());
                break;
            default:
                continue;
            }
        } else if (colors) {
            ::glDisableClientState(GL_COLOR_ARRAY);
            colors = false;
        }

        if (obj.HasTextureCoordinatePointer() != NULL) {
            if (!textures) {
                ::glEnableClientState(GL_TEXTURE_COORD_ARRAY);
                textures = true;
            }
            switch (obj.GetTextureCoordinateDataType()) {
            case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_FLOAT:
                ::glTexCoordPointer(2, GL_FLOAT, 0, obj.GetTextureCoordinatePointerFloat());
                break;
            case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_DOUBLE:
                ::glTexCoordPointer(2, GL_DOUBLE, 0, obj.GetTextureCoordinatePointerDouble());
                break;
            default:
                continue;
            }
        } else if (textures) {
            ::glDisableClientState(GL_TEXTURE_COORD_ARRAY);
            textures = false;
        }

        if (obj.GetMaterial() != NULL) {
            const megamol::geocalls_gl::CallTriMeshDataGL::Material& mat = *obj.GetMaterial();

            if (doLighting) {
                ::glDisable(GL_COLOR_MATERIAL);
                GLfloat mat_ambient[4] = {mat.GetKa()[0], mat.GetKa()[1], mat.GetKa()[2], 1.0f};
                GLfloat mat_diffuse[4] = {mat.GetKd()[0], mat.GetKd()[1], mat.GetKd()[2], 1.0f};
                GLfloat mat_specular[4] = {mat.GetKs()[0], mat.GetKs()[1], mat.GetKs()[2], 1.0f};
                GLfloat mat_emission[4] = {mat.GetKe()[0], mat.GetKe()[1], mat.GetKe()[2], 1.0f};
                GLfloat mat_shininess[1] = {mat.GetNs()};
                ::glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, mat_ambient);
                ::glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);
                ::glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
                ::glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, mat_emission);
                ::glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);
            } else {
                ::glColor3f(mat.GetKd()[0], mat.GetKd()[1], mat.GetKd()[2]);
            }

            GLuint mapid = mat.GetMapID();
            if (mapid > 0) {
                //::glActiveTexture(GL_TEXTURE0);
                ::glEnable(GL_COLOR_MATERIAL);
                ::glBindTexture(GL_TEXTURE_2D, mapid);
                ::glEnable(GL_TEXTURE_2D);
                ::glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
                ::glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            }
        } else {
            GLfloat mat_ambient[4] = {0.2f, 0.2f, 0.2f, 1.0f};
            GLfloat mat_diffuse[4] = {0.8f, 0.8f, 0.8f, 1.0f};
            GLfloat mat_specular[4] = {0.0f, 0.0f, 0.0f, 1.0f};
            GLfloat mat_emission[4] = {0.0f, 0.0f, 0.0f, 1.0f};
            GLfloat mat_shininess[1] = {0.0f};
            ::glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, mat_ambient);
            ::glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);
            ::glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
            ::glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, mat_emission);
            ::glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);
            ::glBindTexture(GL_TEXTURE_2D, 0);
            ::glEnable(GL_COLOR_MATERIAL);
        }

        if (obj.HasTriIndexPointer() != NULL) {
            switch (obj.GetTriDataType()) {
            case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_BYTE:
                ::glDrawElements(GL_TRIANGLES, obj.GetTriCount() * 3, GL_UNSIGNED_BYTE, obj.GetTriIndexPointerByte());
                break;
            case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_UINT16:
                ::glDrawElements(
                    GL_TRIANGLES, obj.GetTriCount() * 3, GL_UNSIGNED_SHORT, obj.GetTriIndexPointerUInt16());
                break;
            case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_UINT32:
                ::glDrawElements(GL_TRIANGLES, obj.GetTriCount() * 3, GL_UNSIGNED_INT, obj.GetTriIndexPointerUInt32());
                break;
            default:
                continue;
            }
        } else {
            ::glDrawArrays(GL_TRIANGLES, 0, obj.GetVertexCount());
        }

        if (!doLighting) {
            ::glColor3f(r, g, b);
        }
    }

    if (normals)
        ::glDisableClientState(GL_NORMAL_ARRAY);
    if (colors)
        ::glDisableClientState(GL_COLOR_ARRAY);
    if (textures)
        ::glDisableClientState(GL_TEXTURE_COORD_ARRAY);

    {
        GLfloat mat_ambient[4] = {0.2f, 0.2f, 0.2f, 1.0f};
        GLfloat mat_diffuse[4] = {0.8f, 0.8f, 0.8f, 1.0f};
        GLfloat mat_specular[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        GLfloat mat_emission[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        GLfloat mat_shininess[1] = {0.0f};
        ::glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, mat_ambient);
        ::glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);
        ::glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
        ::glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, mat_emission);
        ::glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);
        ::glBindTexture(GL_TEXTURE_2D, 0);
    }

    ::glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    if (this->showVertices.Param<param::BoolParam>()->Value()) {
        //::glEnable(GL_POINT_SIZE);
        ::glPointSize(3.0f);
        ::glDisable(GL_LIGHTING);

        ::glColor3f(1.0f, 0.0f, 0.0f);
        for (unsigned int i = 0; i < ctmd->Count(); i++) {
            switch (ctmd->Objects()[i].GetVertexDataType()) {
            case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_FLOAT:
                ::glVertexPointer(3, GL_FLOAT, 0, ctmd->Objects()[i].GetVertexPointerFloat());
                break;
            case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_DOUBLE:
                ::glVertexPointer(3, GL_DOUBLE, 0, ctmd->Objects()[i].GetVertexPointerDouble());
                break;
            default:
                continue;
            }
            ::glDrawArrays(GL_POINTS, 0, ctmd->Objects()[i].GetVertexCount());
        }

        //::glEnable(GL_POINT_SIZE);
        ::glPointSize(1.0f);
    }

    ::glCullFace(cfm);
    ::glFrontFace(twr);
    ::glPolygonMode(GL_FRONT, pm[0]);
    ::glPolygonMode(GL_BACK, pm[1]);

    ::glEnable(GL_CULL_FACE);
    ::glDisableClientState(GL_VERTEX_ARRAY);
    //::glDisable(GL_POINT_SIZE);
    ::glEnable(GL_BLEND);

#if (defined(_MSC_VER) && (_MSC_VER > 1000))
    ::GetLastError();
#endif
    ::glCullFace(cfm);
    ::glFrontFace(twr);
    ::glDisableClientState(GL_VERTEX_ARRAY);

    ::glMatrixMode(GL_PROJECTION);
    ::glPopMatrix();
    ::glMatrixMode(GL_MODELVIEW);
    ::glPopMatrix();

#endif

    // TODO bind texture


    if (!this->nameSlot.Param<param::StringParam>()->Value().empty()) {
        std::string name = this->nameSlot.Param<param::StringParam>()->Value();
        image_calls::Image2DCall* cppIn = this->getImageDataSlot.CallAs<image_calls::Image2DCall>();
        if (!(*cppIn)(image_calls::Image2DCall::CallForGetMetaData))
            return true;
        if (lastHash != cppIn->DataHash() || this->nameSlot.IsDirty()) {
            lastHash = cppIn->DataHash();
            this->nameSlot.ResetDirty();
            if (!(*cppIn)(image_calls::Image2DCall::CallForWaitForData))
                return false;
            if (!(*cppIn)(image_calls::Image2DCall::CallForGetData))
                return false;
            if (!(*cppIn)(image_calls::Image2DCall::CallForWaitForData))
                return false;

            auto immap = cppIn->GetImagePtr();
            bool found = false;
            std::string result = "";
            if (immap != nullptr) {
                for (auto v : *immap) {
                    std::filesystem::path path = v.first;
                    std::string pname = path.stem().string();
                    if (pname.size() > 4) {
                        pname = pname.substr(0, 4);
                    }
                    if (name.compare(pname) == 0) { // strings are equal
                        found = true;
                        result = v.first;
                        break;
                    }
                }
#if 1
                if (found) {
                    auto& im = immap->at(result);
                    auto err = glGetError(); // flush errors
                    glowl::TextureLayout layout(GL_RGB8, im.Width(), im.Height(), 1, GL_RGB, GL_UNSIGNED_BYTE, 1);
                    this->texture = std::make_unique<glowl::Texture2D>("", layout, im.PeekDataAs<BYTE>());
                }
#endif
            }
        }


        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        this->texture->bindTexture();

        glBindVertexArray(this->texVa);
        this->textureShader.Enable();

        ortho = glm::mat4(1.0f);

        glUniform2f(this->textureShader.ParameterLocation("lowerleft"), 0.25f, 0.25f);
        glUniform2f(this->textureShader.ParameterLocation("upperright"), 0.95f, 0.95f);
        glUniform3f(this->textureShader.ParameterLocation("viewvec"), viewdir.x, viewdir.y, viewdir.z);
        glUniformMatrix4fv(this->textureShader.ParameterLocation("mvp"), 1, GL_FALSE, glm::value_ptr(ortho));
        glUniform1i(this->textureShader.ParameterLocation("tex"), 0);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        this->textureShader.Disable();
        glBindVertexArray(0);
        glDisable(GL_TEXTURE_2D);

#if 1
        if (!this->nameSlot.Param<param::StringParam>()->Value().empty()) {
            std::string id = this->nameSlot.Param<param::StringParam>()->Value();
            if (id.size() > 4) {
                id = id.substr(0, 4);
            }
            auto fontSize = 50.0f;
            auto stringToDraw = id.c_str();

            auto lineHeight = theFont.LineHeight(fontSize);
            auto lineWidth = theFont.LineWidth(fontSize, stringToDraw);
            std::array<float, 4> color = {0.0f, 0.0f, 0.0f, 1.0f};

            this->theFont.DrawString(mvp, color.data(), 0.0f, 0.0, -1.0f, fontSize, false, stringToDraw);
        }
#endif


        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
    }

    return true;
}
