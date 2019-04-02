/*
 * BrickStatsRenderer.cpp
 *
 * Copyright (C) 2016 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "BrickStatsRenderer.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/BoolParam.h"

using namespace megamol::core;
using namespace megamol::stdplugin::moldyn::rendering;
#define MAP_BUFFER_LOCALLY
#define DEBUG_BLAHBLAH

const GLuint SSBObindingPoint = 2;
//#define NGS_THE_INSTANCE "gl_InstanceID"
#define NGS_THE_INSTANCE "gl_VertexID"

//typedef void (APIENTRY *GLDEBUGPROC)(GLenum source,GLenum type,GLuint id,GLenum severity,GLsizei length,const GLchar *message,const void *userParam);
extern void APIENTRY MyFunkyDebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length,
    const GLchar* message, const GLvoid* userParam) {
    const char *sourceText, *typeText, *severityText;
    switch (source) {
    case GL_DEBUG_SOURCE_API:
        sourceText = "API";
        break;
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
        sourceText = "Window System";
        break;
    case GL_DEBUG_SOURCE_SHADER_COMPILER:
        sourceText = "Shader Compiler";
        break;
    case GL_DEBUG_SOURCE_THIRD_PARTY:
        sourceText = "Third Party";
        break;
    case GL_DEBUG_SOURCE_APPLICATION:
        sourceText = "Application";
        break;
    case GL_DEBUG_SOURCE_OTHER:
        sourceText = "Other";
        break;
    default:
        sourceText = "Unknown";
        break;
    }
    switch (type) {
    case GL_DEBUG_TYPE_ERROR:
        typeText = "Error";
        break;
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
        typeText = "Deprecated Behavior";
        break;
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
        typeText = "Undefined Behavior";
        break;
    case GL_DEBUG_TYPE_PORTABILITY:
        typeText = "Portability";
        break;
    case GL_DEBUG_TYPE_PERFORMANCE:
        typeText = "Performance";
        break;
    case GL_DEBUG_TYPE_OTHER:
        typeText = "Other";
        break;
    case GL_DEBUG_TYPE_MARKER:
        typeText = "Marker";
        break;
    default:
        typeText = "Unknown";
        break;
    }
    switch (severity) {
    case GL_DEBUG_SEVERITY_HIGH:
        severityText = "High";
        break;
    case GL_DEBUG_SEVERITY_MEDIUM:
        severityText = "Medium";
        break;
    case GL_DEBUG_SEVERITY_LOW:
        severityText = "Low";
        break;
    case GL_DEBUG_SEVERITY_NOTIFICATION:
        severityText = "Notification";
        break;
    default:
        severityText = "Unknown";
        break;
    }
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "[%s %s] (%s %u) %s\n", sourceText, severityText, typeText, id, message);
}


/*
 * moldyn::NGSphereRenderer::NGSphereRenderer
 */
BrickStatsRenderer::BrickStatsRenderer(void) : Renderer3DModule(),
    statSlot("statData", "slot where the brick stats come in"),
    numBricksSlot("numBricks", "how many bricks there are in the data"),
    showBrickSlot("showBrick", "which brick to show, if set"),
    showStatistics("showStats", "indicators for mean, stddev"),
    showBoxes("showBoxes", "shows the bounding boxes for each brick"),
    statsBuffer(0), statsShader(), boxesShader()
    {

        this->statSlot.SetCompatibleCall<BrickStatsCallDescription>();
        this->MakeSlotAvailable(&this->statSlot);

        this->numBricksSlot << new megamol::core::param::IntParam(0);
        this->MakeSlotAvailable(&this->numBricksSlot);

        this->showBrickSlot << new megamol::core::param::StringParam("");
        this->MakeSlotAvailable(&this->showBrickSlot);

        this->showStatistics << new param::BoolParam(false);
        this->MakeSlotAvailable(&this->showStatistics);

        this->showBoxes << new param::BoolParam(true);
        this->MakeSlotAvailable(&this->showBoxes);

}


/*
 * moldyn::NGSphereRenderer::~NGSphereRenderer
 */
BrickStatsRenderer::~BrickStatsRenderer(void) {
    this->Release();
}


bool BrickStatsRenderer::makeProgram(std::string name, vislib::graphics::gl::GLSLShader& program,
    vislib::graphics::gl::ShaderSource& vert, vislib::graphics::gl::ShaderSource& frag) {
    try {
        if (!program.Compile(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to compile %s: Unknown error\n", name.c_str());
            return false;
        }
        if (!program.Link()) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to link %s: Unknown error\n", name.c_str());
            return false;
        }

    } catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile %s (@%s): %s\n", name.c_str(),
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
            ce.FailedAction()), ce.GetMsgA());
        return false;
    } catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile %s: %s\n", name.c_str(), e.GetMsgA());
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile %s: Unknown exception\n", name.c_str());
        return false;
    }
	return false;
}

/*
 * moldyn::SimpleSphereRenderer::create
 */
bool BrickStatsRenderer::create(void) {
#ifdef DEBUG_BLAHBLAH
    glDebugMessageCallback(MyFunkyDebugCallback, NULL);

    // 1|[API Notification] (Other 131185) Buffer detailed info: Buffer object 1 (bound to GL_SHADER_STORAGE_BUFFER, and GL_SHADER_STORAGE_BUFFER (0), usage hint is GL_STATIC_DRAW) will use VIDEO memory as the source for buffer object operation
    std::vector<GLuint> ids = { 131185 };
    glDebugMessageControl(GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_OTHER, GL_DONT_CARE, static_cast<GLsizei>(ids.size()), ids.data(), GL_FALSE);
#endif
    glGenBuffers(1, &this->statsBuffer);

    vislib::graphics::gl::ShaderSource vert, frag;
    if (!this->instance()->ShaderSourceFactory().MakeShaderSource("BrickStatsRenderer::stats::vertex", vert)) {
        return false;
    }
    if (!this->instance()->ShaderSourceFactory().MakeShaderSource("BrickStatsRenderer::stats::fragment", frag)) {
        return false;
    }
    if (!makeProgram("stats shader", statsShader, vert, frag)) {
        return false;
    }

    if (!this->instance()->ShaderSourceFactory().MakeShaderSource("BrickStatsRenderer::boxes::vertex", vert)) {
        return false;
    }
    if (!this->instance()->ShaderSourceFactory().MakeShaderSource("BrickStatsRenderer::boxes::fragment", frag)) {
        return false;
    }
    if (!makeProgram("stats shader", boxesShader, vert, frag)) {
        return false;
    }


    //glGenVertexArrays(1, &this->vertArray);
    //glBindVertexArray(this->vertArray);
    //glGenBuffers(1, &this->theSingleBuffer);
    //glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->theSingleBuffer);
    //glBufferStorage(GL_SHADER_STORAGE_BUFFER, this->bufSize * this->numBuffers, nullptr, singleBufferCreationBits);
    //this->theSingleMappedMem = glMapNamedBufferRangeEXT(this->theSingleBuffer, 0, this->bufSize * this->numBuffers, singleBufferMappingBits);
    //glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    //glBindVertexArray(0);

    return true;
}


/*
* moldyn::AbstractSimpleSphereRenderer::GetExtents
*/
bool BrickStatsRenderer::GetExtents(Call& call) {
    this->assertData(call);

    return true;
}

/*
 * moldyn::SimpleSphereRenderer::release
 */
void BrickStatsRenderer::release(void) {
    glDeleteBuffers(1, &this->statsBuffer);
}


bool BrickStatsRenderer::assertData(Call& call) {
    this->numBricks = 0;
    this->numBricksSlot.Param<param::IntParam>()->SetValue(0);
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    BrickStatsCall *cb = this->statSlot.CallAs<BrickStatsCall>();
    if (cb == NULL) return false;

    cb->SetFrameID(static_cast<unsigned int>(cr->Time()), false);
    if ((*cb)(1)) {
        cr->SetTimeFramesCount(cb->FrameCount());
        cr->AccessBoundingBoxes() = cb->AccessBoundingBoxes();

        scaling = cr->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        if (scaling > 0.0000001) {
            scaling = 10.0f / scaling;
        }
        else {
            scaling = 1.0f;
        }
        cr->AccessBoundingBoxes().MakeScaledWorld(scaling);
    } else {
        cr->SetTimeFramesCount(1);
        cr->AccessBoundingBoxes().Clear();
    }

    if ((*cb)(0)) {
        auto bricks = cb->GetBricks();
        this->numBricks = bricks->Count();
        this->numBricksSlot.Param<param::IntParam>()->SetValue(static_cast<int>(this->numBricks));
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->statsBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, bricks->Count() * sizeof(BrickStatsCall::BrickInfo), bricks->PeekElements(), GL_STATIC_DRAW);
        return true;
    } else {
        return false;
    }

}


/*
 * moldyn::SimpleSphereRenderer::Render
 */
bool BrickStatsRenderer::Render(Call& call) {
#ifdef DEBUG_BLAHBLAH
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
#endif
    this->assertData(call);
   
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->statsBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, this->statsBuffer);

    // this is the apex of suck and must die
    GLfloat modelViewMatrix_column[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix_column);
    GLfloat projMatrix_column[16];
    glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
    // end suck

    glLineWidth(1.0);
    glDisable(GL_LINE_SMOOTH);

    if (this->showStatistics.Param<param::BoolParam>()->Value()) {
        this->statsShader.Enable();
        glUniformMatrix4fv(this->statsShader.ParameterLocation("modelView"), 1, GL_FALSE, modelViewMatrix_column);
        glUniformMatrix4fv(this->statsShader.ParameterLocation("projection"), 1, GL_FALSE, projMatrix_column);
        glUniform1f(this->statsShader.ParameterLocation("scaling"), this->scaling);

        auto show = this->showBrickSlot.Param<param::StringParam>()->Value();
        if (show.IsEmpty()) {
            glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(this->numBricks * 6));
        } else {
            auto num = vislib::TCharTraits::ParseUInt64(show);
            glDrawArrays(GL_LINES, static_cast<GLint>(num * 6), 6);
        }
        this->statsShader.Disable();
    }

    if (this->showBoxes.Param<param::BoolParam>()->Value()) {
        this->boxesShader.Enable();
        glUniformMatrix4fv(this->boxesShader.ParameterLocation("modelView"), 1, GL_FALSE, modelViewMatrix_column);
        glUniformMatrix4fv(this->boxesShader.ParameterLocation("projection"), 1, GL_FALSE, projMatrix_column);
        glUniform1f(this->boxesShader.ParameterLocation("scaling"), this->scaling);

        auto show = this->showBrickSlot.Param<param::StringParam>()->Value();
        if (show.IsEmpty()) {
            glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(this->numBricks * 24));
        } else {
            auto num = vislib::TCharTraits::ParseUInt64(show);
            glDrawArrays(GL_LINES, static_cast<GLint>(num * 24), 24);
        }
        this->boxesShader.Disable();
    }


#ifdef DEBUG_BLAHBLAH
    glDisable(GL_DEBUG_OUTPUT);
    glDisable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
#endif

    return true;
}
