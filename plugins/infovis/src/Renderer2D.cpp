#include "stdafx.h"
#include "Renderer2D.h"

#include "mmcore/CoreInstance.h"

#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/sys/Log.h"

using namespace megamol;
using namespace megamol::infovis;


void Renderer2D::computeDispatchSizes(
    uint64_t numItems, GLint const localSizes[3], GLint const maxCounts[3], GLuint dispatchCounts[3]) const {
    const auto localSize = localSizes[0] * localSizes[1] * localSizes[2];
    const uint64_t needed_groups = (numItems + localSize - 1) / localSize; // round up int div
    dispatchCounts[0] = std::clamp<GLint>(needed_groups, 1, maxCounts[0]);
    dispatchCounts[1] = std::clamp<GLint>((needed_groups + dispatchCounts[0] - 1) / dispatchCounts[0], 1, maxCounts[1]);
    const auto tmp = dispatchCounts[0] * dispatchCounts[1];
    dispatchCounts[2] = std::clamp<GLint>((needed_groups + tmp - 1) / tmp, 1, maxCounts[2]);
    const uint64_t totalCounts = dispatchCounts[0] * dispatchCounts[1] * dispatchCounts[2];
    ASSERT(totalCounts * localSize >= numItems);
    ASSERT(totalCounts * localSize - numItems < localSize);
}


bool Renderer2D::makeProgram(std::string prefix, vislib::graphics::gl::GLSLShader& program) const {
    vislib::graphics::gl::ShaderSource vert, frag;

    vislib::StringA vertname((prefix + "::vert").c_str());
    vislib::StringA fragname((prefix + "::frag").c_str());
    vislib::StringA pref(prefix.c_str());

    if (!this->instance()->ShaderSourceFactory().MakeShaderSource(vertname, vert)) return false;
    if (!this->instance()->ShaderSourceFactory().MakeShaderSource(fragname, frag)) return false;

    try {
        if (!program.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_ERROR, "Unable to compile %s: Unknown error\n", pref.PeekBuffer());
            return false;
        }
        makeDebugLabel(GL_PROGRAM, program.ProgramHandle(), prefix.c_str());
    } catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Unable to compile %s (@%s):\n%s\n",
            pref.PeekBuffer(),
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA());
        return false;
    } catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to compile %s:\n%s\n", pref.PeekBuffer(), e.GetMsgA());
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to compile %s: Unknown exception\n", pref.PeekBuffer());
        return false;
    }

    return true;
}

bool Renderer2D::makeProgram(std::string prefix, vislib::graphics::gl::GLSLGeometryShader& program) const {
    vislib::graphics::gl::ShaderSource vert, geom, frag;

    vislib::StringA vertname((prefix + "::vert").c_str());
    vislib::StringA geomname((prefix + "::geom").c_str());
    vislib::StringA fragname((prefix + "::frag").c_str());
    vislib::StringA pref(prefix.c_str());

    if (!this->instance()->ShaderSourceFactory().MakeShaderSource(vertname, vert)) return false;
    if (!this->instance()->ShaderSourceFactory().MakeShaderSource(geomname, geom)) return false;
    if (!this->instance()->ShaderSourceFactory().MakeShaderSource(fragname, frag)) return false;

    try {
        if (!program.Compile(vert.Code(), vert.Count(), geom.Code(), geom.Count(), frag.Code(), frag.Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_ERROR, "Unable to compile %s: Unknown error\n", pref.PeekBuffer());
            return false;
        }
        if (!program.Link()) {
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_ERROR, "Unable to link %s: Unknown error\n", pref.PeekBuffer());
            return false;
        }
        makeDebugLabel(GL_PROGRAM, program.ProgramHandle(), prefix.c_str());
    } catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Unable to compile %s (@%s):\n%s\n",
            pref.PeekBuffer(),
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA());
        return false;
    } catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to compile %s:\n%s\n", pref.PeekBuffer(), e.GetMsgA());
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to compile %s: Unknown exception\n", pref.PeekBuffer());
        return false;
    }

    return true;
}

bool Renderer2D::makeProgram(std::string prefix, vislib::graphics::gl::GLSLTesselationShader& program) const {
    vislib::graphics::gl::ShaderSource vert, frag, control, eval, geom;

    vislib::StringA vertname((prefix + "::vert").c_str());
    vislib::StringA fragname((prefix + "::frag").c_str());
    vislib::StringA controlname((prefix + "::control").c_str());
    vislib::StringA evalname((prefix + "::eval").c_str());
    vislib::StringA geomname((prefix + "::geom").c_str());
    vislib::StringA pref(prefix.c_str());

    if (!this->instance()->ShaderSourceFactory().MakeShaderSource(vertname, vert)) return false;
    if (!this->instance()->ShaderSourceFactory().MakeShaderSource(fragname, frag)) return false;
    // no complete tess?
    auto r1 = this->instance()->ShaderSourceFactory().MakeShaderSource(controlname, control);
    auto r2 = this->instance()->ShaderSourceFactory().MakeShaderSource(evalname, eval);
    if (r1 != r2) return false;
    bool haveTess = r1;
    bool haveGeom = this->instance()->ShaderSourceFactory().MakeShaderSource(geomname, geom);

    try {
        if (!program.Compile(vert.Code(), vert.Count(), haveTess ? control.Code() : nullptr,
                haveTess ? control.Count() : 0, haveTess ? eval.Code() : nullptr, haveTess ? eval.Count() : 0,
                haveGeom ? geom.Code() : nullptr, haveGeom ? geom.Count() : 0, frag.Code(), frag.Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_ERROR, "Unable to compile %s: Unknown error\n", pref.PeekBuffer());
            return false;
        }
        if (!program.Link()) {
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_ERROR, "Unable to link %s: Unknown error\n", pref.PeekBuffer());
            return false;
        }
        makeDebugLabel(GL_PROGRAM, program.ProgramHandle(), prefix.c_str());
    } catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Unable to compile %s (@%s):\n%s\n",
            pref.PeekBuffer(),
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA());
        return false;
    } catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to compile %s: %s\n", pref.PeekBuffer(), e.GetMsgA());
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to compile %s: Unknown exception\n", pref.PeekBuffer());
        return false;
    }
    return true;
}

bool Renderer2D::makeProgram(std::string prefix, vislib::graphics::gl::GLSLComputeShader& program) const {
    vislib::graphics::gl::ShaderSource comp;

    vislib::StringA compname((prefix + "::comp").c_str());
    vislib::StringA pref(prefix.c_str());

    if (!this->instance()->ShaderSourceFactory().MakeShaderSource(compname, comp)) return false;

    try {
        if (!program.Compile(comp.Code(), comp.Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_ERROR, "Unable to compile %s: Unknown error\n", pref.PeekBuffer());
            return false;
        }
        if (!program.Link()) {
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_ERROR, "Unable to link %s: Unknown error\n", pref.PeekBuffer());
            return false;
        }
        makeDebugLabel(GL_PROGRAM, program.ProgramHandle(), prefix.c_str());
    } catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Unable to compile %s (@%s):\n%s\n",
            pref.PeekBuffer(),
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA());
        return false;
    } catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to compile %s: %s\n", pref.PeekBuffer(), e.GetMsgA());
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to compile %s: Unknown exception\n", pref.PeekBuffer());
        return false;
    }
    return true;
}

void Renderer2D::makeDebugLabel(GLenum identifier, GLuint name, const char* label) const {
#ifdef _DEBUG
    glObjectLabel(identifier, name, -1, label);
#endif
}
void Renderer2D::debugNotify(GLuint name, const char* message) const {
#ifdef _DEBUG
    glDebugMessageInsert(
        GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_MARKER, name, GL_DEBUG_SEVERITY_NOTIFICATION, -1, message);
#endif
}
void Renderer2D::debugPush(GLuint name, const char* groupLabel) const {
#ifdef _DEBUG
    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, name, -1, groupLabel);
#endif
}
void Renderer2D::debugPop() const {
#ifdef _DEBUG
    glPopDebugGroup();
#endif
}
