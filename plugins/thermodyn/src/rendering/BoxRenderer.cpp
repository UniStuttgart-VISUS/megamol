#include "stdafx.h"
#include "BoxRenderer.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/utility/ShaderSourceFactory.h"

#include "thermodyn/BoxDataCall.h"
#include "vislib/math/ShallowMatrix.h"
#include "vislib/math/Matrix.h"


megamol::thermodyn::rendering::BoxRenderer::BoxRenderer() : dataInSlot_("dataIn", "Input of boxes to render") {
    dataInSlot_.SetCompatibleCall<BoxDataCallDescription>();
    dataInSlot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&dataInSlot_);
}


megamol::thermodyn::rendering::BoxRenderer::~BoxRenderer() { this->Release(); }


bool megamol::thermodyn::rendering::BoxRenderer::create() {
    core::utility::ShaderSourceFactory& factory = this->GetCoreInstance()->ShaderSourceFactory();

    try {
        vislib::graphics::gl::ShaderSource vert, frag;

        factory.MakeShaderSource("therm_box::vertex", vert);
        factory.MakeShaderSource("therm_box::fragment", frag);

        boxShader_.Compile(vert.Code(), vert.Count(), frag.Code(), frag.Count());

        boxShader_.Link();
    } catch (vislib::graphics::gl::GLSLShader::CompileException& ce) {
        vislib::sys::Log::DefaultLog.WriteError("BoxRenderer: Unable to compile therm_box shader: %s ... %s\n",
            vislib::graphics::gl::GLSLShader::CompileException::CompileActionName(ce.FailedAction()), ce.GetMsgA());
        return false;
    } catch (vislib::graphics::gl::OpenGLException& oe) {
        vislib::sys::Log::DefaultLog.WriteError("BoxRenderer: Failed to create therm_box shader: %s\n", oe.GetMsgA());

        return false;
    }

    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vvbo_);
    glGenBuffers(1, &cvbo_);

    return true;
}


void megamol::thermodyn::rendering::BoxRenderer::release() {
    boxShader_.Release();

    glDeleteBuffers(1, &vvbo_);
    glDeleteBuffers(1, &cvbo_);
    glDeleteVertexArrays(1, &vao_);
}


bool megamol::thermodyn::rendering::BoxRenderer::Render(megamol::core::view::CallRender3D& call) {
    std::vector<BoxDataCall::box_entry_t> boxes;

    BoxDataCall* inBoxCall = nullptr;
    core::moldyn::MultiParticleDataCall* inParCall = nullptr;
    if ((inBoxCall = dataInSlot_.CallAs<BoxDataCall>()) != nullptr) {
        if (!(*inBoxCall)(0)) return false;

        boxes = *inBoxCall->GetBoxes();

    } else if ((inParCall = dataInSlot_.CallAs<core::moldyn::MultiParticleDataCall>()) != nullptr) {
        if (!(*inParCall)(1)) return false;

        boxes.emplace_back(BoxDataCall::box_entry_t{
            inParCall->AccessBoundingBoxes().ObjectSpaceBBox(), "bbox", {1.0f, 1.0f, 1.0f, 0.5f}});
    } else {
        vislib::sys::Log::DefaultLog.WriteError("BoxRenderer: Could not establish call\n");
        return false;
    }

    auto data = prepareData(boxes);

    // upload data
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vvbo_);
    glBufferData(GL_ARRAY_BUFFER, data.first.size() * sizeof(float), data.first.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, false, 12, nullptr);
    glBindBuffer(GL_ARRAY_BUFFER, cvbo_);
    glBufferData(GL_ARRAY_BUFFER, data.second.size() * sizeof(float), data.second.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, false, 16, nullptr);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // render data
    GLfloat modelViewMatrix_column[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix_column);
    vislib::math::ShallowMatrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> MV(&modelViewMatrix_column[0]);

    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> scaleMat;
    scaleMat.SetAt(0, 0, scaling_);
    scaleMat.SetAt(1, 1, scaling_);
    scaleMat.SetAt(2, 2, scaling_);
    MV = MV * scaleMat;

    GLfloat projMatrix_column[16];
    glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
    vislib::math::ShallowMatrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> const PM(&projMatrix_column[0]);

    auto const MVP = PM * MV;

    //glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    boxShader_.Enable();

    glUniformMatrix4fv(boxShader_.ParameterLocation("mvp"), 1, false, MVP.PeekComponents());

    glDrawArrays(GL_QUADS, 0, data.first.size()/3);

    boxShader_.Disable();

    glDisable(GL_BLEND);
    //glDisable(GL_DEPTH_TEST);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glBindVertexArray(0);

    return true;
}


bool megamol::thermodyn::rendering::BoxRenderer::GetExtents(core::view::CallRender3D& call) {
    BoxDataCall* inBoxCall = nullptr;
    core::moldyn::MultiParticleDataCall* inParCall = nullptr;
    if ((inBoxCall = dataInSlot_.CallAs<BoxDataCall>()) != nullptr) {
        if (!(*inBoxCall)(0)) return false;

        auto const boxes = inBoxCall->GetBoxes();

        if (boxes->empty()) {
            call.AccessBoundingBoxes().SetObjectSpaceBBox({-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f});
            call.AccessBoundingBoxes().SetObjectSpaceClipBox({-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f});
        } else {

            vislib::math::Cuboid<float> box = (*boxes)[0].box_;
            for (size_t i = 1; i < boxes->size(); ++i) {
                box.Union((*boxes)[i].box_);
            }

            call.AccessBoundingBoxes().SetObjectSpaceBBox(box);
            call.AccessBoundingBoxes().SetObjectSpaceClipBox(box);
        }

    } else if ((inParCall = dataInSlot_.CallAs<core::moldyn::MultiParticleDataCall>()) != nullptr) {
        if (!(*inParCall)(1)) return false;

        call.AccessBoundingBoxes().SetObjectSpaceBBox(inParCall->AccessBoundingBoxes().ObjectSpaceBBox());
        call.AccessBoundingBoxes().SetObjectSpaceClipBox(inParCall->AccessBoundingBoxes().ObjectSpaceClipBox());
    } else {
        vislib::sys::Log::DefaultLog.WriteError("BoxRenderer: Could not establish call\n");
        return false;
    }


    scaling_ = call.AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    if (scaling_ > 0.0000001) {
        scaling_ = 10.0f / scaling_;
    } else {
        scaling_ = 1.0f;
    }
    call.AccessBoundingBoxes().MakeScaledWorld(scaling_);

    return true;
}
