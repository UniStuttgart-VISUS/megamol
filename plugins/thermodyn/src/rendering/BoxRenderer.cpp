#include "stdafx.h"
#include "BoxRenderer.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/utility/ShaderSourceFactory.h"

#include "thermodyn/BoxDataCall.h"
#include "vislib/math/Matrix.h"
#include "vislib/math/ShallowMatrix.h"

#include "glm/gtc/type_ptr.hpp"
#include "glm/gtx/transform.hpp"


megamol::thermodyn::rendering::BoxRenderer::BoxRenderer()
    : dataInSlot_("dataIn", "Input of boxes to render")
    , calculateGlobalBoundingBoxParam("calcBoundingBoxEachFrame",
          "Recalculate the global bounding box each frame. This is resource instensive and "
          "might lead to bad frame rates") {
    dataInSlot_.SetCompatibleCall<BoxDataCallDescription>();
    dataInSlot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&dataInSlot_);

    calculateGlobalBoundingBoxParam.SetParameter(new core::param::BoolParam(false));
    this->MakeSlotAvailable(&this->calculateGlobalBoundingBoxParam);
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
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "BoxRenderer: Unable to compile therm_box shader: %s ... %s\n",
            vislib::graphics::gl::GLSLShader::CompileException::CompileActionName(ce.FailedAction()), ce.GetMsgA());
        return false;
    } catch (vislib::graphics::gl::OpenGLException& oe) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "BoxRenderer: Failed to create therm_box shader: %s\n", oe.GetMsgA());

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


bool megamol::thermodyn::rendering::BoxRenderer::Render(megamol::core::view::CallRender3DGL& call) {
    std::vector<BoxDataCall::box_entry_t> boxes;

    BoxDataCall* inBoxCall = nullptr;
    core::moldyn::MultiParticleDataCall* inParCall = nullptr;
    bool dirty = false;
    if ((inBoxCall = dataInSlot_.CallAs<BoxDataCall>()) != nullptr) {
        if (!(*inBoxCall)(0)) return false;

        boxes = *inBoxCall->GetBoxes();
        if (this->inDataHash_ != inBoxCall->DataHash()) {
            dirty = true;
            this->inDataHash_ = inBoxCall->DataHash();
        }
    } else if ((inParCall = dataInSlot_.CallAs<core::moldyn::MultiParticleDataCall>()) != nullptr) {
        if (!(*inParCall)(1)) return false;

        boxes.emplace_back(BoxDataCall::box_entry_t{
            inParCall->AccessBoundingBoxes().ObjectSpaceBBox(), "bbox", {1.0f, 1.0f, 1.0f, 0.5f}});
        if (this->inDataHash_ != inParCall->DataHash()) {
            dirty = true;
            this->inDataHash_ = inParCall->DataHash();
        }
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError("BoxRenderer: Could not establish call\n");
        return false;
    }

    core::view::Camera_2 cam;
    call.GetCamera(cam);
    core::view::Camera_2::snapshot_type cam_snap;
    core::view::Camera_2::matrix_type view, proj;
    cam.calc_matrices(cam_snap, view, proj, core::thecam::snapshot_content::all);

    // upload data
    glBindVertexArray(vao_);
    if (dirty) prepareData(boxes, this->drawData);
    glBindBuffer(GL_ARRAY_BUFFER, vvbo_);
    if (dirty) {
        glBufferData(
            GL_ARRAY_BUFFER, this->drawData.first.size() * sizeof(float), this->drawData.first.data(), GL_STATIC_DRAW);
    }
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, false, 12, nullptr);
    glBindBuffer(GL_ARRAY_BUFFER, cvbo_);
    if (dirty) {
        glBufferData(GL_ARRAY_BUFFER, this->drawData.second.size() * sizeof(float), this->drawData.second.data(),
            GL_STATIC_DRAW);
    }
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, false, 16, nullptr);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // render data
    glm::mat4 MVP = proj * view;

    // glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    boxShader_.Enable();

    glUniformMatrix4fv(boxShader_.ParameterLocation("mvp"), 1, false, glm::value_ptr(MVP));

    glDrawArrays(GL_QUADS, 0, this->drawData.first.size() / 3);

    boxShader_.Disable();

    glDisable(GL_BLEND);
    // glDisable(GL_DEPTH_TEST);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glBindVertexArray(0);

    return true;
}


bool megamol::thermodyn::rendering::BoxRenderer::GetExtents(core::view::CallRender3DGL& call) {
    BoxDataCall* inBoxCall = nullptr;
    core::moldyn::MultiParticleDataCall* inParCall = nullptr;
    if ((inBoxCall = dataInSlot_.CallAs<BoxDataCall>()) != nullptr) {
        if (!(*inBoxCall)(1)) return false;
        call.AccessBoundingBoxes().SetBoundingBox(inBoxCall->AccessBoundingBoxes().ObjectSpaceBBox());
        call.AccessBoundingBoxes().SetClipBox(inBoxCall->AccessBoundingBoxes().ObjectSpaceClipBox());

        if (this->calculateGlobalBoundingBoxParam.Param<core::param::BoolParam>()->Value()) {
            if (!(*inBoxCall)(0)) return false;
            auto const boxes = inBoxCall->GetBoxes();

            if (boxes == nullptr || boxes->empty()) {
                call.AccessBoundingBoxes().SetBoundingBox({-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f});
                call.AccessBoundingBoxes().SetClipBox({-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f});
            } else {

                vislib::math::Cuboid<float> box = (*boxes)[0].box_;
                for (size_t i = 1; i < boxes->size(); ++i) {
                    box.Union((*boxes)[i].box_);
                }

                call.AccessBoundingBoxes().SetBoundingBox(box);
                call.AccessBoundingBoxes().SetClipBox(box);
            }
        }
    } else if ((inParCall = dataInSlot_.CallAs<core::moldyn::MultiParticleDataCall>()) != nullptr) {
        if (!(*inParCall)(1)) return false;

        call.AccessBoundingBoxes().SetBoundingBox(inParCall->AccessBoundingBoxes().ObjectSpaceBBox());
        call.AccessBoundingBoxes().SetClipBox(inParCall->AccessBoundingBoxes().ObjectSpaceClipBox());
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError("BoxRenderer: Could not establish call\n");
        return false;
    }


    scaling_ = call.AccessBoundingBoxes().BoundingBox().LongestEdge();
    if (scaling_ > 0.0000001) {
        scaling_ = 10.0f / scaling_;
    } else {
        scaling_ = 1.0f;
    }
    // call.AccessBoundingBoxes().MakeScaledWorld(scaling_);

    return true;
}
