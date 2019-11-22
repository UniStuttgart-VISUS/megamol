#include "stdafx.h"
#include "HistogramRenderer2D.h"

using namespace megamol;
using namespace megamol::infovis;
using namespace megamol::stdplugin::datatools;

using vislib::sys::Log;

HistogramRenderer2D::HistogramRenderer2D()
    : Renderer2D()
    , tableDataCallerSlot("getData", "Float table input")
    , transferFunctionCallerSlot("getTransferFunction", "Transfer function input")
    , flagStorageCallerSlot("getFlagStorage", "Flag storage input")
    , font("Evolventa-SansSerif", core::utility::SDFFont::RenderType::RENDERTYPE_FILL)
{
    this->tableDataCallerSlot.SetCompatibleCall<table::TableDataCallDescription>();
    this->MakeSlotAvailable(&this->tableDataCallerSlot);

    this->transferFunctionCallerSlot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->transferFunctionCallerSlot);

    this->flagStorageCallerSlot.SetCompatibleCall<core::FlagCallDescription>();
    this->MakeSlotAvailable(&this->flagStorageCallerSlot);
}

HistogramRenderer2D::~HistogramRenderer2D() {
    this->Release();
}

bool HistogramRenderer2D::create() {
    if (!font.Initialise(this->GetCoreInstance())) return false;

    if (!makeProgram("::histo::draw", this->histogramProgram)) return false;

    static const GLfloat quadVertices[] = {
            0.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 0.0f,
    };

    static const GLuint quadIndices[] = {
            0, 1, 2,
            1, 2, 3,
    };

    glGenVertexArrays(1, &quadVertexArray);
    glGenBuffers(1, &quadVertexBuffer);
    glGenBuffers(1, &quadIndexBuffer);

    glBindVertexArray(quadVertexArray);
    glBindBuffer(GL_ARRAY_BUFFER, quadVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 3 * 4, quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quadIndexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 2 * 3 * sizeof(GLuint), quadIndices, GL_STATIC_DRAW);
    glBindVertexArray(0);

    return true;
}

void HistogramRenderer2D::release() {
}

bool HistogramRenderer2D::GetExtents(core::view::CallRender2D &call) {
    call.SetBoundingBox(0.0f, 0.0f, 10.0f, 10.0f);
    return true;
}

bool HistogramRenderer2D::Render(core::view::CallRender2D &call) {
    // this is the apex of suck and must die
    glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix_column);
    glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
    // end suck

    glUseProgram(histogramProgram);
    glUniformMatrix4fv(histogramProgram.ParameterLocation("modelView"), 1, GL_FALSE, modelViewMatrix_column);
    glUniformMatrix4fv(histogramProgram.ParameterLocation("projection"), 1, GL_FALSE, projMatrix_column);

    glBindVertexArray(quadVertexArray);
    glDrawElements(GL_TRIANGLES, 2 * 3, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
    glUseProgram(0);

    float red[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    this->font.DrawString(red, 5.0, 5.0, 0.5f, false, "hello", core::utility::AbstractFont::ALIGN_CENTER_MIDDLE);

    return true;
}

bool HistogramRenderer2D::OnMouseButton(core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {
    return false;
}

bool HistogramRenderer2D::OnMouseMove(double x, double y) {
    return false;
}
