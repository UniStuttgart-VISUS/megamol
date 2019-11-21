#include "stdafx.h"
#include "HistogramRenderer2D.h"

using namespace megamol;
using namespace megamol::infovis;

using vislib::sys::Log;

HistogramRenderer2D::HistogramRenderer2D() : Renderer2D() {
}

HistogramRenderer2D::~HistogramRenderer2D() {
    this->Release();
}

bool HistogramRenderer2D::create() {
    return true;
}

void HistogramRenderer2D::release() {
}

bool HistogramRenderer2D::GetExtents(core::view::CallRender2D &call) {
    call.SetBoundingBox(0.0f, 0.0f, 4.0f, 4.0f);
    return true;
}

bool HistogramRenderer2D::Render(core::view::CallRender2D &call) {
    ::glColor3ub(255, 0, 0);
    ::glBegin(GL_LINE_LOOP);
    ::glVertex2f(call.GetBoundingBox().Left(), call.GetBoundingBox().Bottom());
    ::glVertex2f(call.GetBoundingBox().Right(), call.GetBoundingBox().Bottom());
    ::glVertex2f(call.GetBoundingBox().Right(), call.GetBoundingBox().Top());
    ::glVertex2f(call.GetBoundingBox().Left(), call.GetBoundingBox().Top());
    ::glEnd();
    const float border = 0.1f;
    ::glColor3ub(255, 255, 0);
    ::glBegin(GL_LINE_LOOP);
    ::glVertex2f(call.GetBoundingBox().Left() + border, call.GetBoundingBox().Bottom() + border);
    ::glVertex2f(call.GetBoundingBox().Right() - border, call.GetBoundingBox().Bottom() + border);
    ::glVertex2f(call.GetBoundingBox().Right() - border, call.GetBoundingBox().Top() - border);
    ::glVertex2f(call.GetBoundingBox().Left() + border, call.GetBoundingBox().Top() - border);
    ::glEnd();

    return true;
}

bool HistogramRenderer2D::OnMouseButton(core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {
    return false;
}

bool HistogramRenderer2D::OnMouseMove(double x, double y) {
    return false;
}
