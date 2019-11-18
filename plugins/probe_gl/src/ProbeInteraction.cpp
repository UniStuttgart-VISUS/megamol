#include "ProbeInteraction.h"

megamol::probe_gl::ProbeInteraction::ProbeInteraction()
    : Renderer3DModule_2(), m_probe_fbo_slot("", ""), m_hull_fbo_slot("", ""), m_interaction_collection_slot("","") {

}

megamol::probe_gl::ProbeInteraction::~ProbeInteraction() {}

bool megamol::probe_gl::ProbeInteraction::OnMouseButton(
    core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {
    return false;
}

bool megamol::probe_gl::ProbeInteraction::OnMouseMove(double x, double y) { return false; }

bool megamol::probe_gl::ProbeInteraction::create() { return true; }

void megamol::probe_gl::ProbeInteraction::release() {}

bool megamol::probe_gl::ProbeInteraction::GetExtents(core::view::CallRender3D_2& call) { return true; }

bool megamol::probe_gl::ProbeInteraction::Render(core::view::CallRender3D_2& call) { return true; }
