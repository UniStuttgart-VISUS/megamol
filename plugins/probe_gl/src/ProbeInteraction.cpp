#include "ProbeInteraction.h"

#include "compositing/CompositingCalls.h"
#include "ProbeGlCalls.h"

megamol::probe_gl::ProbeInteraction::ProbeInteraction()
    : Renderer3DModule_2()
    , m_probe_fbo_slot("getProbeFBO", "")
    , m_hull_fbo_slot("getHullFBO", "")
    , m_interaction_collection_slot("deployInteractions","")
{
    this->m_probe_fbo_slot.SetCompatibleCall<compositing::CallFramebufferGLDescription>();
    this->MakeSlotAvailable(&this->m_probe_fbo_slot);

    this->m_hull_fbo_slot.SetCompatibleCall<compositing::CallFramebufferGLDescription>();
    this->MakeSlotAvailable(&this->m_hull_fbo_slot);


}

megamol::probe_gl::ProbeInteraction::~ProbeInteraction() { this->Release(); }

bool megamol::probe_gl::ProbeInteraction::OnMouseButton(
    core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {
    return false;
}

bool megamol::probe_gl::ProbeInteraction::OnMouseMove(double x, double y) {

    double dx = x - this->m_cursor_x;
    double dy = y - this->m_cursor_y;

    this->m_cursor_x = x;
    this->m_cursor_y = y;

    return false; 
}

bool megamol::probe_gl::ProbeInteraction::create() { return true; }

void megamol::probe_gl::ProbeInteraction::release() { }

bool megamol::probe_gl::ProbeInteraction::GetExtents(core::view::CallRender3D_2& call) { return true; }

bool megamol::probe_gl::ProbeInteraction::Render(core::view::CallRender3D_2& call) {

    
    core::view::CallRender3D_2* cr = dynamic_cast<core::view::CallRender3D_2*>(&call);
    if (cr == NULL) return false;

    // obtain camera information
    core::view::Camera_2 cam(cr->GetCamera());
    cam_type::snapshot_type snapshot;
    cam_type::matrix_type view_tmp, proj_tmp;
    cam.calc_matrices(snapshot, view_tmp, proj_tmp, core::thecam::snapshot_content::all);
    m_view_mx_cpy = view_tmp;
    m_proj_mx_cpy = proj_tmp;

    auto call_probe_fbo = this->m_probe_fbo_slot.CallAs<compositing::CallFramebufferGL>();
    auto call_hull_fbo = this->m_hull_fbo_slot.CallAs<compositing::CallFramebufferGL>();

    if (call_probe_fbo == NULL) return false;
    if (call_hull_fbo == NULL) return false;

    auto probe_fbo = call_probe_fbo->getData();
    auto hull_fbo = call_hull_fbo->getData();


    //TODO read obj ids from FBOs...

    return true;
}
