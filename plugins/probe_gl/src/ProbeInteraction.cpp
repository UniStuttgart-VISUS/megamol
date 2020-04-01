#include "ProbeInteraction.h"

#include "compositing/CompositingCalls.h"
#include "ProbeGlCalls.h"

#include "mmcore/CoreInstance.h"

#include <imgui.h>
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_internal.h>
#include "imgui_impl_opengl3.h"
#include "imgui_stdlib.h"

megamol::probe_gl::ProbeInteraction::ProbeInteraction()
    : Renderer3DModule_2()
    , m_version(0)
    , m_cursor_x(0)
    , m_cursor_y(0)
    , m_cursor_x_lastRightClick(0)
    , m_cursor_y_lastRightClick(0)
    , m_open_context_menu(false)
    , m_interactions(new ProbeInteractionCollection())
    , last_active_probe_id(-1)
    , m_probe_fbo_slot("getProbeFBO", "")
    , m_hull_fbo_slot("getHullFBO", "")
    , m_glyph_fbo_slot("getGlyphFBO", "")
    , m_interaction_collection_slot("deployInteractions","")
{
    this->m_probe_fbo_slot.SetCompatibleCall<compositing::CallFramebufferGLDescription>();
    this->MakeSlotAvailable(&this->m_probe_fbo_slot);

    this->m_hull_fbo_slot.SetCompatibleCall<compositing::CallFramebufferGLDescription>();
    this->MakeSlotAvailable(&this->m_hull_fbo_slot);

    this->m_glyph_fbo_slot.SetCompatibleCall<compositing::CallFramebufferGLDescription>();
    this->MakeSlotAvailable(&this->m_glyph_fbo_slot);

    this->m_interaction_collection_slot.SetCallback(
        CallProbeInteraction::ClassName(), "GetData", &ProbeInteraction::getInteractionCollection);
    this->m_interaction_collection_slot.SetCallback(
        CallProbeInteraction::ClassName(), "GetMetaData", &ProbeInteraction::getInteractionMetaData);
    this->MakeSlotAvailable(&this->m_interaction_collection_slot);
}

megamol::probe_gl::ProbeInteraction::~ProbeInteraction() { this->Release(); }

bool megamol::probe_gl::ProbeInteraction::OnMouseButton(
    core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {

    if (button == core::view::MouseButton::BUTTON_LEFT && action == core::view::MouseButtonAction::PRESS) {
        
        if (last_active_probe_id > 0)
        {
            m_interactions->accessPendingManipulations().push_back(
                ProbeManipulation{InteractionType::SELECT, static_cast<uint32_t>(last_active_probe_id), 0, 0, 0});

            return true;
        }

        m_open_context_menu = false;
    } 
    else if (button == core::view::MouseButton::BUTTON_RIGHT)// && action == core::view::MouseButtonAction::PRESS)
    {
        m_mouse_button_states[button] =
            (action == core::view::MouseButtonAction::PRESS) ? true : 
                (action == core::view::MouseButtonAction::RELEASE) ? false : m_mouse_button_states[button];


        if (action == core::view::MouseButtonAction::PRESS) {

            m_open_context_menu = true;

            m_cursor_x_lastRightClick = m_cursor_x;
            m_cursor_y_lastRightClick = m_cursor_y;
        }
    }

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
    auto call_glyph_fbo = this->m_glyph_fbo_slot.CallAs<compositing::CallFramebufferGL>();

    if (call_probe_fbo == NULL) return false;
    if (call_hull_fbo == NULL) return false;
    if (call_glyph_fbo == NULL) return false;

    if ((!(*call_probe_fbo)(0))) return false;
    if ((!(*call_hull_fbo)(0))) return false;
    if ((!(*call_glyph_fbo)(0))) return false;

    auto probe_fbo = call_probe_fbo->getData();
    auto hull_fbo = call_hull_fbo->getData();
    auto glyph_fbo = call_glyph_fbo->getData();

    //TODO read obj ids from FBOs...

    // bind fbo to read buffer for retrieving pixel data and bliting to default framebuffer
    hull_fbo->bindToRead(2);
    {
        //auto err = glGetError();
        //std::cerr << err << std::endl;
    }
    // get object id at cursor location from framebuffer's second color attachment
    float hull_depth_pixel_data = 0.0;
    // TODO check if cursor position is within framebuffer pixel range?
    glReadPixels(static_cast<GLint>(this->m_cursor_x), probe_fbo->getHeight() - static_cast<GLint>(this->m_cursor_y), 1,
        1, GL_RED, GL_FLOAT, &hull_depth_pixel_data);
    {
        //auto err = glGetError();
        //std::cerr << err << std::endl;
    }

    // bind fbo to read buffer for retrieving pixel data and bliting to default framebuffer
    probe_fbo->bindToRead(2);
    {
        //auto err = glGetError();
        //std::cerr << err << std::endl;
    }
    // get object id at cursor location from framebuffer's second color attachment
    float probe_depth_pixel_data = 0.0;
    // TODO check if cursor position is within framebuffer pixel range?
    glReadPixels(
        static_cast<GLint>(this->m_cursor_x), 
        probe_fbo->getHeight() - static_cast<GLint>(this->m_cursor_y),
        1,
        1, GL_RED, GL_FLOAT, &probe_depth_pixel_data);
    {
        //auto err = glGetError();
        //std::cerr << err << std::endl;
    }

    // bind fbo to read buffer for retrieving pixel data and bliting to default framebuffer
    glyph_fbo->bindToRead(2);
    {
        // auto err = glGetError();
        // std::cerr << err << std::endl;
    }
    // get object id at cursor location from framebuffer's second color attachment
    float glyph_depth_pixel_data = 0.0;
    // TODO check if cursor position is within framebuffer pixel range?
    glReadPixels(static_cast<GLint>(this->m_cursor_x), glyph_fbo->getHeight() - static_cast<GLint>(this->m_cursor_y), 1,
        1, GL_RED, GL_FLOAT, &glyph_depth_pixel_data);
    {
        // auto err = glGetError();
        // std::cerr << err << std::endl;
    }

    // bind fbo to read buffer for retrieving pixel data
    glyph_fbo->bindToRead(3);
    {
        // auto err = glGetError();
        // std::cerr << err << std::endl;
    }
    // get object id at cursor location from framebuffer's second color attachment
    GLint glyph_objId_pixel_data = -1;
    // TODO check if cursor position is within framebuffer pixel range?
    glReadPixels(static_cast<GLint>(this->m_cursor_x), glyph_fbo->getHeight() - static_cast<GLint>(this->m_cursor_y), 1,
        1, GL_RED_INTEGER, GL_INT, &glyph_objId_pixel_data);
    {
        // auto err = glGetError();
        // std::cerr << err << std::endl;
    }

    // bind fbo to read buffer for retrieving pixel data and bliting to default framebuffer
    probe_fbo->bindToRead(3);
    {
        //auto err = glGetError();
        //std::cerr << err << std::endl;
    }
    // get object id at cursor location from framebuffer's second color attachment
    GLint probe_objId_pixel_data = -1;
    // TODO check if cursor position is within framebuffer pixel range?
    glReadPixels(static_cast<GLint>(this->m_cursor_x), probe_fbo->getHeight() - static_cast<GLint>(this->m_cursor_y), 1,
        1, GL_RED_INTEGER, GL_INT, &probe_objId_pixel_data);
    {
        //auto err = glGetError();
        //std::cerr << err << std::endl;
    }

    //std::cout << "Object ID at " << m_cursor_x << "," << m_cursor_y << " : " << glyph_objId_pixel_data << std::endl;

    GLint objId = -1;
    float depth = 0.0;

    if (probe_depth_pixel_data > 0.0 && glyph_depth_pixel_data > 0.0) {
        if (probe_depth_pixel_data < glyph_depth_pixel_data) {
            objId = probe_objId_pixel_data;
            depth = probe_depth_pixel_data;
        } else {
            objId = glyph_objId_pixel_data;
            depth = glyph_depth_pixel_data;
        }
    }
    else if (probe_depth_pixel_data > 0.0)
    {
        objId = probe_objId_pixel_data;
        depth = probe_depth_pixel_data;
    }
    else if (glyph_depth_pixel_data > 0.0) 
    {
        objId = glyph_objId_pixel_data;
        depth = glyph_depth_pixel_data;
    }


    if (depth > hull_depth_pixel_data)
    {
        objId = -1;
    }

    //std::cout << "Object ID at " << m_cursor_x << "," << m_cursor_y << " : " << objId << std::endl;

    // Clear interactions from last frame
    m_interactions->accessPendingManipulations().clear();

    if (objId > -1)
    {
        m_interactions->accessPendingManipulations().push_back(
            ProbeManipulation{InteractionType::HIGHLIGHT, static_cast<uint32_t>(objId), 0, 0, 0});
    }
    if (last_active_probe_id > 0 && last_active_probe_id != objId)
    {
        m_interactions->accessPendingManipulations().push_back(
            ProbeManipulation{InteractionType::DEHIGHLIGHT, static_cast<uint32_t>(last_active_probe_id), 0, 0, 0});
    }

    last_active_probe_id = objId;


    if (m_open_context_menu)
    {
        bool my_tool_active = true;
        float my_color[4] = {0.0, 0.0, 0.0, 0.0};

        auto ctx = reinterpret_cast<ImGuiContext*>(this->GetCoreInstance()->GetCurrentImGuiContext());
        if (ctx != nullptr) {
            ImGui::SetCurrentContext(ctx);

            ImGuiIO& io = ImGui::GetIO();
            ImVec2 viewport = ImVec2(io.DisplaySize.x, io.DisplaySize.y);

            ImGui::SetNextWindowPos(ImVec2(m_cursor_x_lastRightClick, m_cursor_y_lastRightClick));

            ImGui::Begin("ProbeInteractionTools", &my_tool_active, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar);

            if (ImGui::Button("AddProbe"))
            {
                // TODO add interaction to stack
                m_open_context_menu = false;
            }

            if (ImGui::Button("MoveProbe")) {
                // TODO add interaction to stack

                m_open_context_menu = false;
            }

            ImGui::End();
        }
    }

    return true;
}

bool megamol::probe_gl::ProbeInteraction::getInteractionCollection(core::Call& call) {
    auto cic = dynamic_cast<CallProbeInteraction*>(&call);
    if (cic == NULL) return false;

    if (!m_interactions->accessPendingManipulations().empty())
    {
        ++m_version;
    }

    cic->setData(m_interactions,m_version);

    return true; 
}

bool megamol::probe_gl::ProbeInteraction::getInteractionMetaData(core::Call& call) {
    return true;
}
