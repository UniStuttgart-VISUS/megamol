#include "ProbeInteraction.h"

#include "ProbeGlCalls.h"
#include "compositing/CompositingCalls.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/EventCall.h"

#include "ProbeEvents.h"

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
    , m_open_showMenu_dropdown(false)
    , m_open_probeMenu_dropdown(false)
    , m_open_dataMenu_dropdown(false)
    , m_show_probes(true)
    , m_show_hull(true)
    , m_show_glyphs(true)
    , last_active_probe_id(-1)
    , m_probe_fbo_slot("getProbeFBO", "")
    , m_hull_fbo_slot("getHullFBO", "")
    , m_glyph_fbo_slot("getGlyphFBO", "")
    , m_event_write_slot("deployInteractionEvents", "") {
    this->m_probe_fbo_slot.SetCompatibleCall<compositing::CallFramebufferGLDescription>();
    this->MakeSlotAvailable(&this->m_probe_fbo_slot);

    this->m_hull_fbo_slot.SetCompatibleCall<compositing::CallFramebufferGLDescription>();
    this->MakeSlotAvailable(&this->m_hull_fbo_slot);

    this->m_glyph_fbo_slot.SetCompatibleCall<compositing::CallFramebufferGLDescription>();
    this->MakeSlotAvailable(&this->m_glyph_fbo_slot);

    this->m_event_write_slot.SetCompatibleCall<megamol::core::EventCallDescription>();
    this->MakeSlotAvailable(&this->m_event_write_slot);
}

megamol::probe_gl::ProbeInteraction::~ProbeInteraction() { this->Release(); }

bool megamol::probe_gl::ProbeInteraction::OnMouseButton(
    core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {

    // Get event storage to queue new events
    auto call_event_storage = this->m_event_write_slot.CallAs<core::EventCall>();
    if (call_event_storage == NULL) return false;
    if ((!(*call_event_storage)(0))) return false;
    auto event_collection = call_event_storage->getData();

    if (button == core::view::MouseButton::BUTTON_LEFT && action == core::view::MouseButtonAction::PRESS &&
        mods.none()) {
        m_selected_probes.clear();
        auto evt = std::make_unique<ClearSelection>(this->GetCoreInstance()->GetFrameID());
        event_collection->add<ClearSelection>(std::move(evt));

        if (last_active_probe_id > 0) {
            // create new selection
            auto evt = std::make_unique<Select>(
                this->GetCoreInstance()->GetFrameID(), static_cast<uint32_t>(last_active_probe_id));
            event_collection->add<Select>(std::move(evt));

            return true;
        }

        m_open_context_menu = false;
    } else if (button == core::view::MouseButton::BUTTON_LEFT && action == core::view::MouseButtonAction::PRESS &&
               mods.test(core::view::Modifier::SHIFT)) {
        if (last_active_probe_id > 0) {
            // add to current selection
            m_selected_probes.push_back(last_active_probe_id);
            auto evt = std::make_unique<Select>(
                this->GetCoreInstance()->GetFrameID(), static_cast<uint32_t>(last_active_probe_id));
            event_collection->add<Select>(std::move(evt));

            return true;
        }
    } else if (button == core::view::MouseButton::BUTTON_RIGHT) // && action == core::view::MouseButtonAction::PRESS)
    {
        m_mouse_button_states[button] = (action == core::view::MouseButtonAction::PRESS) ? true
                                        : (action == core::view::MouseButtonAction::RELEASE)
                                            ? false
                                            : m_mouse_button_states[button];


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

void megamol::probe_gl::ProbeInteraction::release() {}

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

    // TODO read obj ids from FBOs...

    // bind fbo to read buffer for retrieving pixel data and bliting to default framebuffer
    hull_fbo->bindToRead(2);
    {
        // auto err = glGetError();
        // std::cerr << err << std::endl;
    }
    // get object id at cursor location from framebuffer's second color attachment
    float hull_depth_pixel_data = 0.0;
    // TODO check if cursor position is within framebuffer pixel range?
    glReadPixels(static_cast<GLint>(this->m_cursor_x), probe_fbo->getHeight() - static_cast<GLint>(this->m_cursor_y), 1,
        1, GL_RED, GL_FLOAT, &hull_depth_pixel_data);
    {
        // auto err = glGetError();
        // std::cerr << err << std::endl;
    }

    // bind fbo to read buffer for retrieving pixel data and bliting to default framebuffer
    probe_fbo->bindToRead(2);
    {
        // auto err = glGetError();
        // std::cerr << err << std::endl;
    }
    // get object id at cursor location from framebuffer's second color attachment
    float probe_depth_pixel_data = 0.0;
    // TODO check if cursor position is within framebuffer pixel range?
    glReadPixels(static_cast<GLint>(this->m_cursor_x), probe_fbo->getHeight() - static_cast<GLint>(this->m_cursor_y), 1,
        1, GL_RED, GL_FLOAT, &probe_depth_pixel_data);
    {
        // auto err = glGetError();
        // std::cerr << err << std::endl;
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
        // auto err = glGetError();
        // std::cerr << err << std::endl;
    }
    // get object id at cursor location from framebuffer's second color attachment
    GLint probe_objId_pixel_data = -1;
    // TODO check if cursor position is within framebuffer pixel range?
    glReadPixels(static_cast<GLint>(this->m_cursor_x), probe_fbo->getHeight() - static_cast<GLint>(this->m_cursor_y), 1,
        1, GL_RED_INTEGER, GL_INT, &probe_objId_pixel_data);
    {
        // auto err = glGetError();
        // std::cerr << err << std::endl;
    }

    // std::cout << "Object ID at " << m_cursor_x << "," << m_cursor_y << " : " << glyph_objId_pixel_data << std::endl;

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
    } else if (probe_depth_pixel_data > 0.0) {
        objId = probe_objId_pixel_data;
        depth = probe_depth_pixel_data;
    } else if (glyph_depth_pixel_data > 0.0) {
        objId = glyph_objId_pixel_data;
        depth = glyph_depth_pixel_data;
    }


    if (depth > hull_depth_pixel_data) {
        objId = -1;
    }

    // std::cout << "Object ID at " << m_cursor_x << "," << m_cursor_y << " : " << objId << std::endl;

    // Get event storage to queue new events
    auto call_event_storage = this->m_event_write_slot.CallAs<core::EventCall>();
    if (call_event_storage == NULL) return false;
    if ((!(*call_event_storage)(0))) return false;
    auto event_collection = call_event_storage->getData();

    if (objId > -1) {
        auto evt =
            std::make_unique<ProbeHighlight>(this->GetCoreInstance()->GetFrameID(), static_cast<uint32_t>(objId));
        event_collection->add<ProbeHighlight>(std::move(evt));
    }
    if (last_active_probe_id > 0 && last_active_probe_id != objId) {
        auto evt = std::make_unique<ProbeDehighlight>(
            this->GetCoreInstance()->GetFrameID(), static_cast<uint32_t>(last_active_probe_id));
        event_collection->add<ProbeDehighlight>(std::move(evt));
    }

    last_active_probe_id = objId;


    if (m_open_context_menu) {
        bool my_tool_active = true;

        auto ctx = reinterpret_cast<ImGuiContext*>(this->GetCoreInstance()->GetCurrentImGuiContext());
        if (ctx != nullptr) {
            ImGui::SetCurrentContext(ctx);

            ImGuiIO& io = ImGui::GetIO();
            ImVec2 viewport = ImVec2(io.DisplaySize.x, io.DisplaySize.y);

            ImGui::SetNextWindowPos(ImVec2(m_cursor_x_lastRightClick, m_cursor_y_lastRightClick));

            ImGui::Begin(
                "ProbeInteractionTools", &my_tool_active, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar);

            if (ImGui::Button("AddProbe")) {
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

    // Add toolbar in Blender style
    {
        bool my_tool_active = true;

        auto ctx = reinterpret_cast<ImGuiContext*>(this->GetCoreInstance()->GetCurrentImGuiContext());
        if (ctx != nullptr) {
            ImGui::SetCurrentContext(ctx);

            ImGuiIO& io = ImGui::GetIO();
            ImVec2 viewport = ImVec2(io.DisplaySize.x, io.DisplaySize.y);

            ImGui::SetNextWindowPos(ImVec2(300, 20));
            ImGui::Begin("ShowMenuButton", &my_tool_active,
                ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground);
            if (ImGui::Button("Show", ImVec2(75, 20))) {
                m_open_showMenu_dropdown = !m_open_showMenu_dropdown;
            }
            ImGui::End();

            ImGui::SetNextWindowPos(ImVec2(400, 20));
            ImGui::Begin("ProbeMenuButton", &my_tool_active,
                ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground);
            if (ImGui::Button("Probe", ImVec2(75, 20))) {
                m_open_probeMenu_dropdown = !m_open_probeMenu_dropdown;
            }
            ImGui::End();

            ImGui::SetNextWindowPos(ImVec2(500, 20));
            ImGui::Begin("DataMenuButton", &my_tool_active,
                ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground);
            if (ImGui::Button("Data", ImVec2(75, 20))) {
                m_open_dataMenu_dropdown = !m_open_dataMenu_dropdown;
            }
            ImGui::End();


            if (m_open_showMenu_dropdown) {

                ImGui::SetNextWindowPos(ImVec2(310, 50));

                ImGui::Begin("ShowDropdown", &my_tool_active,
                    ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground);

                if (ImGui::Button("Probes", ImVec2(75, 20))) {
                    m_open_showMenu_dropdown = false;
                    m_show_probes = !m_show_probes;

                    auto evt = std::make_unique<ToggleShowProbes>(this->GetCoreInstance()->GetFrameID());
                    event_collection->add<ToggleShowProbes>(std::move(evt));
                }
                ImGui::SameLine();
                ImGui::PushID(0);
                if (ImGui::Checkbox("", &m_show_probes)) {
                    auto evt = std::make_unique<ToggleShowProbes>(this->GetCoreInstance()->GetFrameID());
                    event_collection->add<ToggleShowProbes>(std::move(evt));
                }
                ImGui::PopID();

                if (ImGui::Button("Hull", ImVec2(75, 20))) {
                    m_open_showMenu_dropdown = false;
                    m_show_hull = !m_show_hull;

                    auto evt = std::make_unique<ToggleShowHull>(this->GetCoreInstance()->GetFrameID());
                    event_collection->add<ToggleShowHull>(std::move(evt));
                }
                ImGui::SameLine();
                if (ImGui::Checkbox("", &m_show_hull)) {
                    auto evt = std::make_unique<ToggleShowHull>(this->GetCoreInstance()->GetFrameID());
                    event_collection->add<ToggleShowHull>(std::move(evt));
                }

                if (ImGui::Button("Glyphs", ImVec2(75, 20))) {
                    m_open_showMenu_dropdown = false;
                    m_show_glyphs = !m_show_glyphs;

                    auto evt = std::make_unique<ToggleShowGlyphs>(this->GetCoreInstance()->GetFrameID());
                    event_collection->add<ToggleShowGlyphs>(std::move(evt));
                }
                ImGui::SameLine();
                ImGui::PushID(2);
                if (ImGui::Checkbox("", &m_show_glyphs)) {
                    auto evt = std::make_unique<ToggleShowGlyphs>(this->GetCoreInstance()->GetFrameID());
                    event_collection->add<ToggleShowGlyphs>(std::move(evt));
                }
                ImGui::PopID();

                ImGui::Separator();

                ImGui::End();
            }

            if (m_open_probeMenu_dropdown) {

                ImGui::SetNextWindowPos(ImVec2(410, 50));

                ImGui::Begin("ProbeDropdown", &my_tool_active,
                    ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground);

                if (ImGui::Button("Add Probe", ImVec2(75, 20))) {
                    m_open_probeMenu_dropdown = false;

                    // m_interactions->accessPendingManipulations().push_back(ProbeManipulation{
                    //    InteractionType::TOGGLE_SHOW_PROBES, static_cast<uint32_t>(last_active_probe_id), 0, 0, 0});
                }

                if (ImGui::Button("Deselect All", ImVec2(75, 20))) {
                    m_open_probeMenu_dropdown = false;

                    // TODO for each selected probe, add deselect interaction

                    // m_interactions->accessPendingManipulations().push_back(ProbeManipulation{
                    //    InteractionType::TOGGLE_SHOW_PROBES, static_cast<uint32_t>(last_active_probe_id), 0, 0, 0});
                }

                ImGui::Separator();

                ImGui::End();
            }
        }
    }

    return true;
}

bool megamol::probe_gl::ProbeInteraction::getInteractionMetaData(core::Call& call) { return true; }
