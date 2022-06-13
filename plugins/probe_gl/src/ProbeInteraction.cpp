#include "ProbeInteraction.h"

#include "ProbeGlCalls.h"
#include "compositing_gl/CompositingCalls.h"

#include "mmcore/CoreInstance.h"
#include "mmstd/event/EventCall.h"

#include "ProbeEvents.h"

#include <imgui.h>
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui_impl_opengl3.h"
#include "imgui_stdlib.h"
#include <imgui_internal.h>

megamol::probe_gl::ProbeInteraction::ProbeInteraction()
        : Renderer3DModuleGL()
        , m_version(0)
        , m_cursor_x(0)
        , m_cursor_y(0)
        , m_cursor_x_lastRightClick(0)
        , m_cursor_y_lastRightClick(0)
        , m_open_context_menu(false)
        , m_open_showMenu_dropdown(false)
        , m_open_probeMenu_dropdown(false)
        , m_open_dataMenu_dropdown(false)
        , m_open_dataFilterByDepth_popup(false)
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

    this->m_event_write_slot.SetCompatibleCall<megamol::core::CallEventDescription>();
    this->MakeSlotAvailable(&this->m_event_write_slot);
}

megamol::probe_gl::ProbeInteraction::~ProbeInteraction() {
    this->Release();
}

bool megamol::probe_gl::ProbeInteraction::OnMouseButton(
    core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {

    // Get event storage to queue new events
    auto call_event_storage = this->m_event_write_slot.CallAs<core::CallEvent>();
    if (call_event_storage == NULL)
        return false;
    if ((!(*call_event_storage)(0)))
        return false;
    auto event_collection = call_event_storage->getData();

    if (button == core::view::MouseButton::BUTTON_LEFT && action == core::view::MouseButtonAction::PRESS &&
        mods.none()) {
        m_selected_probes.clear();

        if (last_active_probe_id > 0) {
            // create new selection
            auto evt = std::make_unique<ProbeSelectExclusive>(
                this->GetCoreInstance()->GetFrameID(), static_cast<uint32_t>(last_active_probe_id));
            event_collection->add<ProbeSelectExclusive>(std::move(evt));
            megamol::core::utility::log::Log::DefaultLog.WriteMsg(
                megamol::core::utility::log::Log::LEVEL_INFO, "Selected probe: %d\n", last_active_probe_id);
        } else {
            auto evt = std::make_unique<ProbeClearSelection>(this->GetCoreInstance()->GetFrameID());
            event_collection->add<ProbeClearSelection>(std::move(evt));
        }

        return true;

        m_open_context_menu = false;
    } else if (button == core::view::MouseButton::BUTTON_LEFT && action == core::view::MouseButtonAction::PRESS &&
               mods.test(core::view::Modifier::SHIFT)) {
        if (last_active_probe_id > 0) {
            // add to current selection
            m_selected_probes.push_back(last_active_probe_id);
            auto evt = std::make_unique<ProbeSelectToggle>(
                this->GetCoreInstance()->GetFrameID(), static_cast<uint32_t>(last_active_probe_id));
            event_collection->add<ProbeSelectToggle>(std::move(evt));

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

bool megamol::probe_gl::ProbeInteraction::create() {
    return true;
}

void megamol::probe_gl::ProbeInteraction::release() {}

bool megamol::probe_gl::ProbeInteraction::GetExtents(mmstd_gl::CallRender3DGL& call) {
    return true;
}

bool megamol::probe_gl::ProbeInteraction::Render(mmstd_gl::CallRender3DGL& call) {

    mmstd_gl::CallRender3DGL* cr = dynamic_cast<mmstd_gl::CallRender3DGL*>(&call);
    if (cr == NULL)
        return false;

    // obtain camera information
    core::view::Camera cam = cr->GetCamera();
    m_view_mx_cpy = cam.getViewMatrix();
    m_proj_mx_cpy = cam.getProjectionMatrix();

    auto call_probe_fbo = this->m_probe_fbo_slot.CallAs<compositing::CallFramebufferGL>();
    auto call_hull_fbo = this->m_hull_fbo_slot.CallAs<compositing::CallFramebufferGL>();
    auto call_glyph_fbo = this->m_glyph_fbo_slot.CallAs<compositing::CallFramebufferGL>();

    float hull_depth_pixel_data = 0.0f;
    float probe_depth_pixel_data = 0.0f;
    float glyph_depth_pixel_data = 0.0f;

    GLint glyph_objId_pixel_data = -1;
    GLint probe_objId_pixel_data = -1;

    if (call_hull_fbo != nullptr) {
        if ((!(*call_hull_fbo)(0))) {
            return false;
        }
        auto hull_fbo = call_hull_fbo->getData();

        hull_fbo->bindToRead(2);
        {
            // auto err = glGetError();
            // std::cerr << err << std::endl;
        }
        // get depth at cursor location from framebuffer's second color attachment
        // TODO check if cursor position is within framebuffer pixel range?
        glReadPixels(static_cast<GLint>(this->m_cursor_x), hull_fbo->getHeight() - static_cast<GLint>(this->m_cursor_y),
            1, 1, GL_RED, GL_FLOAT, &hull_depth_pixel_data);
        {
            // auto err = glGetError();
            // std::cerr << err << std::endl;
        }
    }

    if (call_probe_fbo != nullptr) {
        if ((!(*call_probe_fbo)(0))) {
            return false;
        }
        auto probe_fbo = call_probe_fbo->getData();

        probe_fbo->bindToRead(2);
        {
            // auto err = glGetError();
            // std::cerr << err << std::endl;
        }
        // get depth at cursor location from framebuffer's second color attachment
        // TODO check if cursor position is within framebuffer pixel range?
        glReadPixels(static_cast<GLint>(this->m_cursor_x),
            probe_fbo->getHeight() - static_cast<GLint>(this->m_cursor_y), 1, 1, GL_RED, GL_FLOAT,
            &probe_depth_pixel_data);
        {
            // auto err = glGetError();
            // std::c+err << err << std::endl;
        }

        probe_fbo->bindToRead(3);
        {
            // auto err = glGetError();
            // std::cerr << err << std::endl;
        }
        // get object id at cursor location from framebuffer's third color attachment
        // TODO check if cursor position is within framebuffer pixel range?
        glReadPixels(static_cast<GLint>(this->m_cursor_x),
            probe_fbo->getHeight() - static_cast<GLint>(this->m_cursor_y), 1, 1, GL_RED_INTEGER, GL_INT,
            &probe_objId_pixel_data);
        {
            // auto err = glGetError();
            // std::cerr << err << std::endl;
        }
    }

    if (call_glyph_fbo != nullptr) {
        if ((!(*call_glyph_fbo)(0))) {
            return false;
        }
        auto glyph_fbo = call_glyph_fbo->getData();


        glyph_fbo->bindToRead(2);
        {
            // auto err = glGetError();
            // std::cerr << err << std::endl;
        }
        // get depth at cursor location from framebuffer's second color attachment
        // TODO check if cursor position is within framebuffer pixel range?
        glReadPixels(static_cast<GLint>(this->m_cursor_x),
            glyph_fbo->getHeight() - static_cast<GLint>(this->m_cursor_y), 1, 1, GL_RED, GL_FLOAT,
            &glyph_depth_pixel_data);
        {
            // auto err = glGetError();
            // std::cerr << err << std::endl;
        }

        glyph_fbo->bindToRead(3);
        {
            // auto err = glGetError();
            // std::cerr << err << std::endl;
        }
        // get object id at cursor location from framebuffer's thrid color attachment
        // TODO check if cursor position is within framebuffer pixel range?
        glReadPixels(static_cast<GLint>(this->m_cursor_x),
            glyph_fbo->getHeight() - static_cast<GLint>(this->m_cursor_y), 1, 1, GL_RED_INTEGER, GL_INT,
            &glyph_objId_pixel_data);
        {
            // auto err = glGetError();
            // std::cerr << err << std::endl;
        }
    }

    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

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

    if (hull_depth_pixel_data > 0.0f && depth > hull_depth_pixel_data) {
        objId = -1;
    }

    // std::cout << "Object ID at " << m_cursor_x << "," << m_cursor_y << " : " << objId << std::endl;

    // Get event storage to queue new events
    auto call_event_storage = this->m_event_write_slot.CallAs<core::CallEvent>();
    if (call_event_storage == NULL)
        return false;
    if ((!(*call_event_storage)(0)))
        return false;
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
        //    bool my_tool_active = true;
        //
        //    auto ctx = reinterpret_cast<ImGuiContext*>(this->GetCoreInstance()->GetCurrentImGuiContext());
        //    if (ctx != nullptr) {
        //        ImGui::SetCurrentContext(ctx);
        //
        //        ImGuiIO& io = ImGui::GetIO();
        //        ImVec2 viewport = ImVec2(io.DisplaySize.x, io.DisplaySize.y);
        //
        //        ImGui::SetNextWindowPos(ImVec2(m_cursor_x_lastRightClick, m_cursor_y_lastRightClick));
        //
        //        ImGui::Begin(
        //            "ProbeInteractionTools", &my_tool_active, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar);
        //
        //        if (ImGui::Button("AddProbe")) {
        //            // TODO add interaction to stack
        //            m_open_context_menu = false;
        //        }
        //
        //        if (ImGui::Button("MoveProbe")) {
        //            // TODO add interaction to stack
        //
        //            m_open_context_menu = false;
        //        }
        //
        //        ImGui::End();
        //    }
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

                m_open_probeMenu_dropdown = false;
                m_open_dataMenu_dropdown = false;
            }
            ImGui::End();

            ImGui::SetNextWindowPos(ImVec2(450, 20));
            ImGui::Begin("ProbeMenuButton", &my_tool_active,
                ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground);
            if (ImGui::Button("Probe", ImVec2(75, 20))) {
                m_open_probeMenu_dropdown = !m_open_probeMenu_dropdown;

                m_open_showMenu_dropdown = false;
                m_open_dataMenu_dropdown = false;
            }
            ImGui::End();

            ImGui::SetNextWindowPos(ImVec2(600, 20));
            ImGui::Begin("DataMenuButton", &my_tool_active,
                ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground);
            if (ImGui::Button("Data", ImVec2(75, 20))) {
                m_open_dataMenu_dropdown = !m_open_dataMenu_dropdown;

                m_open_showMenu_dropdown = false;
                m_open_probeMenu_dropdown = false;
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

                ImGui::SetNextWindowPos(ImVec2(460, 50));

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

            if (m_open_dataMenu_dropdown) {
                ImGui::SetNextWindowPos(ImVec2(610, 50));

                ImGui::Begin("DataDropdown", &my_tool_active,
                    ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground);

                if (ImGui::Button("Filter By Probe", ImVec2(75, 20))) {
                    m_open_dataMenu_dropdown = false;

                    auto evt = std::make_unique<DataFilterByProbeSelection>(this->GetCoreInstance()->GetFrameID());
                    event_collection->add<DataFilterByProbeSelection>(std::move(evt));
                }

                if (ImGui::Button("Filter By Probing Depth", ImVec2(75, 20))) {
                    m_open_dataMenu_dropdown = false;
                    m_open_dataFilterByDepth_popup = true;
                }

                if (ImGui::Button("Clear Filter", ImVec2(75, 20))) {
                    m_open_dataMenu_dropdown = false;

                    auto evt = std::make_unique<DataClearFilter>(this->GetCoreInstance()->GetFrameID());
                    event_collection->add<DataClearFilter>(std::move(evt));
                }

                ImGui::Separator();

                ImGui::End();
            }

            if (m_open_dataFilterByDepth_popup) {

                auto ctx = reinterpret_cast<ImGuiContext*>(this->GetCoreInstance()->GetCurrentImGuiContext());
                if (ctx != nullptr) {
                    ImGui::SetCurrentContext(ctx);

                    ImGuiIO& io = ImGui::GetIO();
                    ImVec2 viewport = ImVec2(io.DisplaySize.x, io.DisplaySize.y);

                    ImGui::SetNextWindowPos(ImVec2(750, 150));

                    ImGui::Begin("Filter By Probing Depth", &m_open_dataFilterByDepth_popup, ImGuiWindowFlags_NoResize);

                    static float probing_depth = 5.0f;
                    if (ImGui::SliderFloat("DataFilter::robingDepth", &probing_depth, 0.0f, 100.0f)) {
                        auto evt = std::make_unique<DataFilterByProbingDepth>(
                            this->GetCoreInstance()->GetFrameID(), probing_depth);
                        event_collection->add<DataFilterByProbingDepth>(std::move(evt));
                    }

                    //if (ImGui::Button("Close")) {

                    //    m_open_dataFilterByDepth_popup = false;
                    //}

                    ImGui::End();
                }
            }
        }
    }

    return true;
}

bool megamol::probe_gl::ProbeInteraction::getInteractionMetaData(core::Call& call) {
    return true;
}
