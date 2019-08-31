#include "stdafx.h"
#include "Render3DUI.h"
#include "mesh/MeshCalls.h"

bool megamol::mesh::Render3DUI::OnMouseButton(
    core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {

    if (button == core::view::MouseButton::BUTTON_LEFT && action == core::view::MouseButtonAction::PRESS)
    {
        // TODO add select interaction
        if (m_cursor_on_interaction_obj.first) {

            m_active_interaction_obj = {true, m_cursor_on_interaction_obj.second};

            return true;
        }
    } 
    else if (button == core::view::MouseButton::BUTTON_LEFT && action == core::view::MouseButtonAction::RELEASE)
    {
        // TODO add deselect interaction
        m_active_interaction_obj = {false, -1};
    }

    return false;
}

bool megamol::mesh::Render3DUI::OnMouseMove(double x, double y) {

    double dx = x - this->m_cursor_x;
    double dy = y - this->m_cursor_y;

    this->m_cursor_x = x;
    this->m_cursor_y = y;

    if (m_fbo != nullptr)
    {
        dx = dx / m_fbo->getWidth();
        dy = -dy / m_fbo->getHeight();

    }
    
    if (m_active_interaction_obj.first)
    {
        // TODO check interaction type of active object
        Call3DInteraction* ci = this->m_3DInteraction_callerSlot.CallAs<Call3DInteraction>();
        if (ci == NULL) return false;
        auto interaction_collection = ci->getData();

        auto available_interactions =
            interaction_collection->getAvailableInteractions(static_cast<uint32_t>(m_active_interaction_obj.second));

        for (auto& interaction : available_interactions)
        {
            if (interaction.type == InteractionType::MOVE_ALONG_AXIS)
            {

                glm::vec4 tgt_pos(interaction.origin_x,interaction.origin_y,interaction.origin_z,1.0f);
                //vislib::math::Vector<float, 3> cam_pos = GCoreComponents::transformManager().getWorldPosition(active_camera);

                // Compute tgt pos and tgt + transform axisvector in screenspace
                glm::vec4 obj_ss = m_proj_mx_cpy * m_view_mx_cpy * tgt_pos;
                obj_ss /= obj_ss.w;

                glm::vec4 transform_tgt = tgt_pos + glm::vec4(interaction.axis_x, interaction.axis_y, interaction.axis_z, 0.0f);
                glm::vec4 transform_tgt_ss = m_proj_mx_cpy * m_view_mx_cpy * transform_tgt;
                transform_tgt_ss /= transform_tgt_ss.w;

                glm::vec2 transform_axis_ss =
                    glm::vec2(transform_tgt_ss.x, transform_tgt_ss.y) -
                    glm::vec2(obj_ss.x, obj_ss.y);

                glm::vec2 mouse_move =
                    glm::vec2(static_cast<float>(dx), static_cast<float>(dy)) * 2.0f;

                float scale = 0.0f;

                if (transform_axis_ss.length() > 0.0)
                {
                    auto mm_lenght = mouse_move.length();
                    auto ta_ss_length = transform_axis_ss.length();

                    auto mm_norm = glm::normalize(mouse_move);
                    auto ta_ss_norm = glm::normalize(transform_axis_ss);

                    scale = glm::dot(mm_norm, ta_ss_norm);
                    scale *= (mm_lenght / ta_ss_length);
                }

                std::cout << "Adding move manipulation: " << interaction.axis_x << " " << interaction.axis_y << " "
                          << interaction.axis_z << " " << scale << std::endl;

                interaction_collection->accessPendingManipulations().push(ThreeDimensionalManipulation{
                    InteractionType::MOVE_ALONG_AXIS, static_cast<uint32_t>(m_active_interaction_obj.second),
                    interaction.axis_x, interaction.axis_y, interaction.axis_z, scale});
                // TODO add manipulation task with scale * axis

            }
        }
    }

    // TODO compute manipulation based on mouse movement 

    return false; 
}

megamol::mesh::Render3DUI::Render3DUI()
    : RenderMDIMesh()
    , m_cursor_x(0.0)
    , m_cursor_y(0.0)
    , m_cursor_on_interaction_obj({false,-1})
    , m_fbo(nullptr)
    , m_3DInteraction_callerSlot(
          "getInteraction", "Connects to the interaction slot of a suitable RenderTaskDataSource") {
    this->m_3DInteraction_callerSlot.SetCompatibleCall<Call3DInteractionDescription>();
    this->MakeSlotAvailable(&this->m_3DInteraction_callerSlot);
}

megamol::mesh::Render3DUI::~Render3DUI() { this->Release(); }

bool megamol::mesh::Render3DUI::create() { return true; }

void megamol::mesh::Render3DUI::release() { m_fbo.reset(); }

bool megamol::mesh::Render3DUI::GetExtents(core::view::CallRender3D_2& call) {
    RenderMDIMesh::GetExtents(call);

    return true;
}

bool megamol::mesh::Render3DUI::Render(core::view::CallRender3D_2& call) {

    core::view::CallRender3D_2* cr = dynamic_cast<core::view::CallRender3D_2*>(&call);
    if (cr == NULL) return false;

    // obtain camera information
    core::view::Camera_2 cam(cr->GetCamera());
    cam_type::snapshot_type snapshot;
    cam_type::matrix_type view_tmp, proj_tmp;
    cam.calc_matrices(snapshot, view_tmp, proj_tmp, core::thecam::snapshot_content::all);
    m_view_mx_cpy = view_tmp;
    m_proj_mx_cpy = proj_tmp;

    // check for interacton call get access to interaction collection
    Call3DInteraction* ci = this->m_3DInteraction_callerSlot.CallAs<Call3DInteraction>();
    if (ci == NULL) return false;
    if ((!(*ci)(0))) return false;
    auto interaction_collection = ci->getData();

    GLfloat viewport[4];
    glGetFloatv(GL_VIEWPORT, viewport);

    if (m_fbo == nullptr) {    
        m_fbo = std::make_unique<glowl::FramebufferObject>(viewport[2], viewport[3],true);
        m_fbo->createColorAttachment(GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT); // output image
        m_fbo->createColorAttachment(GL_R32I, GL_RED, GL_INT); // object ids
    }
    else if (m_fbo->getWidth() != viewport[2] || m_fbo->getHeight() != viewport[3]){
        m_fbo->resize(viewport[2], viewport[3]);
    }

    m_fbo->bind();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    GLint in[1] = {0};
    glClearBufferiv(GL_COLOR, 1, in);

    RenderMDIMesh::Render(call);

    //glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // bind fbo to read buffer for retrieving pixel data and bliting to default framebuffer
    m_fbo->bindToRead(1);

    {
        auto err = glGetError();
        std::cerr << err << std::endl;
    }

    // get object id at cursor location from framebuffer's second color attachment
    GLint pixel_data = -1;
    // TODO check if cursor position is within framebuffer pixel range?
    glReadPixels(static_cast<GLint>(this->m_cursor_x), m_fbo->getHeight() - static_cast<GLint>(this->m_cursor_y), 1, 1,
        GL_RED_INTEGER,
        GL_INT, &pixel_data);

    {
        auto err = glGetError();
        std::cerr << err << std::endl;
    }

    if (pixel_data > 0)
    {
        m_cursor_on_interaction_obj = {true, pixel_data};
    }
    else
    {
        m_cursor_on_interaction_obj = {false,-1};
    }

    if (interaction_collection != nullptr) {
        if (m_active_interaction_obj.first) {
            interaction_collection->accessPendingManipulations().push(ThreeDimensionalManipulation{
                InteractionType::HIGHLIGHT, static_cast<uint32_t>(m_active_interaction_obj.second), 0.0f, 0.0f, 0.0f, 0.0f});
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    m_fbo->bindToRead(0);

    // blit from fbo to default framebuffer
    //glReadBuffer(GL_COLOR_ATTACHMENT0);
    glBlitFramebuffer(0, 0, m_fbo->getWidth(), m_fbo->getHeight(), 0, 0, viewport[2], viewport[3],
        GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT, GL_NEAREST);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return true;
}
