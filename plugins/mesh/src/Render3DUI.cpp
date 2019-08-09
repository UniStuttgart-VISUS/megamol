#include "stdafx.h"
#include "Render3DUI.h"

#include "mesh/Call3DInteraction.h"

bool megamol::mesh::Render3DUI::OnMouseButton(
    core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {

    if (button == core::view::MouseButton::BUTTON_LEFT && action == core::view::MouseButtonAction::PRESS)
    {
        // TODO add select interaction
    } 
    else if (button == core::view::MouseButton::BUTTON_LEFT && action == core::view::MouseButtonAction::RELEASE)
    {
        // TODO add deselect interaction
    }

    return false;
}

bool megamol::mesh::Render3DUI::OnMouseMove(double x, double y) {

    this->m_cursor_x = x;
    this->m_cursor_y = y;

    return false; 
}

megamol::mesh::Render3DUI::Render3DUI()
    : RenderMDIMesh()
    , m_cursor_x(0.0)
    , m_cursor_y(0.0)
    , m_fbo(nullptr)
    , m_3DInteraction_callerSlot(
          "getInteraction", "Connects to the interaction slot of a suitable RenderTaskDataSource") {
    this->m_3DInteraction_callerSlot.SetCompatibleCall<Call3DInteractionDescription>();
    this->MakeSlotAvailable(&this->m_3DInteraction_callerSlot);
}

megamol::mesh::Render3DUI::~Render3DUI() { this->Release(); }

bool megamol::mesh::Render3DUI::create() { return true; }

void megamol::mesh::Render3DUI::release() { m_fbo.reset(); }

bool megamol::mesh::Render3DUI::GetExtents(core::Call& call) {
    RenderMDIMesh::GetExtents(call);

    return true;
}

bool megamol::mesh::Render3DUI::Render(core::Call& call) {

    // check for interacton call get access to interaction collection
    Call3DInteraction* ci = this->m_3DInteraction_callerSlot.CallAs<Call3DInteraction>();
    if (ci == NULL) return false;
    if ((!(*ci)(0))) return false;
    auto interaction_collection = ci->getInteractionCollection();

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

    // TODO translate to meaningful interaction type

    // TODO add highlight interaction

    std::cout << "Hello: " << pixel_data << " at " << this->m_cursor_x << " " << this->m_cursor_y << std::endl;

    if (interaction_collection != nullptr) {
        if (pixel_data > 0) {
            interaction_collection->accessPendingManipulations().push(ThreeDimensionalManipulation{
                InteractionType::HIGHLIGHT, static_cast<uint32_t>(pixel_data), 0.0f, 0.0f, 0.0f, 0.0f});
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
