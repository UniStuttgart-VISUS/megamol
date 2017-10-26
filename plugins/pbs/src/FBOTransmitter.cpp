#include "stdafx.h"
#include "FBOTransmitter.h"

#include "mmcore/param/IntParam.h"
#include "mmcore/view/CallRenderView.h"

using namespace megamol;
using namespace megamol::pbs;


FBOTransmitter::FBOTransmitter(void)
    : core::Module(),
      core::job::AbstractJob(),
      fboWidthSlot("width", "Sets width of FBO"),
      fboHeightSlot("height", "Sets height of FBO"),
      zmq_ctx(1),
      zmq_socket(zmq_ctx, zmq::socket_type::push) {
}


FBOTransmitter::~FBOTransmitter(void) {
}


bool FBOTransmitter::create(void) {

    return true;
}


void FBOTransmitter::release(void) {
}


void FBOTransmitter::BeforeRender(core::view::AbstractView *view) {
    view->UnregisterHook(this);

    core::view::CallRenderView crv;

    // create FBO
    int width = this->fboWidthSlot.Param<core::param::IntParam>()->Value();
    int height = this->fboHeightSlot.Param<core::param::IntParam>()->Value();

    auto color_rbo = createRBO(GL_RGBA8, width, height);
    auto depth_rbo = createRBO(GL_DEPTH_COMPONENT24, width, height);
    auto fbo = createFBOFromRBO(color_rbo, depth_rbo);

    // bind FBO
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, width, height);
    crv.SetOutputBuffer(fbo, width, height);

    view->OnRenderView(crv);
    glFlush();

    std::vector<unsigned char> color_buf(width * height * 4);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, color_buf.data());

    std::vector<unsigned char> depth_buf(width * height * 3);
    glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT24, GL_UNSIGNED_BYTE, depth_buf.data());

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    // unbind FBO

    // send buffers over socket
    if (this->zmq_socket.connected()) {
        // do stuff
        this->zmq_socket.send(zmq::message_t(&MSG_STARTFRAME, sizeof(int)));
        this->zmq_socket.send(zmq::message_t(&MSG_SENDVIEWPORT, sizeof(int)));
        int viewport[] = {0, 0, width, height};
        this->zmq_socket.send(zmq::message_t(viewport, sizeof(int) * 4));
        this->zmq_socket.send(zmq::message_t(&MSG_SENDDATA, sizeof(int)));
        this->zmq_socket.send(color_buf.begin(), color_buf.end());
        this->zmq_socket.send(depth_buf.begin(), depth_buf.end());
        this->zmq_socket.send(zmq::message_t(&MSG_ENDFRAME, sizeof(int)));
    } else {
        // connect socket
    }
}


GLuint createFBOFromTex(GLuint &color_tex, GLuint &depth_tex) {
    GLuint fbo = 0;

    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo); // probably need to store old FBO id
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_tex, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_tex, 0);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glReadBuffer(GL_COLOR_ATTACHMENT0);

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        glDeleteFramebuffers(1, &fbo);
        throw std::runtime_error("Could not create FBO\n");
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return fbo;
}


void deleteFBO(GLuint &fbo) {
    if (glIsFramebuffer(fbo)) {
        glDeleteFramebuffers(1, &fbo);
        fbo = 0;
    }
}


GLuint createTexture(GLint internal_format, GLsizei width, GLsizei height, GLenum format, GLenum type) {
    GLuint texture = 0;

    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, format, type, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    return texture;
}


void deleteRBO(GLuint &rbo) {
    if (glIsRenderbuffer(rbo)) {
        glDeleteRenderbuffers(1, &rbo);
        rbo = 0;
    }
}


GLuint createRBO(GLenum internal_format, GLsizei width, GLsizei height) {
    GLuint rbo = 0;

    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, internal_format, width, height);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    return rbo;
}


void deleteTexture(GLuint &texture) {
    if (glIsTexture(texture)) {
        glDeleteTextures(1, &texture);
        texture = 0;
    }
}


GLuint createFBOFromRBO(GLuint &color_rbo, GLuint &depth_rbo) {
    GLuint fbo = 0;

    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo); // probably need to store old FBO id
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color_rbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rbo);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glReadBuffer(GL_COLOR_ATTACHMENT0);

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        glDeleteFramebuffers(1, &fbo);
        throw std::runtime_error("Could not create FBO\n");
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return fbo;
}
