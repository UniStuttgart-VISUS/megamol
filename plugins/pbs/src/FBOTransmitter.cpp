/*
 * FBOTransmitter.cpp
 *
 * Copyright (C) 2017 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "FBOTransmitter.h"

#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/view/CallRenderView.h"

using namespace megamol;
using namespace megamol::pbs;


FBOTransmitter::FBOTransmitter(void)
    : core::Module(),
      core::job::AbstractJob(),
      fboWidthSlot("width", "Sets width of FBO"),
      fboHeightSlot("height", "Sets height of FBO"),
      ipAddressSlot("address", "IP address of reciever"),
      animTimeParamNameSlot("timeparamname", "Name of the time parameter"),
      zmq_ctx(1),
      zmq_socket(zmq_ctx, zmq::socket_type::push),
      ip_address("127.0.0.1:34242") {
}


FBOTransmitter::~FBOTransmitter(void) {
}


bool megamol::pbs::FBOTransmitter::IsRunning(void) const {
    return false;
}


bool megamol::pbs::FBOTransmitter::Start(void) {
    return false;
}


bool megamol::pbs::FBOTransmitter::Terminate(void) {
    return false;
}


bool FBOTransmitter::create(void) {
    // create FBO
    this->width = this->fboWidthSlot.Param<core::param::IntParam>()->Value();
    this->height = this->fboHeightSlot.Param<core::param::IntParam>()->Value();

    this->color_rbo = createRBO(GL_RGBA8, this->width, this->height);
    this->depth_rbo = createRBO(GL_DEPTH_COMPONENT24, width, height);
    this->fbo = createFBOFromRBO(this->color_rbo, this->depth_rbo);

    this->color_buf = std::vector<unsigned char>(this->width * this->height * 4);
    this->depth_buf = std::vector<unsigned char>(this->width * this->height * 3);

    return true;
}


void FBOTransmitter::release(void) {
    this->zmq_socket.disconnect(this->ip_address);
}


void FBOTransmitter::BeforeRender(core::view::AbstractView *view) {
    view->UnregisterHook(this);

    core::view::CallRenderView crv;

    float frameTime = -1.0f;
    core::param::ParamSlot *time = this->findTimeParam(view);
    if (time != NULL) {
        frameTime = time->Param<core::param::FloatParam>()->Value();
    }

    // save old framebuffer state
    GLint old_draw_fbo = 0, old_read_fbo = 0;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &old_read_fbo);
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &old_draw_fbo);

    // bind FBO
    glBindFramebuffer(GL_FRAMEBUFFER, this->fbo);
    glViewport(0, 0, this->width, this->height);
    crv.SetOutputBuffer(this->fbo, this->width, this->height);
    crv.SetTile(static_cast<float>(this->width), static_cast<float>(this->height),
        0.0f, 0.0f, static_cast<float>(this->width), static_cast<float>(this->height));
    crv.SetTime(frameTime);

    view->OnRenderView(crv);
    glFlush();
    
    glReadPixels(0, 0, this->width, this->height, GL_RGBA, GL_UNSIGNED_BYTE, this->color_buf.data());
    
    glReadPixels(0, 0, this->width, this->height, GL_DEPTH_COMPONENT24, GL_UNSIGNED_BYTE, this->depth_buf.data());

    // restore old framebuffer state
    glBindFramebuffer(GL_READ_FRAMEBUFFER, old_read_fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, old_draw_fbo);

    // send buffers over socket
    if (this->zmq_socket.connected()) {
        // do stuff
        if (!this->zmq_socket.send(zmq::message_t(&MSG_STARTFRAME, sizeof(int)))) {
            return;
        }
        if (!this->zmq_socket.send(zmq::message_t(&MSG_SENDVIEWPORT, sizeof(int)))) {
            return;
        }
        int viewport[] = {0, 0, this->width, this->height};
        if (!this->zmq_socket.send(zmq::message_t(viewport, sizeof(int) * 4))) {
            return;
        }
        if (!this->zmq_socket.send(zmq::message_t(&MSG_SENDDATA, sizeof(int)))) {
            return;
        }
        if (!this->zmq_socket.send(this->color_buf.begin(), this->color_buf.end())) {
            return;
        }
        if (!this->zmq_socket.send(this->depth_buf.begin(), this->depth_buf.end())) {
            return;
        }
        if (!this->zmq_socket.send(zmq::message_t(&MSG_ENDFRAME, sizeof(int)))) {
            return;
        }
    } else {
        // connect socket
        this->ip_address = this->ipAddressSlot.Param<core::param::StringParam>()->Value();
        this->connectSocket(this->ip_address);
    }
}


bool FBOTransmitter::resizeCallback(core::param::ParamSlot &p) {
    this->width = this->fboWidthSlot.Param<core::param::IntParam>()->Value();
    this->height = this->fboHeightSlot.Param<core::param::IntParam>()->Value();

    this->color_buf.resize(this->width*this->height*4);
    this->depth_buf.resize(this->width*this->height*3);

    deleteRBO(this->color_rbo);
    deleteRBO(this->depth_rbo);
    deleteFBO(this->fbo);

    this->color_rbo = createRBO(GL_RGBA8, width, height);
    this->depth_rbo = createRBO(GL_DEPTH_COMPONENT24, width, height);
    this->fbo = createFBOFromRBO(color_rbo, depth_rbo);

    return true;
}


bool FBOTransmitter::connectSocketCallback(core::param::ParamSlot &p) {
    if (this->zmq_socket.connected()) {
        this->zmq_socket.disconnect(this->ip_address);
    }

    this->ip_address = this->ipAddressSlot.Param<core::param::StringParam>()->Value();
    this->connectSocket(this->ip_address);

    return true;
}


void FBOTransmitter::connectSocket(std::string &address) {
    this->zmq_socket.connect("tcp://" + address);

    if (!this->zmq_socket.connected()) {
        throw std::runtime_error("Socket not connected after return of connect");
    }
}


core::param::ParamSlot *FBOTransmitter::findTimeParam(core::view::AbstractView *view) {
    vislib::TString name(this->animTimeParamNameSlot.Param<core::param::StringParam>()->Value());
    core::param::ParamSlot *timeSlot = nullptr;

    if (name.IsEmpty()) {
        timeSlot = dynamic_cast<core::param::ParamSlot*>(view->FindNamedObject("anim::time").get());
    } else {
        AbstractNamedObjectContainer * anoc = dynamic_cast<AbstractNamedObjectContainer*>(view->RootModule().get());
        timeSlot = dynamic_cast<core::param::ParamSlot*>(anoc->FindNamedObject(vislib::StringA(name)).get());
    }

    return timeSlot;
}


GLuint megamol::pbs::createFBOFromTex(GLuint &color_tex, GLuint &depth_tex) {
    GLuint fbo = 0;

    // save old framebuffer state
    GLint old_draw_fbo = 0, old_read_fbo = 0;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &old_read_fbo);
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &old_draw_fbo);

    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_tex, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_tex, 0);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glReadBuffer(GL_COLOR_ATTACHMENT0);

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        glDeleteFramebuffers(1, &fbo);
        throw std::runtime_error("Could not create FBO");
    }

    // restore old framebuffer state
    glBindFramebuffer(GL_READ_FRAMEBUFFER, old_read_fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, old_draw_fbo);

    return fbo;
}


void megamol::pbs::deleteFBO(GLuint &fbo) {
    if (glIsFramebuffer(fbo)) {
        glDeleteFramebuffers(1, &fbo);
        fbo = 0;
    }
}


GLuint megamol::pbs::createTexture(GLint internal_format, GLsizei width, GLsizei height, GLenum format, GLenum type) {
    GLuint texture = 0;

    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, format, type, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    return texture;
}


void megamol::pbs::deleteRBO(GLuint &rbo) {
    if (glIsRenderbuffer(rbo)) {
        glDeleteRenderbuffers(1, &rbo);
        rbo = 0;
    }
}


GLuint megamol::pbs::createRBO(GLenum internal_format, GLsizei width, GLsizei height) {
    GLuint rbo = 0;

    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, internal_format, width, height);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    return rbo;
}


void megamol::pbs::deleteTexture(GLuint &texture) {
    if (glIsTexture(texture)) {
        glDeleteTextures(1, &texture);
        texture = 0;
    }
}


GLuint megamol::pbs::createFBOFromRBO(GLuint &color_rbo, GLuint &depth_rbo) {
    GLuint fbo = 0;

    // save old framebuffer state
    GLint old_draw_fbo = 0, old_read_fbo = 0;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &old_read_fbo);
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &old_draw_fbo);

    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color_rbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rbo);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glReadBuffer(GL_COLOR_ATTACHMENT0);

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        glDeleteFramebuffers(1, &fbo);
        throw std::runtime_error("Could not create FBO\n");
    }

    // restore old framebuffer state
    glBindFramebuffer(GL_READ_FRAMEBUFFER, old_read_fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, old_draw_fbo);

    return fbo;
}
