/*
 * FBOTransmitter.cpp
 *
 * Copyright (C) 2017 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "FBOTransmitter.h"

#include <thread>
#include <chrono>

#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/view/CallRenderView.h"

#include "vislib/sys/Log.h"

using namespace megamol;
using namespace megamol::pbs;


FBOTransmitter::FBOTransmitter(void)
    : core::Module(),
      core::job::AbstractJob(),
      viewNameSlot("view", "The name of the view instance to be used"),
      fboWidthSlot("width", "Sets width of FBO"),
      fboHeightSlot("height", "Sets height of FBO"),
      ipAddressSlot("address", "IP address of reciever"),
      animTimeParamNameSlot("timeparamname", "Name of the time parameter"),
      triggerButtonSlot("trigger", "The trigger button"),
      zmq_ctx(1),
      zmq_socket(zmq_ctx, zmq::socket_type::rep),
      ip_address("*:34242") {
    this->viewNameSlot << new core::param::StringParam("");
    this->MakeSlotAvailable(&this->viewNameSlot);

    this->fboWidthSlot << new core::param::IntParam(1, 1, 3840);
    this->fboWidthSlot.SetUpdateCallback(&FBOTransmitter::resizeCallback);
    this->MakeSlotAvailable(&this->fboWidthSlot);

    this->fboHeightSlot << new core::param::IntParam(1, 1, 2160);
    this->fboHeightSlot.SetUpdateCallback(&FBOTransmitter::resizeCallback);
    this->MakeSlotAvailable(&this->fboHeightSlot);

    this->ipAddressSlot << new core::param::StringParam("*:34242");
    this->ipAddressSlot.SetUpdateCallback(&FBOTransmitter::connectSocketCallback);
    this->MakeSlotAvailable(&this->ipAddressSlot);

    this->animTimeParamNameSlot << new core::param::StringParam("inst::time");
    this->MakeSlotAvailable(&this->animTimeParamNameSlot);

    this->triggerButtonSlot << new core::param::ButtonParam(vislib::sys::KeyCode::KEY_MOD_ALT | 't');
    this->triggerButtonSlot.SetUpdateCallback(&FBOTransmitter::triggerButtonClicked);
    this->MakeSlotAvailable(&this->triggerButtonSlot);
}


FBOTransmitter::~FBOTransmitter(void) {
    this->Release();
}


bool megamol::pbs::FBOTransmitter::IsRunning(void) const {
    return this->is_running;
}


bool megamol::pbs::FBOTransmitter::Start(void) {
    this->is_running = true;
    return true;
}


bool megamol::pbs::FBOTransmitter::Terminate(void) {
    this->is_running = false;
    return true;
}


bool FBOTransmitter::create(void) {
    //connectSocketCallback(this->ipAddressSlot);

    // create FBO
    this->width = this->fboWidthSlot.Param<core::param::IntParam>()->Value();
    this->height = this->fboHeightSlot.Param<core::param::IntParam>()->Value();

    /*this->color_rbo = createRBO(GL_RGBA8, this->width, this->height);
    this->depth_rbo = createRBO(GL_DEPTH_COMPONENT24, width, height);
    this->fbo = createFBOFromRBO(this->color_rbo, this->depth_rbo);*/

    this->color_buf = std::vector<unsigned char>(this->width * this->height * 4);
    this->depth_buf = std::vector<unsigned char>(this->width * this->height * 3);

    return true;
}


void FBOTransmitter::release(void) {
    this->zmq_socket.unbind("tcp://" + this->ip_address);
}


void FBOTransmitter::BeforeRender(core::view::AbstractView *view) {

    glGetIntegerv(GL_VIEWPORT, this->viewport);


    //view->Resize(this->width, this->height);

    //view->UnregisterHook(this);

    //core::view::CallRenderView crv;

    //float frameTime = -1.0f;
    //core::param::ParamSlot *time = this->findTimeParam(view);
    //if (time != NULL) {
    //    frameTime = time->Param<core::param::FloatParam>()->Value();
    //}

    //// save old framebuffer state
    //GLint old_draw_fbo = 0, old_read_fbo = 0;
    //glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &old_read_fbo);
    //glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &old_draw_fbo);

    //// bind FBO
    //glBindFramebuffer(GL_FRAMEBUFFER, this->fbo);
    //glViewport(0, 0, this->width, this->height);
    //crv.SetOutputBuffer(this->fbo, this->width, this->height);
    //crv.SetTile(static_cast<float>(this->width), static_cast<float>(this->height),
    //    0.0f, 0.0f, static_cast<float>(this->width), static_cast<float>(this->height));
    //crv.SetTime(frameTime);

    //view->OnRenderView(crv);
    //glFlush();
    //
    //glReadPixels(0, 0, this->width, this->height, GL_RGBA, GL_UNSIGNED_BYTE, this->color_buf.data());
    //
    //glReadPixels(0, 0, this->width, this->height, GL_DEPTH_COMPONENT24, GL_UNSIGNED_BYTE, this->depth_buf.data());

    //// restore old framebuffer state
    //glBindFramebuffer(GL_READ_FRAMEBUFFER, old_read_fbo);
    //glBindFramebuffer(GL_DRAW_FRAMEBUFFER, old_draw_fbo);

    //// send buffers over socket
    //if (this->zmq_socket.connected()) {
    //    // do stuff
    //    if (!this->zmq_socket.send(zmq::message_t(&MSG_STARTFRAME, sizeof(int)))) {
    //        return;
    //    }
    //    if (!this->zmq_socket.send(zmq::message_t(&MSG_SENDVIEWPORT, sizeof(int)))) {
    //        return;
    //    }
    //    int viewport[] = {0, 0, this->width, this->height};
    //    if (!this->zmq_socket.send(zmq::message_t(viewport, sizeof(int) * 4))) {
    //        return;
    //    }
    //    if (!this->zmq_socket.send(zmq::message_t(&MSG_SENDDATA, sizeof(int)))) {
    //        return;
    //    }
    //    if (!this->zmq_socket.send(this->color_buf.begin(), this->color_buf.end())) {
    //        return;
    //    }
    //    if (!this->zmq_socket.send(this->depth_buf.begin(), this->depth_buf.end())) {
    //        return;
    //    }
    //    if (!this->zmq_socket.send(zmq::message_t(&MSG_ENDFRAME, sizeof(int)))) {
    //        return;
    //    }
    //} else {
    //    // connect socket
    //    this->ip_address = this->ipAddressSlot.Param<core::param::StringParam>()->Value();
    //    this->connectSocket(this->ip_address);
    //}
}


bool FBOTransmitter::resizeCallback(core::param::ParamSlot &p) {
    /*this->width = this->fboWidthSlot.Param<core::param::IntParam>()->Value();
    this->height = this->fboHeightSlot.Param<core::param::IntParam>()->Value();

    this->color_buf.resize(this->width*this->height*4);
    this->depth_buf.resize(this->width*this->height*4);*/

    /*deleteRBO(this->color_rbo);
    deleteRBO(this->depth_rbo);
    deleteFBO(this->fbo);

    this->color_rbo = createRBO(GL_RGBA8, width, height);
    this->depth_rbo = createRBO(GL_DEPTH_COMPONENT24, width, height);
    this->fbo = createFBOFromRBO(color_rbo, depth_rbo);*/

    return true;
}


void FBOTransmitter::AfterRender(core::view::AbstractView *view) {
    //view->UnregisterHook(this);

    this->width = this->viewport[2] - this->viewport[0];
    this->height = this->viewport[3] - this->viewport[1];

    this->color_buf.resize(this->width*this->height * 4);
    this->depth_buf.resize(this->width*this->height * 4);

    glReadPixels(0, 0, this->width, this->height, GL_RGBA, GL_UNSIGNED_BYTE, this->color_buf.data());

    glReadPixels(0, 0, this->width, this->height, GL_DEPTH_COMPONENT24, GL_UNSIGNED_INT_24_8, this->depth_buf.data());


    // send buffers over socket
    zmq::message_t dump;
    try {
        if (this->zmq_socket.connected()) {
            // do stuff
            while (!this->zmq_socket.recv(&dump, ZMQ_DONTWAIT)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                vislib::sys::Log::DefaultLog.WriteInfo("FBOTransmitter: Waiting for request\n");
            }
            /*vislib::sys::Log::DefaultLog.WriteInfo("FBOTransmitter: Waiting for request\n");
            this->zmq_socket.recv(&dump);*/
            zmq::message_t msg(sizeof(int) * 4 + this->color_buf.size() + this->depth_buf.size());
            //int viewport[] = {0, 0, this->width, this->height};
            char *ptr = reinterpret_cast<char*>(msg.data());
            memcpy(ptr, this->viewport, sizeof(this->viewport));
            ptr += sizeof(this->viewport);
            memcpy(ptr, this->color_buf.data(), this->color_buf.size());
            ptr += color_buf.size();
            memcpy(ptr, this->depth_buf.data(), this->depth_buf.size());
            this->zmq_socket.send(msg);

            //std::this_thread::sleep_for(std::chrono::milliseconds(10));

            
            //while (!this->zmq_socket.recv(&dump, ZMQ_DONTWAIT)) {
            //    std::this_thread::sleep_for(std::chrono::milliseconds(10));
            //    vislib::sys::Log::DefaultLog.WriteInfo("FBOTransmitter: Waiting for viewport request\n");
            //}
            ///*if (!this->zmq_socket.send(zmq::message_t(&MSG_STARTFRAME, sizeof(int)))) {
            //    return;
            //}*/
            ///*if (!this->zmq_socket.send(zmq::message_t(&MSG_SENDVIEWPORT, sizeof(int)))) {
            //    return;
            //}*/
            //vislib::sys::Log::DefaultLog.WriteInfo("FBOTransmitter: Replying viewport\n");
            //int viewport[] = {0, 0, this->width, this->height};
            //if (!this->zmq_socket.send(zmq::message_t(viewport, sizeof(int) * 4))) {
            //    return;
            //}
            //this->zmq_socket.recv(&dump);
            //vislib::sys::Log::DefaultLog.WriteInfo("FBOTransmitter: Replying color\n");
            //if (!this->zmq_socket.send(this->color_buf.begin(), this->color_buf.end())) {
            //    return;
            //}
            //this->zmq_socket.recv(&dump);
            //vislib::sys::Log::DefaultLog.WriteInfo("FBOTransmitter: Replying depth\n");
            //if (!this->zmq_socket.send(this->depth_buf.begin(), this->depth_buf.end())) {
            //    return;
            //}
        } else {
            // connect socket
            this->connectSocketCallback(this->ipAddressSlot);
        }
    } catch (zmq::error_t e) {
        vislib::sys::Log::DefaultLog.WriteError("FBO Transmitter: ZMQ error %s", e.what());
    }
}


bool FBOTransmitter::connectSocketCallback(core::param::ParamSlot &p) {
    if (this->is_connected) {
        this->zmq_socket.unbind("tcp://"+this->ip_address);
        this->is_connected = false;
    }

    this->ip_address = this->ipAddressSlot.Param<core::param::StringParam>()->Value();
    this->connectSocket(this->ip_address);

    return true;
}


void FBOTransmitter::connectSocket(std::string &address) {
    try {
        this->zmq_socket.bind("tcp://" + address);
        this->is_connected = true;
    } catch (zmq::error_t e) {
        vislib::sys::Log::DefaultLog.WriteError("FBO Transmitter: ZMQ error %s", e.what());
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


bool FBOTransmitter::triggerButtonClicked(core::param::ParamSlot &slot) {
    // happy trigger finger hit button action happend
    using vislib::sys::Log;

    std::string mvn(this->viewNameSlot.Param<core::param::StringParam>()->Value());
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 100,
        "ScreenShot of \"%s\" requested", mvn.c_str());

    this->ModuleGraphLock().LockExclusive();
    AbstractNamedObjectContainer::ptr_type anoc = AbstractNamedObjectContainer::dynamic_pointer_cast(this->RootModule());
    AbstractNamedObject::ptr_type ano = anoc->FindNamedObject(mvn.c_str());
    core::view::AbstractView *vi = dynamic_cast<core::view::AbstractView *>(ano.get());
    if (vi != NULL) {
        vi->RegisterHook(this);
    } else {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to find view \"%s\" for ScreenShot",
            mvn.c_str());
    }
    this->ModuleGraphLock().UnlockExclusive();

    return true;
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
