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
#include "mmcore/CallerSlot.h"
#include "mmcore/view/CallRender3D.h"

#include "vislib/sys/Log.h"

using namespace megamol;
using namespace megamol::pbs;


FBOTransmitter::FBOTransmitter(void)
    : core::Module(),
      core::job::AbstractJob(),
      viewNameSlot("view", "The name of the view instance to be used"),
      /*fboWidthSlot("width", "Sets width of FBO"),
      fboHeightSlot("height", "Sets height of FBO"),*/
      ipAddressSlot("address", "IP address of reciever"),
      animTimeParamNameSlot("timeparamname", "Name of the time parameter"),
      triggerButtonSlot("trigger", "The trigger button"),
      identifierSlot("id", "the mpi rank"),
      frameSkipSlot("frame_skip", "Sets the number of frame to skip before comm"),
      zmq_ctx(1),
      zmq_socket(zmq_ctx, zmq::socket_type::push),
      ip_address("*:34242") {
    this->viewNameSlot << new core::param::StringParam("");
    this->MakeSlotAvailable(&this->viewNameSlot);

    /*this->fboWidthSlot << new core::param::IntParam(1, 1, 3840);
    this->fboWidthSlot.SetUpdateCallback(&FBOTransmitter::resizeCallback);
    this->MakeSlotAvailable(&this->fboWidthSlot);

    this->fboHeightSlot << new core::param::IntParam(1, 1, 2160);
    this->fboHeightSlot.SetUpdateCallback(&FBOTransmitter::resizeCallback);
    this->MakeSlotAvailable(&this->fboHeightSlot);*/

    this->ipAddressSlot << new core::param::StringParam("*:34242");
    this->ipAddressSlot.SetUpdateCallback(&FBOTransmitter::connectSocketCallback);
    this->MakeSlotAvailable(&this->ipAddressSlot);

    this->animTimeParamNameSlot << new core::param::StringParam("inst::time");
    this->MakeSlotAvailable(&this->animTimeParamNameSlot);

    this->triggerButtonSlot << new core::param::ButtonParam(vislib::sys::KeyCode::KEY_MOD_ALT | 't');
    this->triggerButtonSlot.SetUpdateCallback(&FBOTransmitter::triggerButtonClicked);
    this->MakeSlotAvailable(&this->triggerButtonSlot);

    this->identifierSlot << new core::param::IntParam(0);
    this->MakeSlotAvailable(&this->identifierSlot);

    this->frameSkipSlot << new core::param::IntParam(3, 0);
    this->MakeSlotAvailable(&this->frameSkipSlot);
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
    /*this->width = this->fboWidthSlot.Param<core::param::IntParam>()->Value();
    this->height = this->fboHeightSlot.Param<core::param::IntParam>()->Value();*/

    /*this->color_rbo = createRBO(GL_RGBA8, this->width, this->height);
    this->depth_rbo = createRBO(GL_DEPTH_COMPONENT24, width, height);
    this->fbo = createFBOFromRBO(this->color_rbo, this->depth_rbo);*/

    /*this->color_buf = std::vector<unsigned char>(this->width * this->height * 4);
    this->depth_buf = std::vector<unsigned char>(this->width * this->height * 3);*/

    return true;
}


void FBOTransmitter::release(void) {
    this->zmq_socket.unbind("tcp://" + this->ip_address);
}


void FBOTransmitter::BeforeRender(core::view::AbstractView *view) {
    glGetIntegerv(GL_VIEWPORT, this->viewport);
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


bool FBOTransmitter::extractBoundingBox(float bbox[6]) {
    bool success = true;
    std::string mvn(this->viewNameSlot.Param<core::param::StringParam>()->Value());
    this->ModuleGraphLock().LockExclusive();
    AbstractNamedObjectContainer::ptr_type anoc = AbstractNamedObjectContainer::dynamic_pointer_cast(this->RootModule());
    AbstractNamedObject::ptr_type ano = anoc->FindNamedObject(mvn.c_str());
    core::view::AbstractView *vi = dynamic_cast<core::view::AbstractView *>(ano.get());
    if (vi != NULL) {
        for (auto c = vi->ChildList_Begin(); c != vi->ChildList_End(); c++) {
            core::CallerSlot *sl = dynamic_cast<core::CallerSlot*>((*c).get());
            if (sl != nullptr) {
                core::view::CallRender3D *r = sl->CallAs<core::view::CallRender3D>();
                if (r != nullptr) {
                    bbox[0] = r->AccessBoundingBoxes().ObjectSpaceBBox().GetLeft();
                    bbox[1] = r->AccessBoundingBoxes().ObjectSpaceBBox().GetBottom();
                    bbox[2] = r->AccessBoundingBoxes().ObjectSpaceBBox().GetBack();
                    bbox[3] = r->AccessBoundingBoxes().ObjectSpaceBBox().GetRight();
                    bbox[4] = r->AccessBoundingBoxes().ObjectSpaceBBox().GetTop();
                    bbox[5] = r->AccessBoundingBoxes().ObjectSpaceBBox().GetFront();
                    break;
                }
            }
        }
    } else {
        success = false;
    }
    this->ModuleGraphLock().UnlockExclusive();
    return success;
}


void FBOTransmitter::AfterRender(core::view::AbstractView *view) {
    //view->UnregisterHook(this);
    /*static std::chrono::high_resolution_clock::time_point last;
    auto now = std::chrono::high_resolution_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last) >= 
        std::chrono::milliseconds(50)) {
        last = now;
    } else {
        return;
    }*/

    uint32_t skip = this->frameSkipSlot.Param<core::param::IntParam>()->Value();
    this->frame_id++;
    if (this->frame_id % skip != 0) {
        return;
    }

    this->width = this->viewport[2] - this->viewport[0];
    this->height = this->viewport[3] - this->viewport[1];

    this->color_buf.resize(this->width*this->height * 4);
    this->depth_buf.resize(this->width*this->height * 4);

    glReadPixels(0, 0, this->width, this->height, GL_RGBA, GL_UNSIGNED_BYTE, this->color_buf.data());

    glReadPixels(0, 0, this->width, this->height, GL_DEPTH_COMPONENT, GL_FLOAT, this->depth_buf.data());
    
    float bbox[6];
    if (!this->extractBoundingBox(bbox)) {
        vislib::sys::Log::DefaultLog.WriteError("FBOTransmitter: could not extract bounding box");
    }

    // send buffers over socket
    zmq::message_t dump;
    try {
        if (this->zmq_socket.connected()) {
            // do stuff
           /* while (!this->zmq_socket.recv(&dump, ZMQ_DONTWAIT)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                vislib::sys::Log::DefaultLog.WriteInfo("FBOTransmitter: Waiting for request\n");
            }*/
            vislib::sys::Log::DefaultLog.WriteInfo("FBOTransmitter: Waiting for request\n");
            //this->zmq_socket.recv(&dump);*/
            zmq::message_t msg(sizeof(int32_t) + sizeof(uint32_t) + sizeof(int) * 4 
                + sizeof(float) * 6 + this->color_buf.size() + this->depth_buf.size());
            //int viewport[] = {0, 0, this->width, this->height};
            char *ptr = reinterpret_cast<char*>(msg.data());
            int32_t *rank = reinterpret_cast<int32_t*>(ptr);
            *rank = this->identifierSlot.Param<core::param::IntParam>()->Value();
            ptr += sizeof(int32_t);
            uint32_t *fid = reinterpret_cast<uint32_t*>(ptr);
            *fid = this->frame_id;
            ptr += sizeof(uint32_t);
            memcpy(ptr, this->viewport, sizeof(this->viewport));
            ptr += sizeof(this->viewport);
            memcpy(ptr, bbox, sizeof(bbox));
            ptr += sizeof(bbox);
            memcpy(ptr, this->color_buf.data(), this->color_buf.size());
            ptr += color_buf.size();
            memcpy(ptr, this->depth_buf.data(), this->depth_buf.size());
            this->zmq_socket.send(msg);
            //this->frame_id++;
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
