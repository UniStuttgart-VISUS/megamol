/*
 * FBOTransmitter.h
 *
 * Copyright (C) 2017 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_PBS_FBOTRANSMITTER_H_INCLUDED
#define MEGAMOL_PBS_FBOTRANSMITTER_H_INCLUDED

#include "mmcore/Module.h"
#include "mmcore/ViewInstance.h"
#include "mmcore/job/AbstractJob.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/AbstractView.h"

#include "zmq.hpp"

#include "glad/glad.h"

namespace megamol {
namespace pbs {

class FBOTransmitter : public core::job::AbstractJob, public core::Module, public core::view::AbstractView::Hooks {
public:
    static const int MSG_STARTFRAME = 1;

    static const int MSG_SENDVIEWPORT = 2;

    static const int MSG_SENDDATA = 3;

    static const int MSG_ENDFRAME = 4;

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "FBOTransmitter";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) {
        return "A simple job module used to transmit FBOs over TCP/IP";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /**
     * Disallow usage in quickstarts
     *
     * @return false
     */
    static bool SupportQuickstart(void) {
        return false;
    }

    /**
     * Ctor
     */
    FBOTransmitter(void);

    /**
     * Dtor
     */
    virtual ~FBOTransmitter(void);

    /**
     * Answers whether or not this job is still running.
     *
     * @return 'true' if this job is still running, 'false' if it has
     *         finished.
     */
    virtual bool IsRunning(void) const;

    /**
     * Starts the job thread.
     *
     * @return true if the job has been successfully started.
     */
    virtual bool Start(void);

    /**
     * Terminates the job thread.
     *
     * @return true to acknowledge that the job will finish as soon
     *         as possible, false if termination is not possible.
     */
    virtual bool Terminate(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

    /**
     * Hook method to be called before the view is rendered.
     *
     * @param view The calling view
     */
    virtual void BeforeRender(core::view::AbstractView *view);

    /**
     * Hook method to be called after the view is rendered.
     *
     * @param view The calling view
     */
    virtual void AfterRender(core::view::AbstractView *view);

private:
    bool connectSocketCallback(core::param::ParamSlot &p);

    void connectSocket(std::string &address);

    core::param::ParamSlot *findTimeParam(core::view::AbstractView *view);

    bool triggerButtonClicked(core::param::ParamSlot &slot);

    bool resizeCallback(core::param::ParamSlot &p);

    zmq::context_t zmq_ctx;

    zmq::socket_t zmq_socket;

    std::string ip_address;

    core::param::ParamSlot viewNameSlot;

    core::param::ParamSlot fboWidthSlot;

    core::param::ParamSlot fboHeightSlot;

    core::param::ParamSlot ipAddressSlot;

    core::param::ParamSlot animTimeParamNameSlot;

    core::param::ParamSlot triggerButtonSlot;

    int width;

    int height;

    GLuint color_rbo, depth_rbo, fbo;

    std::vector<unsigned char> color_buf, depth_buf;

    bool is_running = false;

    bool is_connected = false;
}; /* end class FBOTransmitter */

GLuint createTexture(GLint internal_format, GLsizei width, GLsizei height, GLenum format, GLenum type);

void deleteTexture(GLuint &texture);

GLuint createFBOFromTex(GLuint &color_tex, GLuint &depth_tex);

GLuint createFBOFromRBO(GLuint &color_rbo, GLuint &depth_rbo);

void deleteFBO(GLuint &fbo);

GLuint createRBO(GLenum internal_format, GLsizei width, GLsizei height);

void deleteRBO(GLuint &rbo);

} /* end namespace pbs */
} /* end namespace megamol */

#endif // end ifndef MEGAMOL_PBS_FBOTRANSMITTER_H_INCLUDED
