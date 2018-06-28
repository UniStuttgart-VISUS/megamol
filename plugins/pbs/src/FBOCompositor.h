/*
 * FBOCompositor.h
 *
 * Copyright (C) 2017 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_PBS_FBOCOMPOSITOR_H_INCLUDED
#define MEGAMOL_PBS_FBOCOMPOSITOR_H_INCLUDED

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

#include "mmcore/Call.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/Renderer3DModule.h"

#include "zmq.hpp"

#include "glad/glad.h"

namespace megamol {
namespace pbs {

/*
 * Module receiving FBO contents from FBOTransmitters.
 * Uses depth-compositing.
 */
class FBOCompositor : public core::view::Renderer3DModule {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "FBOCompositor"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Composits images from socket into a rendering."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return gladLoadGL(); }

    FBOCompositor(void);

    ~FBOCompositor(void);

protected:
    virtual bool create(void) override;

    virtual void release(void) override;

    virtual bool GetCapabilities(core::Call& call) override;

    virtual bool GetExtents(core::Call& call) override;

    virtual bool Render(core::Call& call) override;

private:
    /*
     * Struct holding the parsed messafe from an FBOTransmitter.
     */
    typedef struct _fbo_data {
        uint32_t fid;
        int viewport[4];
        float bbox[6];
        std::vector<unsigned char> color_buf;
        std::vector<unsigned char> depth_buf;
    } fbo_data;

    bool printShaderInfoLog(GLuint shader) const;

    bool printProgramInfoLog(GLuint shaderProg) const;

    bool connectSocketCallback(core::param::ParamSlot& p);

    void connectSocket(std::string& address);

    /*
     * Callback that is listening for FBO messages
     */
    void receiverCallback(void);

    void resizeBuffers(void);

    void swapFBOData(void);

    core::param::ParamSlot ipAddressSlot;

    core::param::ParamSlot numRenderNodesSlot;

    core::param::ParamSlot fboWidthSlot;

    core::param::ParamSlot fboHeightSlot;

    zmq::context_t zmq_ctx;

    zmq::socket_t zmq_socket;

    std::vector<std::string> ip_address;

    std::thread receiverThread;

    /** storage for parsed messages from the receiver (pointer to single vector for simpler swapping) */
    std::vector<fbo_data>* receiverData;

    /** storage for parsed messages for rendering (pointer to single vector for simpler swapping) */
    std::vector<fbo_data>* renderData;

    /** lock to synchronize swapping with rendering */
    std::mutex swap_guard;

    /** atomic flag signalling that new data is available */
    std::atomic<bool> is_new_data;

    bool stopRequested = false;

    int num_render_nodes = 0;

    int fbo_width;

    int fbo_height;

    int max_viewport[4];

    GLuint* color_textures;

    GLuint* depth_textures;

    GLuint shader;

    GLuint vao, vbo;

    bool is_connected = false;

    // int viewport[4];
}; /* end class FBOCompositor */

} /* end namespace pbs */
} /* end namespace megamol */

#endif /* end ifndef MEGAMOL_PBS_FBOCOMPOSITOR_H_INCLUDED */
