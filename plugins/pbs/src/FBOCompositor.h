/*
 * FBOCompositor.h
 *
 * Copyright (C) 2017 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_PBS_FBOCOMPOSITOR_H_INCLUDED
#define MEGAMOL_PBS_FBOCOMPOSITOR_H_INCLUDED

#include <atomic>
#include <thread>
#include <vector>
#include <mutex>

#include "mmcore/Call.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/Renderer3DModule.h"

#include "zmq.hpp"

#include "glad/glad.h"

namespace megamol {
namespace pbs {

class FBOCompositor : public core::view::Renderer3DModule {
public:
    /**
    * Answer the name of this module.
    *
    * @return The name of this module.
    */
    static const char *ClassName(void) {
        return "FBOCompositor";
    }

    /**
    * Answer a human readable description of this module.
    *
    * @return A human readable description of this module.
    */
    static const char *Description(void) {
        return "Composits images from socket into a rendering.";
    }

    /**
    * Answers whether this module is available on the current system.
    *
    * @return 'true' if the module is available, 'false' otherwise.
    */
    static bool IsAvailable(void) {
        return gladLoadGL();
    }

    FBOCompositor(void);

    ~FBOCompositor(void);
protected:
    virtual bool create(void) override;

    virtual void release(void) override;

    virtual bool GetCapabilities(core::Call &call) override;

    virtual bool GetExtents(core::Call &call) override;

    virtual bool Render(core::Call &call) override;
private:
    typedef struct _fbo_data {
        int viewport[4];
        std::vector<unsigned char> color_buf;
        std::vector<unsigned char> depth_buf;
    } fbo_data;

    bool printShaderInfoLog(GLuint shader) const;

    bool printProgramInfoLog(GLuint shaderProg) const;

    bool connectSocketCallback(core::param::ParamSlot &p);

    void connectSocket(std::string &address);

    void receiverCallback(void);

    bool updateNumRenderNodesCallback(core::param::ParamSlot &p);

    void swapFBOData(void);

    core::param::ParamSlot ipAddressSlot;

    core::param::ParamSlot numRenderNodesSlot;

    core::param::ParamSlot fboWidthSlot;

    core::param::ParamSlot fboHeightSlot;

    zmq::context_t zmq_ctx;

    zmq::socket_t zmq_socket;

    std::string ip_address;

    std::thread receiverThread;

    std::vector<fbo_data> receiverData;

    std::vector<fbo_data> renderData;

    std::mutex swap_guard;

    std::atomic<bool> is_new_data;

    int num_render_nodes = 0;

    int fbo_width;

    int fbo_height;

    GLuint *color_textures;

    GLuint *depth_textures;

    GLuint shader;

    GLuint vao, vbo;

    bool is_connected = false;
}; /* end class FBOCompositor */

} /* end namespace pbs */
} /* end namespace megamol */

#endif /* end ifndef MEGAMOL_PBS_FBOCOMPOSITOR_H_INCLUDED */