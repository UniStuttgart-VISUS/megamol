#pragma once

#include <memory>
#include <atomic>
#include <future>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>

#include "glad/glad.h"

//#include "mmcore/utility/gl/FramebufferObject.h"
#include "mmcore/view/Renderer3DModule.h"

#include "FBOCommFabric.h"
#include "FBOProto.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/utility/sys/FutureReset.h"

#include "image_calls/Image2DCall.h"


namespace megamol {
namespace pbs {

class FBOCompositor2 : public megamol::core::view::Renderer3DModule {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "FBOCompositor2"; }

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

    FBOCompositor2(void);

    ~FBOCompositor2(void);

protected:
    bool create(void) override;

    void release(void) override;

private:
    bool GetExtents(core::Call& call) override;

    bool Render(core::Call& call) override;

    void swapBuffers(void) {
        std::scoped_lock<std::mutex, std::mutex> guard{buffer_write_guard_, buffer_recv_guard_};
        swap(fbo_msg_recv_, fbo_msg_write_);
        /*swap(color_buf_recv_, color_buf_write_);
        swap(depth_buf_recv_, depth_buf_write_);*/
        data_has_changed_.store(true);
    }

    void receiverJob(
        FBOCommFabric& comm, core::utility::sys::FutureReset<fbo_msg_t>* fbo_msg_future, std::future<bool>&& close);

    void collectorJob(std::vector<FBOCommFabric>&& comms);

    void registerJob(std::vector<std::string>& addresses);

    void initTextures(size_t n, GLsizei width, GLsizei heigth);

    void resize(size_t n, GLsizei width, GLsizei height);

    bool initThreads();

    bool shutdownThreads();

    bool getImageCallback(megamol::core::Call& c);

    bool startCallback(megamol::core::param::ParamSlot& p);

    static void RGBAtoRGB(std::vector<char> const& rgba, std::vector<unsigned char>& rgb);

    megamol::core::CalleeSlot provide_img_slot_;

    std::vector<std::string> getAddresses(std::string const& str) const noexcept;

    std::vector<FBOCommFabric> connectComms(std::vector<std::string> const& addr) const;

    std::vector<FBOCommFabric> connectComms(std::vector<std::string>&& addr) const;

    bool printShaderInfoLog(GLuint shader) const;

    bool printProgramInfoLog(GLuint shaderProg) const;

    // megamol::core::param::ParamSlot addressesSlot_;

    megamol::core::param::ParamSlot commSelectSlot_;

    megamol::core::param::ParamSlot targetBandwidthSlot_;

    megamol::core::param::ParamSlot numRendernodesSlot_;

    megamol::core::param::ParamSlot handshakePortSlot_;

    megamol::core::param::ParamSlot startSlot_;

    megamol::core::param::ParamSlot restartSlot_;

    megamol::core::param::ParamSlot renderOnlyRequestedFramesSlot_;

    // megamol::core::utility::gl::FramebufferObject fbo_;

    std::thread collector_thread_;

    std::promise<bool> close_promise_;

    std::future<bool> close_future_;

    std::condition_variable_any heartbeat_;

    std::condition_variable_any promise_exchange_;

    std::condition_variable_any promise_release_;

    std::atomic<bool> promise_atomic_;

    std::shared_mutex heartbeat_lock_;

    std::shared_mutex promise_exchange_lock_;

    std::shared_mutex promise_release_lock_;

    std::mutex buffer_write_guard_;

    std::mutex buffer_recv_guard_;

    // std::vector<std::thread> receiver_thread_pool_;

    std::unique_ptr<std::vector<fbo_msg>> fbo_msg_write_;

    std::unique_ptr<std::vector<fbo_msg>> fbo_msg_recv_;

    /*std::unique_ptr<fbo_msg_header_t> fbo_msg_write_;

    std::unique_ptr<fbo_msg_header_t> fbo_msg_recv_;

    std::unique_ptr<std::vector<char>> color_buf_write_;

    std::unique_ptr<std::vector<char>> depth_buf_write_;

    std::unique_ptr<std::vector<char>> color_buf_recv_;

    std::unique_ptr<std::vector<char>> depth_buf_recv_;*/

    std::atomic<bool> data_has_changed_;

    int col_buf_el_size_;

    int depth_buf_el_size_;

    GLsizei width_;

    GLsizei height_;

    float frame_times_[2];

    float camera_params_[9];

    std::vector<GLuint> color_textures_;

    std::vector<GLuint> depth_textures_;

    bool connected_;

    GLuint shader;

    GLuint vao, vbo;

    FBOCommFabric registerComm_;

    std::thread registerThread_;

    std::thread initThreadsThread_;

    std::atomic<bool> isRegistered_;

    std::vector<std::string> addresses_;

    std::vector<unsigned char> img_data_;

    // std::shared_ptr<unsigned char[]> img_data_ptr_;

    size_t hash_;

    bool shutdown_ = false;

    bool register_done_ = false;

}; // end class FBOCompositor2

} // end namespace pbs
} // end namespace megamol
