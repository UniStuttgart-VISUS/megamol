#pragma once

#include <atomic>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

#include "glad/glad.h"

//#include "mmcore/utility/gl/FramebufferObject.h"
#include "mmcore/view/Renderer3DModule.h"

#include "FBOCommFabric.h"
#include "FBOProto.h"
#include "mmcore/param/ParamSlot.h"


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
    bool GetCapabilities(core::Call& call) override;

    bool GetExtents(core::Call& call) override;

    bool Render(core::Call& call) override;

    void swapBuffers(void) {
        std::scoped_lock<std::mutex, std::mutex> guard{buffer_write_guard_, buffer_recv_guard_};
        swap(fbo_msg_recv_, fbo_msg_write_);
        /*swap(color_buf_recv_, color_buf_write_);
        swap(depth_buf_recv_, depth_buf_write_);*/
        data_has_changed_.store(true);
    }

    void receiverJob(FBOCommFabric& comm, std::promise<fbo_msg_t>&& fbo_msg_promise, std::future<bool>&& close) const;

    void collectorJob(std::vector<FBOCommFabric>&& comms, std::future<bool>&& close);

    void initTextures(size_t n, GLsizei width, GLsizei heigth);

    void resize(size_t n, GLsizei width, GLsizei height);

    std::vector<std::string> getAddresses(std::string const& str) const noexcept;

    std::vector<FBOCommFabric> connectComms(std::vector<std::string> const& addr) const;

    std::vector<FBOCommFabric> connectComms(std::vector<std::string>&& addr) const;

    megamol::core::param::ParamSlot addressesSlot_;

    // megamol::core::utility::gl::FramebufferObject fbo_;

    std::thread collector_thread_;

    std::promise<bool> close_promise_;

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

    std::vector<GLuint> color_textures_;

    std::vector<GLuint> depth_textures_;

    bool connected_;
}; // end class FBOCompositor2

} // end namespace pbs
} // end namespace megamol
