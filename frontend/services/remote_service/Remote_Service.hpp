/*
 * Remote_Service.hpp
 *
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "AbstractFrontendService.hpp"

#include "RuntimeConfig.h"

namespace megamol {
namespace frontend {

class Remote_Service final : public AbstractFrontendService {
public:
    struct HeadNode;
    struct RenderNode;
    struct MpiNode;

    enum class Role {
        None,
        HeadNode,
        RenderNode,
        MPIRenderNode
    };
    struct Config {
        Role role = Role::None;

        bool head_broadcast_quit            = false; // if MegaMol exists in a normal way, broadcast mmQuit()
        bool head_broadcast_initial_project = false; // send initial project/graph state at startup
        bool head_connect_on_start          = false; // start headnode thread on startup or wait till later

        std::string zmq_control_send_to      = "tcp://127.0.0.1:62562"; // "Address of headnode in ZMQ syntax (e.g. \"tcp://127.0.0.1:33333\")"
        std::string zmq_control_receive_from = "tcp://*:62562"; // "Address of headnode in ZMQ syntax (e.g. \"tcp://127.0.0.1:33333\")"

        int  mpi_broadcast_rank = 0;     // "Set which MPI rank is the broadcast master"

        // UNUSED: bool render_use_mpi               = false; // "Requests initialization of MPI and the communicator for the view."
        // UNUSED: bool render_sync_data_sources_mpi = false; // "Requests synchronization of data sources in the MPI world."
        // TODO: decouple ZMQ send/recv vs MPI send/recv
    };

    std::string serviceName() const override { return "Remote_Service"; }

    Remote_Service();
    ~Remote_Service();

    bool init(const Config& config);
    bool init(void* configPtr) override;
    void close() override;

    std::vector<FrontendResource>& getProvidedResources() override;

    const std::vector<std::string> getRequestedResourceNames() const override;
    void setRequestedResources(std::vector<FrontendResource> resources) override;

    void updateProvidedResources() override;
    void digestChangedRequestedResources() override;
    void resetProvidedResources() override;
    void preGraphRender() override;  
    void postGraphRender() override;

    // from AbstractFrontendService 
    // int setPriority(const int p) // priority initially 0
    // int getPriority() const;
    // 
    // bool shouldShutdown() const; // shutdown initially false
    // void setShutdown(const bool s = true);

private:
    std::vector<FrontendResource> m_providedResourceReferences;
    std::vector<std::string> m_requestedResourcesNames;
    std::vector<FrontendResource> m_requestedResourceReferences;


    // it seems the camera gives us some sync number
    // we update and send camera state only if we have an older sync number than the camera
    // bool megamol::remote::HeadnodeServer::get_cam_upd(std::vector<char>& msg) {
    unsigned int camera_syncnumber_ = -1;

    std::function<void()> m_do_remote_things;
    void do_headnode_things();
    void do_rendernode_things();
    void do_mpi_things();

    void head_send_message(std::string const& string);
    void execute_message(std::vector<char> const& message);

    struct PimplData;
    std::unique_ptr<PimplData, std::function<void(PimplData*)>> m_pimpl;
    Config m_config;
};

std::string handle_remote_session_config(megamol::frontend_resources::RuntimeConfig const& config, Remote_Service::Config& remote_config);

} // namespace frontend
} // namespace megamol

