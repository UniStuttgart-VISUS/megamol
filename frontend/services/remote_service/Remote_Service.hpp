/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once
#undef None // on linux X.h defines None, crashing this header

#include "AbstractFrontendService.hpp"

#include "RuntimeConfig.h"

#include <memory> // unique_ptr

namespace megamol::frontend {

class Remote_Service final : public AbstractFrontendService {
public:
    struct HeadNode;
    struct RenderNode;
    struct MpiNode;

    enum class Role { None, HeadNode, RenderNode, MPIRenderNode };
    struct Config {
        Role role = Role::None;

        bool headnode_broadcast_quit = false;            // if MegaMol exits in a normal way, broadcast mmQuit()
        bool headnode_broadcast_initial_project = false; // send initial project/graph state at startup
        bool headnode_connect_on_start = false;          // start headnode thread on startup or wait till later

        std::string headnode_zmq_target_address =
            "tcp://127.0.0.1:62562"; // "Address of headnode in ZMQ syntax (e.g. \"tcp://127.0.0.1:33333\")"
        std::string rendernode_zmq_source_address =
            "tcp://*:62562"; // "Address of headnode in ZMQ syntax (e.g. \"tcp://127.0.0.1:33333\")"
        //bool        head_distribute_local_project_at_startup = true;                    // "Sends project file on connect"

        int mpi_broadcast_rank = 0; // "Set which MPI rank is the broadcast master"

        // UNUSED: bool render_use_mpi               = false; // "Requests initialization of MPI and the communicator for the view."
        // UNUSED: bool render_sync_data_sources_mpi = false; // "Requests synchronization of data sources in the MPI world."
        // TODO: decouple ZMQ send/recv vs MPI send/recv
    };

    std::string serviceName() const override {
        return "Remote_Service";
    }

    Remote_Service();
    ~Remote_Service() override;

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

    // EXPERIMENTAL
    struct HeadNodeRemoteControl {
        enum class Command {
            None = 0,
            StartHeadNode,
            CloseHeadNode,
            ClearGraph,
            SendGraph,
            KeepSendingParams,
            DontSendParams,
            SetParamSendingModules,
            SendLuaCommand,
            Count // not a commnd, gives number of enum entries
        };
        bool keep_sending_params = false;
        std::string modules_to_send_params_of = "all";
        std::string lua_command = "";

        std::vector<Command> commands_queue;
    };
    HeadNodeRemoteControl m_headnode_remote_control;
    void add_headnode_remote_command(HeadNodeRemoteControl::Command command, std::string const& value = "");
    void remote_control_window();

    bool start_headnode(bool start_or_shutdown = true);
};

std::string handle_remote_session_config(
    megamol::frontend_resources::RuntimeConfig const& config, Remote_Service::Config& remote_config);

} // namespace megamol::frontend
