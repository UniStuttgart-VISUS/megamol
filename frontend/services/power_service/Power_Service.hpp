/*
 * Power_Service.hpp
 *
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#ifdef MEGAMOL_USE_POWER

#include <chrono>
#include <unordered_map>
#include <variant>

#include "AbstractFrontendService.hpp"

#include "ParallelPortTrigger.h"

#include "PowerCallbacks.h"

#include <power_overwhelming/emi_sensor.h>
#include <power_overwhelming/msr_sensor.h>
#include <power_overwhelming/nvml_sensor.h>
#include <power_overwhelming/rtx_instrument.h>
#include <power_overwhelming/rtx_instrument_configuration.h>
#include <power_overwhelming/tinkerforge_sensor.h>

#include <sol/sol.hpp>

#include "RTXInstruments.h"
#include "SampleBuffer.h"

namespace megamol {
namespace frontend {

// search/replace Power_Service with your class name
// you should also delete the FAQ comments in these template files after you read and understood them
class Power_Service final : public AbstractFrontendService {
public:
    // We encourage you to use a configuration struct
    // that can be passed to your init() function.
    struct Config {
        std::string lpt = "lpt1";
        bool write_to_files = false;
        std::string folder = "./";
    };

    // sometimes somebody wants to know the name of the service
    std::string serviceName() const override {
        return "Power_Service";
    }

    // constructor should not take arguments, actual object initialization deferred until init()
    Power_Service();
    ~Power_Service();
    // your service will be constructed and destructed, but not copy-constructed or move-constructed
    // so no need to worry about copy or move constructors.

    // implement the following functions/callbacks to get your service hooked into the frontend

    // init service with input config data, e.g. init GLFW with OpenGL and open window with certain decorations/hints
    // if init() fails return false (this will terminate program execution), on success return true
    bool init(const Config& config);
    bool init(void* configPtr) override;
    void close() override;

    // expose the resources or input events this service provides via getProvidedResources(): e.g. Keyboard inputs, Controller inputs, GLFW Window events
    // the FrontendResource is a named wrapper that wraps some type (struct) in an std::any and casts its content to a requested type
    // each service may provide a set of resources for other services or graph modules to use
    // usually resources shared among services and modules are read only, in the sense that the FrontendResource wrapper only returns const& to held resources
    // if you need to manipulate a resource that you do not own, make sure you know what you're doing before using const_cast<>
    // keep in mind that the FrontendResource is just a wrapper that holds a void* to an object that you provided
    // thus, keep objects that you broadcast as resources alive until your close() gets called!
    // if lifetime of one of your resources ends before your close() gets called you produce dangling references in other services!
    // if you need to re-initialize or swap resource contents in a way that needs an objects lifetime to end, consider wrapping that behaviour in a way that is user friendly
    std::vector<FrontendResource>& getProvidedResources() override;

    // a service may request a set of resources that are provided by other services or the system
    // this works in two steps: the service tells the system which services it requests using the service names (usually the type name of the structs)
    // the frontend system makes sure to provide the service with the requested resources, or else program execution is terminated with an error message
    // note that the list of resources given to the service is in order of the initial resource requests,
    // thus you don't need to search for resources by name but can rather access the set resource vector directly at the expected index
    // the idea of behind resources is that when your service gets its list of requested resources,
    // the contract is that there will be an actual resource available and not some null pointer.
    // if somebody does not play along and provides fake resources with wrong wrapped types to the system, thus giving your code garbage to work with,
    // we can not really stop him from doing so, but there should occur an unhandled exception thrown by std::any for a bad type cast
    // the gist is that you should expect to get the correct resources you requested from the system
    // and you can work on those resources without tedious error and type checking
    // if you want to see how resources are distributed among the services, look into FrontendServiceCollection.cpp::assignRequestedResources()
    // the lifetime of the resources you get starts when your setRequestedResources() gets called and ends before your close() gets called
    // dont depend on requested resources being available in your close(). you yourself sould destroy or close the resources you provide in close().
    // init() gets called in order of service priority (see below) and close() gets called in reverse order,
    // so if you really need access to some resource in your close() make sure
    // the priority order of your service is _after_ the service that provides your critical resources (i.e. your set priority number should be higher)
    const std::vector<std::string> getRequestedResourceNames() const override;
    void setRequestedResources(std::vector<FrontendResource> resources) override;

    // the following resource update and graph render callbacks get called in each iteration of the main loop
    // this is probably where most work of your service is done
    // the service callbacks get called in the main loop in the following order:
    //
    // auto services = {lua_service, opengl_service, gui_service}; // wrapper that loops over all services in service priority order
    // services.init();
    // services.assignResourcesAmongServices();
    //
    // while (true) {
    //     services.updateProvidedResources();
    //     services.digestChangedRequestedResources();
    //
    //     if (services.shouldShutdown())
    //         break;
    //
    //     {// render step
    //         services.preGraphRender();
    //         megamol_graph.RenderNextFrame();
    //         services.postGraphRender(); // calls service callbacks in reverse order
    //     }
    //
    //     services.resetProvidedResources();
    // }
    // services.close(); // calls service callbacks in reverse order

    // called first in main loop, each service updates its shared resources to some new state here (e.g. receive keyboard inputs, network traffic)
    void updateProvidedResources() override;

    // after each service updates its provided resources, services may check for updates in their requested resources
    // for example, a GUI may check for user inputs placed in keyboard or mouse input resources that are provided by some other service
    // usually working with resources should not modify them,
    // but if you are really sure that it is ok for you to change resources, you may cast the away the const from the resource reference and manipulate the resource
    // for example, you may cast away the const from the MegaMolGraph to issue creation/deletion of modules and calls
    // or you may delete keyboard and mouse inputs from corresponding resources if you are sure they only affect your service
    // this callback is also a good place to verify if your service received a shutdown request and propagate it to the system via setShutdown()
    void digestChangedRequestedResources() override;

    // after rendering of a frame finished and before the next iteration of the main loop, services may want to reset resource state to some value
    // e.g. after user inputs (keyboard, mouse) or window resize evets got handled by the relevant services or modules,
    // the service providing and managing those resource structs may want to clear those inputs before the next frame starts
    // (in some cases the distinction between updateProvidedResources() at the beginning of a main loop iteration
    // and a resetProvidedResources() at the end of that iteration seems to be necessary)
    void resetProvidedResources() override;

    // gets called before graph rendering, you may prepare rendering with some API, e.g. set frame-timers, etc
    void preGraphRender() override;
    // clean up after rendering, e.g. render gui over graph rendering, stop and show frame-timers in GLFW window, swap buffers, glClear for next framebuffer
    void postGraphRender() override;

    // from AbstractFrontendService
    // you inherit the following functions that manage priority of your service and shutdown requests to terminate the program
    // you and others may use those functions, but you will not override them
    // priority indicates the order in which services get their callbacks called, i.e. this is the sorting of the vector that holds all services
    // lower priority numbers get called before the bigger ones. for close() and postGraphRender() services get called in the reverse order,
    // i.e. this works like construction and destruction order of objects in a c++
    //
    // int setPriority(const int p) // priority initially 0
    // int getPriority() const;

    // your service can signal to the program that a shutdown request has been received.
    // call setShutdown() to set your shutdown status to true, this is best done in your updateProvidedResources() or digestChangedRequestedResources().
    // if a servie signals a shutdown the system calls close() on all services in reverse priority order, then program execution terminates.
    //
    // bool shouldShutdown() const; // shutdown initially false
    // void setShutdown(const bool s = true);

private:
    // this can hold references to the resources (i.e. structs) we provide to others, e.g. you may fill this and return it in getProvidedResources()
    // provided resources will be queried by the system only once,
    // there is no requirement to store the resources in a vector the whole time, you just need to return such a vector in getProvidedResources()
    // but you need to store the actual resource objects you provide and manage
    // note that FrontendResource wraps a void* to the objects you provide, thus your resource objects will not be copied, but they will be referenced
    // (however the FrontendResource objects themselves will be copied)
    std::vector<FrontendResource> m_providedResourceReferences;

    // names of resources you request for your service can go here
    // requested resource names will be queried by the system only once,
    // there is no requirement to store the names in a vector the whole time, you just need to return such a vector in getRequestedResourceNames()
    std::vector<std::string> m_requestedResourcesNames;

    // you may store the resources you requested in this vector by filling it when your setRequestedResources() gets called
    // the resources provided to you by the system match the names you requested in getRequestedResourceNames() and are expected to reference actual existing objects
    // the sorting of resources matches the order of your requested resources names, you can use this to directly index into the vector provided by setRequestedResources()
    // if every service follows the rules the provided resources should be valid existing objects, thus you can use them directly without error or nullptr checking,
    // but we in the end we must blindly rely on the std::any in FrontendResource to hold the struct or type you expect it to hold
    // (or else std::any will throw a bad type cast exception that should terminate program execution.
    // you do NOT catch or check for that exception or need to care for it in any way!)
    std::vector<FrontendResource> m_requestedResourceReferences;

    std::unique_ptr<ParallelPortTrigger> trigger_;

    frontend_resources::PowerCallbacks callbacks_;

    std::vector<visus::power_overwhelming::rtx_instrument> rtx_instr_;

    std::unordered_map<std::string, visus::power_overwhelming::nvml_sensor> nvml_sensors_;
    std::vector<SampleBuffer> nvml_buffers_;

    std::unordered_map<std::string, visus::power_overwhelming::emi_sensor> emi_sensors_;
    std::vector<SampleBuffer> emi_buffers_;

    std::unordered_map<std::string, visus::power_overwhelming::msr_sensor> msr_sensors_;
    std::vector<SampleBuffer> msr_buffers_;

    std::unordered_map<std::string, visus::power_overwhelming::tinkerforge_sensor> tinker_sensors_;
    std::vector<SampleBuffer> tinker_buffers_;

    //std::chrono::nanoseconds trigger_offset_;

    //std::vector<int64_t> sample_times_;

    //std::vector<std::string> sensor_names_;

    //double timer_mul_;

    void setup_measurement();

    void start_measurement();

    //bool is_measurement_pending() const {
    //    return pending_measurement_;
    //}

    void trigger();

    void fill_lua_callbacks();

    bool waiting_on_trigger() const;

    static bool init_sol_commands_;

    //std::shared_ptr<sol::state_view> sol_state_;

    sol::state sol_state_;

    std::unordered_map<std::string, visus::power_overwhelming::rtx_instrument_configuration> config_map_;

    bool pending_measurement_ = false;

    //bool pending_read_ = false;

    bool enforce_software_trigger_ = false;

    std::chrono::system_clock::time_point last_trigger_;

    int64_t tracy_last_trigger_;

    bool have_triggered_ = false;

    std::unordered_map<std::string, std::string> exp_map_;

    std::vector<float> examine_expression(std::string const& name, std::string const& exp_path, int s_idx);

    int64_t qpc_frequency_;

    int64_t get_tracy_time(int64_t base, int64_t tracy_offset, float seg_off) const;

    int64_t get_tracy_time(int64_t base, int64_t tracy_offset) const;

    std::vector<std::unordered_map<std::string, std::variant<std::vector<float>, std::vector<int64_t>>>> values_map_;

    bool write_to_files_ = false;

    std::string write_folder_ = "./";

    enum class file_type { RAW, CSV };

    void write_to_files(std::string const& folder_path, file_type ft) const;

    RTXInstruments rtx_;
};

} // namespace frontend
} // namespace megamol

#endif
