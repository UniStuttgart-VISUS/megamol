/*
 * ImagePresentationEntryPoints.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "EntryPoint.h"
#include "FrontendResource.h"
#include "ImagePresentationSink.h"

#include <any>
#include <functional>
#include <vector>

namespace megamol {
namespace frontend_resources {

using EntryPointRenderFunctions = std::tuple<
    // ptr to entry point object
    void*,
    // rendering execution function
    EntryPointExecutionCallback,
    // get requested resources function
    std::function<std::vector<std::string>()>>;

// the ImagePresentation Service manages the entry points - things that are held and "get rendered" by the frontend
// using the ImagePresentationEntryPoints resource participants (services) may add/remove entry points to the image presentation
struct ImagePresentationEntryPoints {
    static std::string const GLFW_Sink_Name;
    std::function<bool(std::string const&, EntryPointRenderFunctions const&)> add_entry_point;
    std::function<bool(std::string const&, const int)> set_entry_point_priority;
    std::function<bool(std::string const&)> remove_entry_point;
    std::function<bool(std::string const&, std::string const&)> rename_entry_point;
    std::function<void()> clear_entry_points;

    std::function<bool(ImagePresentationSink const&)> add_sink;
    std::function<bool(std::string const&)> remove_sink;

    std::function<bool(std::string const&, std::string const&)> bind_sink_entry_point;
    std::function<bool(std::string const&, std::string const&)> unbind_sink_entry_point;

    // services may also subscribe to entry point changes to get notified when entry points get added/removed
    enum class SubscriptionEvent { Add, Remove, Rename, Clear, AddSink, RemoveSink, BindSink, UnbindSink };

    using SubscriberFunction =
        std::function<void(frontend_resources::ImagePresentationEntryPoints::SubscriptionEvent const&,
            std::vector<std::any> const& /*function input args*/)>;

    std::function<void(SubscriberFunction const&)> subscribe_to_entry_point_changes;
    std::function<optional<EntryPoint>(std::string const&)> get_entry_point;
};

} /* end namespace frontend_resources */
} /* end namespace megamol */
