/*
 * GraphManager.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_GRAPHMANAGER_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_GRAPHMANAGER_H_INCLUDED


#include "mmcore/CoreInstance.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/utility/plugins/AbstractPluginInstance.h"

#include "utility/plugins/PluginManager.h"

#include "vislib/UTF8Encoder.h"
#include "vislib/sys/Log.h"

#include <map>
#include <vector>

#include "Graph.h"


namespace megamol {
namespace gui {
namespace configurator {


class GraphManager {
public:
    typedef std::shared_ptr<Graph> GraphPtrType;
    typedef std::vector<GraphPtrType> GraphsType;

    GraphManager(void);

    virtual ~GraphManager(void);

    bool AddGraph(std::string name);
    bool DeleteGraph(ImGuiID graph_uid);
    const GraphManager::GraphsType& GetGraphs(void);
    const GraphPtrType GetGraph(ImGuiID graph_uid);

    bool UpdateModulesCallsStock(const megamol::core::CoreInstance* core_instance);
    inline const ModuleStockVectorType& GetModulesStock(void) { return this->modules_stock; }
    inline const CallStockVectorType& GetCallsStock(void) { return this->calls_stock; }

    bool LoadCurrentCoreProject(const std::string& name, megamol::core::CoreInstance* core_instance);

    bool LoadProjectFile(const std::string& project_filename, megamol::core::CoreInstance* core_instance);
    bool SaveProjectFile(
        ImGuiID graph_id, const std::string& project_filename, megamol::core::CoreInstance* core_instance);

    // GUI Presentation -------------------------------------------------------

    // Returns uid of the currently active/drawn graph.
    ImGuiID GUI_Present(float in_child_width, ImFont* in_graph_font, HotKeyArrayType& inout_hotkeys) {
        return this->present.Present(*this, in_child_width, in_graph_font, inout_hotkeys);
    }

private:
    // VARIABLES --------------------------------------------------------------

    GraphsType graphs;

    ModuleStockVectorType modules_stock;
    CallStockVectorType calls_stock;

    /**
     * Defines GUI graph present.
     */
    class Presentation {
    public:
        Presentation(void);

        ~Presentation(void);

        ImGuiID Present(GraphManager& inout_graph_manager, float in_child_width, ImFont* in_graph_font,
            HotKeyArrayType& inout_hotkeys);

    private:
        struct DropCallData {
            int trigger_drop_call;
            int dragged_call_data;
            ImGuiID hovered_call_slot_uid;
            ImGuiID selected_call_slot_uid;
        } drop_call_data;
        ImGuiID delete_graph_uid;
        GUIUtils utils;

    } present;

    // FUNCTIONS --------------------------------------------------------------

    bool get_call_stock_data(
        Call::StockCall& call, const std::shared_ptr<const megamol::core::factories::CallDescription> call_desc);

    bool get_module_stock_data(
        Module::StockModule& mod, const std::shared_ptr<const megamol::core::factories::ModuleDescription> mod_desc);

    // Can be used for mmCreateView, mmCreateModule and mmCreateCall lua commands
    bool readLuaProjectCommandArguments(const std::string& line, size_t arg_count, std::vector<std::string>& out_args);

    ImVec2 readLuaProjectConfPos(const std::string& line);

    std::string writeLuaProjectConfPos(const ImVec2& pos);

    bool separateNameAndPrefix(const std::string& full_name, std::string& name_space, std::string& name);

    // ------------------------------------------------------------------------
};

} // namespace configurator
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_GRAPHMANAGER_H_INCLUDED