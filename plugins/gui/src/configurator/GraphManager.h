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
    bool DeleteGraph(int graph_uid);
    const GraphManager::GraphsType& GetGraphs(void);
    const GraphPtrType GetGraph(int graph_uid);

    bool UpdateModulesCallsStock(const megamol::core::CoreInstance* core_instance);
    inline const ModuleStockVectorType& GetModulesStock(void) { return this->modules_stock; }
    inline const CallStockVectorType& GetCallsStock(void) { return this->calls_stock; }

    bool LoadCurrentCoreProject(const std::string& name, megamol::core::CoreInstance* core_instance);

    bool LoadProjectFile(int graph_id, const std::string& project_filename, megamol::core::CoreInstance* core_instance);
    bool SaveProjectFile(int graph_id, const std::string& project_filename, megamol::core::CoreInstance* core_instance);

    // GUI Presentation -------------------------------------------------------
    bool GUI_Present(
        float child_width, ImFont* graph_font, HotkeyData& paramter_search, HotkeyData& delete_graph_element) {
        return this->present.GUI_Present(*this, child_width, graph_font, paramter_search, delete_graph_element);
    }

    const GraphManager::GraphPtrType GUI_GetPresentedGraph(void) { return this->present.GetPresentedGraph(); }

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

        bool GUI_Present(GraphManager& graph_manager, float child_width, ImFont* graph_font,
            HotkeyData& paramter_search, HotkeyData& delete_graph_element);

        const GraphManager::GraphPtrType GetPresentedGraph(void) { return this->presented_graph; }

    private:
        GraphManager::GraphPtrType presented_graph;
        int delete_graph_uid;

        bool close_unsaved_popup(bool open_popup);

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

    bool readModuleAndSlotName(const std::string& full_slot_name, std::string& module_name, std::string& slot_name);

    // ------------------------------------------------------------------------
};

} // namespace configurator
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_GRAPHMANAGER_H_INCLUDED