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

#include "FileUtils.h"
#include "Graph.h"


namespace megamol {
namespace gui {
namespace configurator {

typedef std::shared_ptr<Graph> GraphPtrType;
typedef std::vector<GraphPtrType> GraphsType;

class GraphManager {
public:
    GraphManager(void);

    virtual ~GraphManager(void);

    ImGuiID AddGraph(void);
    bool DeleteGraph(ImGuiID graph_uid);
    bool GetGraph(ImGuiID graph_uid, GraphPtrType& out_graph_ptr);

    bool UpdateModulesCallsStock(const megamol::core::CoreInstance* core_instance);
    inline const ModuleStockVectorType& GetModulesStock(void) { return this->modules_stock; }
    inline const CallStockVectorType& GetCallsStock(void) { return this->calls_stock; }

    bool LoadProjectCore(megamol::core::CoreInstance* core_instance);
    bool AddProjectCore(ImGuiID graph_uid, megamol::core::CoreInstance* core_instance);

    bool LoadAddProjectFile(ImGuiID graph_uid, const std::string& project_filename);

    bool SaveProjectFile(ImGuiID graph_uid, const std::string& project_filename);
    bool SaveGroupFile(ImGuiID group_uid, const std::string& project_filename);

    // GUI Presentation -------------------------------------------------------

    void GUI_Present(GraphStateType& state) { this->present.Present(*this, state); }

private:
    // VARIABLES --------------------------------------------------------------

    GraphsType graphs;

    ModuleStockVectorType modules_stock;
    CallStockVectorType calls_stock;

    unsigned int graph_name_uid;

    /** ************************************************************************
     * Defines GUI graph present.
     */
    class Presentation {
    public:
        Presentation(void);

        ~Presentation(void);

        void Present(GraphManager& inout_graph_manager, GraphStateType& state);

    private:
        ImGuiID graph_delete_uid;
        GUIUtils utils;

    } present;

    // FUNCTIONS --------------------------------------------------------------

    const GraphsType& get_graphs(void) { return this->graphs; }

    bool get_call_stock_data(
        Call::StockCall& call, const std::shared_ptr<const megamol::core::factories::CallDescription> call_desc);

    bool get_module_stock_data(
        Module::StockModule& mod, const std::shared_ptr<const megamol::core::factories::ModuleDescription> mod_desc);

    // Can be used for mmCreateView, mmCreateModule and mmCreateCall lua commands
    bool readLuaProjectCommandArguments(const std::string& line, size_t arg_count, std::vector<std::string>& out_args);

    ImVec2 readLuaProjectConfPos(const std::string& line);
    std::string writeLuaProjectConfPos(const ImVec2& pos);

    std::vector<std::string> readLuaProjectConfGroupInterface(const std::string& line);
    std::string writeLuaProjectConfGroupInterface(const ModulePtrType& module_ptr, const GraphPtrType& graph_ptr);

    bool separateNameAndNamespace(const std::string& full_name, std::string& name_space, std::string& name);

    inline const std::string generate_unique_graph_name(void) {
        return ("Project_" + std::to_string(++graph_name_uid));
    }
};

} // namespace configurator
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_GRAPHMANAGER_H_INCLUDED
