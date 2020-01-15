/*
 * Graph.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_H_INCLUDED

#include "vislib/sys/Log.h"

#include <imgui.h>
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_internal.h>
#include "imgui_impl_opengl3.h"
#include "imgui_stdlib.h"

#include <vector>
#include <map>

#include "mmcore/CoreInstance.h"
#include "mmcore/Module.h"
#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "utility/plugins/PluginManager.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/TernaryParam.h"
#include "mmcore/param/TransferFunctionParam.h"
#include "mmcore/param/Vector2fParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/Vector4fParam.h"


namespace megamol {
namespace gui {

class Graph {
public:

    // DATA STRUCTS -------------------
    struct ParamSlotData {
        std::string class_name;
        std::string description;
        std::string type;
    };

    struct CallSlotData {
        std::string class_name;
        std::string description;
        std::vector<size_t> compatible_call_idxs; // (Storing only indices of compatible calls for faster comparison.)
    };

    struct ModuleData {
        std::string class_name;
        std::string description;
        std::string plugin_name;
        std::vector<Graph::ParamSlotData> param_slots;
        std::vector<Graph::CallSlotData> callee_slots;
        std::vector<Graph::CallSlotData> caller_slots;
    };

    struct CallData {
        std::string class_name;
        std::string description;
        std::string plugin_name;
    };


    // GRAPH STRUCTS ------------------
    class Slot {
    public:
        Graph::CallSlotData basic;

        struct Gui {

        } gui;
    };

    class Call {
    public:
        Graph::CallData basic;
        std::shared_ptr<Graph::Slot> callee_slot; // = std::make_shared<Slot>(node.data.callee_slots[idx]);
        std::shared_ptr<Graph::Slot> caller_slot; // = std::make_shared<Slot>(node.data.caller_slots[idx]);

        struct Gui {

        } gui;
    };

    class Module {
    public:

        inline ImVec2 GetCalleeSlotPos(int slot_idx) const {
            return ImVec2(this->gui.position.x, this->gui.position.y + this->gui.size.y * ((float)slot_idx + 1) / ((float)this->basic.callee_slots.size() + 1));
        }

        inline ImVec2 GetCallerSlotPos(int slot_idx) const {
            return ImVec2(this->gui.position.x + this->gui.size.x, this->gui.position.y + this->gui.size.y * ((float)slot_idx + 1) / ((float)this->basic.caller_slots.size() + 1));
        }

        std::string name;
        std::string full_name;
        Graph::ModuleData basic;

        struct Gui {
            ImVec2 position;
            ImVec2 size;
        } gui;
    };

    // --------------------------------

    Graph(void);

    virtual ~Graph(void);


    bool AddModule(const std::string& module_class_name);

    bool AddCall(const std::string& call_class_name);

    bool UpdateAvailableModulesCallsOnce(const megamol::core::CoreInstance* core_instance);

    inline const std::vector<Graph::ModuleData>& GetAvailableModulesList(void) const { return this->modules_list; }

    inline const std::string GetCompatibleCallNamet(size_t idx) const {
        if (idx < this->calls_list.size()) {
            return this->calls_list[idx].class_name;
        }
        return std::string();
    }

    inline std::vector<Graph::Module>& GetGraphModules(void) { return this->mods; }

private:

    // VARIABLES --------------------------------------------------------------

    std::vector<Graph::Module> mods;
    std::vector<Graph::Call> calls;

    std::vector<Graph::CallData> calls_list;
    std::vector<Graph::ModuleData> modules_list;

    // FUNCTIONS --------------------------------------------------------------

    bool read_module_data(Graph::ModuleData& mod_data, const std::shared_ptr<const megamol::core::factories::ModuleDescription> mod_desc, const megamol::core::CoreInstance* core_instance);
    bool read_call_data(Graph::CallData& call_data, const std::shared_ptr<const megamol::core::factories::CallDescription> call_desc);


    // ------------------------------------------------------------------------
};

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_H_INCLUDED