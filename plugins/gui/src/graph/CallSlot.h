/*
 * CallSlot.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_CALLSLOT_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_CALLSLOT_H_INCLUDED


#include "GUIUtils.h"
#include "mmcore/AbstractCallSlotPresentation.h"
#include "widgets/HoverToolTip.h"


namespace megamol {
namespace gui {


    // Forward declarations
    class CallSlot;
    class Call;
    class Module;
    class Parameter;
    class InterfaceSlot;
    typedef std::shared_ptr<Parameter> ParamPtr_t;
    typedef std::shared_ptr<Call> CallPtr_t;
    typedef std::shared_ptr<Module> ModulePtr_t;
    typedef std::shared_ptr<InterfaceSlot> InterfaceSlotPtr_t;

// Types
#ifndef _CALL_SLOT_TYPE_
    enum CallSlotType { CALLEE, CALLER };
#define _CALL_SLOT_TYPE_
#endif
    typedef std::shared_ptr<CallSlot> CallSlotPtr_t;
    typedef std::vector<CallSlotPtr_t> CallSlotPtrVector_t;
    typedef std::map<CallSlotType, CallSlotPtrVector_t> CallSlotPtrMap_t;


    /** ************************************************************************
     * Defines call slot data structure for graph.
     */
    class CallSlot {
    public:
        struct StockCallSlot {
            std::string name;
            std::string description;
            std::vector<size_t> compatible_call_idxs;
            CallSlotType type;
            megamol::core::AbstractCallSlotPresentation::Necessity necessity;
        };

        CallSlot(ImGuiID uid, const std::string& name, const std::string& description,
            const std::vector<size_t>& compatible_call_idxs, CallSlotType type,
            megamol::core::AbstractCallSlotPresentation::Necessity necessity);
        ~CallSlot();

        bool CallsConnected(void) const;
        bool ConnectCall(const CallPtr_t& call_ptr);
        bool DisconnectCall(ImGuiID call_uid);
        bool DisconnectCalls(void);
        const std::vector<CallPtr_t>& GetConnectedCalls(void);

        bool IsParentModuleConnected(void) const;
        bool ConnectParentModule(ModulePtr_t pm);
        bool DisconnectParentModule(void);
        const ModulePtr_t& GetParentModule(void);

        static ImGuiID GetCompatibleCallIndex(const CallSlotPtr_t& callslot_1, const CallSlotPtr_t& callslot_2);
        static ImGuiID GetCompatibleCallIndex(
            const CallSlotPtr_t& callslot, const CallSlot::StockCallSlot& stock_callslot);

        bool IsConnectionValid(CallSlot& callslot);

        void Draw(megamol::gui::PresentPhase phase, GraphItemsState_t& state);
        void Update(const GraphItemsState_t& state);

        inline const ImGuiID UID(void) const {
            return this->uid;
        }
        inline const std::string Name(void) const {
            return this->name;
        }
        inline const CallSlotType Type(void) const {
            return this->type;
        }
        inline InterfaceSlotPtr_t InterfaceSlotPtr(void) const {
            return this->gui_interfaceslot_ptr;
        }
        inline ImVec2 Position(void) const {
            return this->gui_position;
        }
        inline const std::vector<size_t> CompatibleCallIdxs(void) const {
            return this->compatible_call_idxs;
        }

        void SetInterfaceSlotPtr(InterfaceSlotPtr_t interfaceslot_ptr) {
            this->gui_interfaceslot_ptr = interfaceslot_ptr;
        }

    private:
        // VARIABLES --------------------------------------------------------------

        const ImGuiID uid;
        const std::string name;
        const std::string description;
        // Storing only indices of compatible calls for faster comparison.
        const std::vector<size_t> compatible_call_idxs;
        const CallSlotType type;
        megamol::core::AbstractCallSlotPresentation::Necessity necessity;

        ModulePtr_t parent_module;
        std::vector<CallPtr_t> connected_calls;

        InterfaceSlotPtr_t gui_interfaceslot_ptr;

        bool gui_selected;
        ImVec2 gui_position; /// Absolute position including canvas offset and zooming
        bool gui_update_once;
        bool gui_show_modulestock;
        ImGuiID gui_last_compat_callslot_uid;
        ImGuiID gui_last_compat_interface_uid;
        bool gui_compatible;

        HoverToolTip gui_tooltip;
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_CALLSLOT_H_INCLUDED
