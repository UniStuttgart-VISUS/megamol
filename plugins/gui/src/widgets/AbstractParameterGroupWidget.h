/*
 * AbstractParameterGroupWidget.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_ABSTARCTPARAMETERGROUPWIDGET_INCLUDED
#define MEGAMOL_GUI_ABSTARCTPARAMETERGROUPWIDGET_INCLUDED


#include "GUIUtils.h"
#include "WidgetPicking_gl.h"
#include "graph/Parameter.h"


namespace megamol {
namespace gui {

    /**
     * Animation widget for parameter group.
     */
    class AbstractParameterGroupWidget : public megamol::core::param::AbstractParamPresentation {
    public:
        typedef std::vector<megamol::gui::Parameter*> ParamPtrVector_t;

        ~AbstractParameterGroupWidget(void) = default;

        virtual bool Check(bool only_check, ParamPtrVector_t& params) = 0;

        virtual bool Draw(ParamPtrVector_t params, const std::string& in_module_fullname, const std::string& in_search,
            megamol::gui::ParameterPresentation::WidgetScope in_scope, PickingBuffer* inout_picking_buffer) = 0;

        bool IsActive(void) const {
            return this->active;
        }

        void SetActive(bool active) {
            this->active = active;
        }

        std::string GetName(void) const {
            return this->name;
        };

    protected:
        // VARIABLES ----------------------------------------------------------

        bool active;
        std::string name;
        const ImGuiID uid;

        // FUNCTIONS ----------------------------------------------------------

        AbstractParameterGroupWidget(ImGuiID uid)
                : megamol::core::param::AbstractParamPresentation(), active(false), name(), uid(uid){};
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_ABSTARCTPARAMETERGROUPWIDGET_INCLUDED
