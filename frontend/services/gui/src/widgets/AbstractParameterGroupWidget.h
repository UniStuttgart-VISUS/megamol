/*
 * AbstractParameterGroupWidget.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_ABSTARCTPARAMETERGROUPWIDGET_INCLUDED
#define MEGAMOL_GUI_ABSTARCTPARAMETERGROUPWIDGET_INCLUDED
#pragma once


#include "graph/Parameter.h"
#include "mmcore/utility/Picking_gl.h"


namespace megamol {
namespace gui {

    /** ************************************************************************
     * Animation widget for parameter group
     */
    class AbstractParameterGroupWidget : public megamol::core::param::AbstractParamPresentation {
    public:
        typedef std::vector<megamol::gui::Parameter*> ParamPtrVector_t;

        ~AbstractParameterGroupWidget() override = default;

        virtual bool Check(bool only_check, ParamPtrVector_t& params) = 0;

        virtual bool Draw(ParamPtrVector_t params, const std::string& in_search,
            megamol::gui::Parameter::WidgetScope in_scope,
            megamol::core::utility::PickingBuffer* inout_picking_buffer) = 0;

        bool IsActive() const {
            return this->active;
        }

        void SetActive(bool a) {
            this->active = a;
        }

        std::string GetName() const {
            return this->name;
        };

    protected:
        // VARIABLES ----------------------------------------------------------

        bool active;
        std::string name;
        const ImGuiID uid;

        // FUNCTIONS ----------------------------------------------------------

        explicit AbstractParameterGroupWidget(ImGuiID uid)
                : megamol::core::param::AbstractParamPresentation(), active(false), name(), uid(uid){};
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_ABSTARCTPARAMETERGROUPWIDGET_INCLUDED
