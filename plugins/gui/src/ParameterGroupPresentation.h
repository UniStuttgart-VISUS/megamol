/*
 * ParameterGroupPresentation.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_PARAMETERGROUPPRESENTATION_H_INCLUDED
#define MEGAMOL_GUI_PARAMETERGROUPPRESENTATION_H_INCLUDED


#include "GUIUtils.h"
#include "configurator/Parameter.h"


namespace megamol {
namespace gui {


/** ************************************************************************
 * Defines parameter widget groups depending on parameter namespaces.
 */
class ParameterGroupPresentation {
public:
    ParameterGroupPresentation(void);
    ~ParameterGroupPresentation(void);

    bool PresentGUI(megamol::gui::configurator::ParamVectorType& inout_params, const std::string& in_module_fullname,
        const std::string& in_search, bool in_extended, bool in_ignore_extended,
        megamol::gui::configurator::ParameterPresentation::WidgetScope in_scope,
        const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool& out_open_external_tf_editor);


private:
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_PARAMETERGROUPPRESENTATION_H_INCLUDED
