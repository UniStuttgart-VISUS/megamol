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
    
    bool Present(configurator::ParamVectorType& inout_params);


private:




};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_PARAMETERGROUPPRESENTATION_H_INCLUDED
