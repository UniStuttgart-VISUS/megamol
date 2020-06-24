/*
 * ParameterGroupPresentation.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ParameterGroupPresentation.h"


using namespace megamol;
using namespace megamol::gui;
using namespace megamol::gui::configurator;


megamol::gui::ParameterGroupPresentation::ParameterGroupPresentation(void) {}


megamol::gui::ParameterGroupPresentation::~ParameterGroupPresentation(void) {}


bool megamol::gui::ParameterGroupPresentation::Present(megamol::gui::configurator::ParamVectorType& inout_params) {
    
    std::map<std::string, std::vector<megamol::gui::ParamType>> group_map;
    
    for (auto& param : inout_params) {
        std::string group_name = param.GetNameSpace();
        auto param_type = param.type;
        
    }
    
    
    
    
    
    
    
    return true;
}
