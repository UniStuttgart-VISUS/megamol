/*
 * ParameterManager.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_PARAMETERMANAGER_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_PARAMETERMANAGER_H_INCLUDED

#include "mmcore/CoreInstance.h"
#include "mmcore/param/ParamSlot.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/TernaryParam.h"
#include "mmcore/param/TransferFunctionParam.h"
#include "mmcore/param/Vector2fParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/Vector4fParam.h"


namespace megamol {
namespace gui {


/** ************************************************************************
 * Defines the parameter manager.
 */
class ParameterManager {
public:
    ParameterManager(void);

    virtual ~ParameterManager(void);


private:
    // VARIABLES --------------------------------------------------------------


    // FUNCTIONS --------------------------------------------------------------
};

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_PARAMETERMANAGER_H_INCLUDED
