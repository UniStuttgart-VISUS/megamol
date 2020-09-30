/*
 * GraphCollection.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_GRAPHCOLLECTION_PRESENTATION_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_GRAPHCOLLECTION_PRESENTATION_H_INCLUDED


#include "GUIUtils.h"
#include "widgets/FileBrowserWidget.h"
#include "widgets/MinimalPopUp.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/Module.h"
#include "mmcore/param/AbstractParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/utility/plugins/AbstractPluginInstance.h"

#include "utility/plugins/PluginManager.h"


namespace megamol {
namespace gui {


// Forward declarations
class GraphCollection;


/** ************************************************************************
 * Defines GUI graph collection presentation.
 */
class GraphCollectionPresentation {
public:
    friend class GraphCollection;
    // FUNCTIONS --------------------------------------------------------------

    GraphCollectionPresentation(void);
    ~GraphCollectionPresentation(void);

    void SaveProjectToFile(bool open_popup, GraphCollection& inout_graph_collection, GraphState_t& state);

private:
    // VARIABLES --------------------------------------------------------------

    FileBrowserWidget file_browser;
    ImGuiID graph_delete_uid;

    // FUNCTIONS --------------------------------------------------------------

    void Present(GraphCollection& inout_graph_collection, GraphState_t& state);
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_GRAPHCOLLECTION_PRESENTATION_H_INCLUDED
