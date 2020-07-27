/*
 * Cinematic.cpp
*
* Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/utility/plugins/PluginRegister.h"
#include "mmcore/versioninfo.h"

#include "vislib/vislibversion.h"

#include "TrackingShotRenderer.h"
#include "TimeLineRenderer.h"
#include "CallKeyframeKeeper.h"
#include "KeyframeKeeper.h"
#include "CinematicView.h"
#include "ReplacementRenderer.h"

namespace megamol::cinematic {
    /** Implementing the instance class of this plugin */
    class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
        REGISTERPLUGIN(plugin_instance)
    public:

        /** ctor */
        plugin_instance(void)
            : ::megamol::core::utility::plugins::Plugin200Instance(

                /* machine-readable plugin assembly name */
                "Cinematic", 

                /* human-readable plugin description */
                "The Cinematic plugin allows the video rendering (separate file per frame) of any rendering output in MegaMol." 
                "By defining fixed keyframes for desired camera positions and specific animation times, arbitrary tracking shots can be created.") {
			
            // here we could perform addition initialization
        };
        /** Dtor */
        virtual ~plugin_instance(void) {
            // here we could perform addition de-initialization
        }
        /** Registers modules and calls */
        virtual void registerClasses(void) {
			
            // register modules here:
			this->module_descriptions.RegisterAutoDescription<megamol::cinematic::TrackingShotRenderer>();
			this->module_descriptions.RegisterAutoDescription<megamol::cinematic::TimeLineRenderer>();
			this->module_descriptions.RegisterAutoDescription<megamol::cinematic::KeyframeKeeper>();
			this->module_descriptions.RegisterAutoDescription<megamol::cinematic::CinematicView>();
            this->module_descriptions.RegisterAutoDescription<megamol::cinematic::ReplacementRenderer>();

            // register calls here:
			this->call_descriptions.RegisterAutoDescription < megamol::cinematic::CallKeyframeKeeper>();

        }
    };
} // namespace megamol::cinematic
