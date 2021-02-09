/*
 * compositing.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/utility/plugins/PluginRegister.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "compositing/CompositingCalls.h"
#include "DrawToScreen.h"
#include "InteractionRenderTarget.h"
#include "LocalLighting.h"
#include "ScreenSpaceEffect.h"
#include "SimpleRenderTarget.h"
#include "TextureCombine.h"
#include "TextureDepthCompositing.h"

namespace megamol::compositing {
    /** Implementing the instance class of this plugin */
    class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
        REGISTERPLUGIN(plugin_instance)
    public:
        /** ctor */
        plugin_instance(void)
            : ::megamol::core::utility::plugins::Plugin200Instance(

                /* machine-readable plugin assembly name */
                "compositing_gl", // TODO: Change this!

                /* human-readable plugin description */
                "Describing compositing (TODO: Change this!)") {

            // here we could perform addition initialization
        };
        /** Dtor */
        virtual ~plugin_instance(void) {
            // here we could perform addition de-initialization
        }
        /** Registers modules and calls */
        virtual void registerClasses(void) {

            // register modules here:

            //
            // TODO: Register your plugin's modules here
            // like:
            //   this->module_descriptions.RegisterAutoDescription<megamol::compositing::MyModule1>();
            //   this->module_descriptions.RegisterAutoDescription<megamol::compositing::MyModule2>();
            //   ...
            //
            this->module_descriptions.RegisterAutoDescription<megamol::compositing::DrawToScreen>();
            this->module_descriptions.RegisterAutoDescription<megamol::compositing::InteractionRenderTarget>();
            this->module_descriptions.RegisterAutoDescription<megamol::compositing::LocalLighting>();
            this->module_descriptions.RegisterAutoDescription<megamol::compositing::ScreenSpaceEffect>();
            this->module_descriptions.RegisterAutoDescription<megamol::compositing::SimpleRenderTarget>();
            this->module_descriptions.RegisterAutoDescription<megamol::compositing::TextureCombine>();
            this->module_descriptions.RegisterAutoDescription<megamol::compositing::TextureDepthCompositing>();

            // register calls here:

            //
            // TODO: Register your plugin's calls here
            // like:
            //   this->call_descriptions.RegisterAutoDescription<megamol::compositing::MyCall1>();
            //   this->call_descriptions.RegisterAutoDescription<megamol::compositing::MyCall2>();
            //   ...
            //
            this->call_descriptions.RegisterAutoDescription<megamol::compositing::CallTexture2D>();
            this->call_descriptions.RegisterAutoDescription<megamol::compositing::CallCamera>();
            this->call_descriptions.RegisterAutoDescription<megamol::compositing::CallFramebufferGL>();

        }
    };
} // namespace megamol::compositing
