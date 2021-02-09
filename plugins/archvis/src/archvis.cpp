/*
 * archvis.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/utility/plugins/PluginRegister.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "ArchVisMSMDataSource.h"
#include "CreateFEMModel.h"
#include "CreateMSM.h"
#include "FEMDataCall.h"
#include "FEMModel.h"
#include "FEMMeshDataSource.h"
#include "FEMMaterialDataSource.h"
#include "FEMRenderTaskDataSource.h"
#include "FEMLoader.h"
#include "MSMConvexHullMeshDataSource.h"
#include "MSMDataCall.h"
#include "MSMRenderTaskDataSource.h"

namespace megamol::archvis {
    /** Implementing the instance class of this plugin */
    class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
        REGISTERPLUGIN(plugin_instance)
    public:
        /** ctor */
        plugin_instance(void)
            : ::megamol::core::utility::plugins::Plugin200Instance(

                /* machine-readable plugin assembly name */
                "archvis", // TODO: Change this!

                /* human-readable plugin description */
                "Describing archvis (TODO: Change this!)") {

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
            //   this->module_descriptions.RegisterAutoDescription<megamol::archvis::MyModule1>();
            //   this->module_descriptions.RegisterAutoDescription<megamol::archvis::MyModule2>();
            //   ...
            //
			this->module_descriptions.RegisterAutoDescription<megamol::archvis::ArchVisMSMDataSource>();

            this->module_descriptions.RegisterAutoDescription<megamol::archvis::CreateFEMModel>();
            this->module_descriptions.RegisterAutoDescription<megamol::archvis::CreateMSM>();
			this->module_descriptions.RegisterAutoDescription<megamol::archvis::FEMMeshDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::archvis::FEMMaterialDataSource>();
			this->module_descriptions.RegisterAutoDescription<megamol::archvis::FEMRenderTaskDataSource>();
			this->module_descriptions.RegisterAutoDescription<megamol::archvis::FEMLoader>();
            this->module_descriptions.RegisterAutoDescription<megamol::archvis::MSMConvexHullDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::archvis::MSMRenderTaskDataSource>();


            // register calls here:

            //
            // TODO: Register your plugin's calls here
            // like:
            //   this->call_descriptions.RegisterAutoDescription<megamol::archvis::MyCall1>();
            //   this->call_descriptions.RegisterAutoDescription<megamol::archvis::MyCall2>();
            //   ...
            //
			this->call_descriptions.RegisterAutoDescription<megamol::archvis::FEMDataCall>();
            this->call_descriptions.RegisterAutoDescription<megamol::archvis::MSMDataCall>();

        }
    };
} // namespace megamol::archvis
