/*
 * PluginsStateFileGeneratorJob.h
 *
 * Copyright (C) 2016 by MegaMol Team; TU Dresden.
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_JOB_PLUGINSSTATEFILEGENERATORJOB_H_INCLUDED
#define MEGAMOLCORE_JOB_PLUGINSSTATEFILEGENERATORJOB_H_INCLUDED
#pragma once


#include "mmcore/job/AbstractThreadedJob.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include <fstream>
#include "utility/plugins/PluginManager.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/LinearTransferFunctionParam.h"


namespace megamol {
namespace core {
namespace job {


    /**
     * Class implementing a the job to collect informations of all plugins,
     * modules, and calls, for the use in the MegaMol configurator.
     */
    class PluginsStateFileGeneratorJob : public AbstractThreadedJob,
        public Module {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "PluginsStateFileGeneratorJob";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Class implementing a the job to collect informations of "
                   "all plugins, modules, and calls, for the use in the "
                   "MegaMol configurator.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /**
         * Disallow usage in quickstarts
         *
         * @return false
         */
        static bool SupportQuickstart(void) {
            return false;
        }

        /**
         * Ctor
         */
        PluginsStateFileGeneratorJob();

        /**
         * Dtor
         */
        virtual ~PluginsStateFileGeneratorJob();

    protected:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

    private:

        /**
         * Perform the work of a thread.
         *
         * @param userData A pointer to user data that are passed to the thread,
         *                 if it started.
         *
         * @return The application dependent return code of the thread. This 
         *         must not be STILL_ACTIVE (259).
         */
        virtual DWORD Run(void *userData);

        /** Writes info of one plugin */
        void writePluginInfo(std::ofstream& file,
            const factories::ModuleDescriptionManager *modMan,
            const factories::CallDescriptionManager *callMan) const;

        /** Writes info of one module */
        void WriteModuleInfo(std::ofstream& file,
            const factories::ModuleDescription* desc) const;

        /** Writes info of one call */
        void WriteCallInfo(std::ofstream& file,
            const factories::CallDescription* desc) const;

        /** Writes info of one parameter type */
        void WriteParamInfo(std::ofstream& file, const param::AbstractParam               * param) const;
        void WriteParamInfo(std::ofstream& file, const param::BoolParam                   * param) const;
        void WriteParamInfo(std::ofstream& file, const param::ButtonParam                 * param) const;
        void WriteParamInfo(std::ofstream& file, const param::EnumParam                   * param) const;
        void WriteParamInfo(std::ofstream& file, const param::FlexEnumParam               * param) const;
        void WriteParamInfo(std::ofstream& file, const param::FloatParam                  * param) const;
        void WriteParamInfo(std::ofstream& file, const param::IntParam                    * param) const;
        void WriteParamInfo(std::ofstream& file, const param::FilePathParam               * param) const;
        void WriteParamInfo(std::ofstream& file, const param::ColorParam                  * param) const;
        void WriteParamInfo(std::ofstream& file, const param::LinearTransferFunctionParam * param) const;

        /** The file name to store the data in */
        param::ParamSlot fileNameSlot;

    };


} /* end namespace job */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_JOB_PLUGINSSTATEFILEGENERATORJOB_H_INCLUDED */
