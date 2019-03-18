/*
 * LinearTransferFunction.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_LINEARTRANSFERFUNCTION_H_INCLUDED
#define MEGAMOLCORE_LINEARTRANSFERFUNCTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/LinearTransferFunctionParam.h"

#include "vislib/sys/sysfunctions.h"
#include "vislib/sys/Log.h"


namespace megamol {
namespace core {
namespace view {


    /**
     * Module defining a piecewise linear transfer function based on the interval [0..1]
     */
    class MEGAMOLCORE_API LinearTransferFunction : public Module {
    public:

        /**
        * Linear interpolation of transfer function data in range [0..texsize]
        */
        static void LinearInterpolation(std::vector<float> &out_texdata, unsigned int in_texsize, const megamol::core::param::LinearTransferFunctionParam::TFType &in_tfdata);

        // --------------------------------------------------------------------

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "LinearTransferFunction";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Module defining a piecewise linear transfer function";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor. */
        LinearTransferFunction(void);

        /** Dtor. */
        virtual ~LinearTransferFunction(void);


    private:

        // FUNCTIONS ----------------------------------------------------------

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

        /**
         * Callback called when the transfer function is requested.
         *
         * @param call The calling call
         *
         * @return 'true' on success, 'false' otherwise.
         */
        bool requestTF(Call& call);

        /**
         * Callback checks if any parameter is dirty.
         *
         * @param call The calling call
         *
         * @return 'true' on success, 'false' otherwise.
         */
        bool interfaceIsDirty(Call& call);

        /**
         * Callback resets all dirty parameters.
         *
         * @param call The calling call
         *
         * @return 'true' on success, 'false' otherwise.
         */
        bool interfaceResetDirty(Call& call);

        // VARIABLES ----------------------------------------------------------

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */

        /** The callee slot called on request of a transfer function */
        CalleeSlot getTFSlot;

        /** Parameter continaing the transfer function data serialized into JSON string */
        param::ParamSlot tfParam;

        /** The OpenGL texture object id */
        unsigned int texID;

        /** The texture size in texel */
        unsigned int texSize;

        /** The texture data */
        std::vector<float> tex;

        /** The texture format */
        CallGetTransferFunction::TextureFormat texFormat;

        /** The interpolation mode */
        param::LinearTransferFunctionParam::InterpolationMode interpolMode;

#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */
    };


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_LINEARTRANSFERFUNCTION_H_INCLUDED */
