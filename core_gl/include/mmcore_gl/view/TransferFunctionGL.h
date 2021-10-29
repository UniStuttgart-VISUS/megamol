/*
 * TransferFunction.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_TRANSFERFUNCTION_H_INCLUDED
#define MEGAMOLCORE_TRANSFERFUNCTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "mmcore/CoreInstance.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/param/TransferFunctionParam.h"
#include "mmcore_gl/view/CallGetTransferFunctionGL.h"

#include "vislib/sys/sysfunctions.h"
#include "mmcore/utility/log/Log.h"


namespace megamol {
namespace core_gl {
namespace view {


    /**
     * Module defining a transfer function.
     */
    class MEGAMOLCORE_API TransferFunctionGL : public core::Module {
    public:
        
        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "TransferFunction";
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
        TransferFunctionGL(void);

        /** Dtor. */
        virtual ~TransferFunctionGL(void);

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
        bool requestTF(core::Call& call);

        // VARIABLES ----------------------------------------------------------

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */

        /** The callee slot called on request of a transfer function */
        core::CalleeSlot getTFSlot;

        /** Parameter containing the transfer function data serialized into JSON string */
        core::param::ParamSlot tfParam;

        /** The OpenGL texture object id */
        unsigned int texID;

        /** The texture size in texel */
        unsigned int texSize;

        /** The texture data */
        std::vector<float> tex;

        /** The texture format */
        CallGetTransferFunctionGL::TextureFormat texFormat;

        /** The interpolation mode */
        core::param::TransferFunctionParam::InterpolationMode interpolMode;

        /** The value range */
        std::array<float, 2> range;

        /** Version of texture */
        uint32_t version;

        /** Global frame ID */
        uint32_t last_frame_id;

#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */
    };


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_TRANSFERFUNCTION_H_INCLUDED */
