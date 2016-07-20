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
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "vislib/sys/BufferedFile.h"



namespace megamol {
namespace core {
namespace view {


    /**
     * Module defining a piecewise linear transfer function based on the
     * interval [0..1]
     */
    class LinearTransferFunction : public Module {
    public:

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

        /**
         * Maximum number of intermediate colour definitions
         */
        static const SIZE_T INTER_COLOUR_COUNT = 11;

        /**
         * struct holding all members required for intermediate colour definition
         */
        typedef struct interColour_t {

            /** The enable flag slot */
            param::ParamSlot *enableSlot;

            /** The colour slot */
            param::ParamSlot *colSlot;

            /** The value slot */
            param::ParamSlot *valSlot;

        } InterColour;

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
         * @return 'true' on success
         */
        bool requestTF(Call& call);

        /**
         * Callback called when the TFload button is pressed.
         *
         * @param slot The slot causing it
         *
         * @return 'true' on success
         */
        bool loadTFPressed(param::ParamSlot& slot);

        /**
         * Callback called when the TFstore button is pressed.
         *
         * @param slot The slot causing it
         *
         * @return 'true' on success
         */
        bool storeTFPressed(param::ParamSlot& slot);

        /** convenience for serializing a ParamSlot */
        void writeParameterFileParameter(param::ParamSlot& param,
            vislib::sys::BufferedFile &outFile);

        /** The callee slot called on request of a transfer function */
        CalleeSlot getTFSlot;

        /** The slot defining the colour for the minimum value */
        param::ParamSlot minColSlot;

        /** The slot defining the colour for the maximum value */
        param::ParamSlot maxColSlot;

        /** The slot defining the texture size to generate */
        param::ParamSlot texSizeSlot;

        /** The slot containing a path for (de)serializing the current TF */
        param::ParamSlot pathSlot;

        /** Button for loading the TF from the pathSlot file */
        param::ParamSlot loadTFSlot;

        /** Button for storing the TF in the pathSlot file */
        param::ParamSlot storeTFSlot;

        /** The OpenGL texture object id */
        unsigned int texID;

        /** The texture size in texel */
        unsigned int texSize;

        /** The texture format */
        CallGetTransferFunction::TextureFormat texFormat;

        /** The array of intermediate colour definitions */
        InterColour interCols[INTER_COLOUR_COUNT];

        bool firstRequest;

    };


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_LINEARTRANSFERFUNCTION_H_INCLUDED */
