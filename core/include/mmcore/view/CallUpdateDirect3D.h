/*
 * CallUpdateDirect3D.h
 *
 * Copyright (C) 2012 by Visualisierungsinstitut der Universität Stuttgart.
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CALLUPDATEDIRECT3D_H_INCLUDED
#define MEGAMOLCORE_CALLUPDATEDIRECT3D_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/mmd3d.h"



namespace megamol {
namespace core {
namespace view {


    class CallUpdateDirect3D : public Call {

    public:

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        inline static const char *ClassName(void) {
            return "CallUpdateDirect3D";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        inline static const char *Description(void) {
            return "Call for updating Direct3D resources";
        }

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        inline static unsigned int FunctionCount(void) {
            return (sizeof(FUNCTIONS) / sizeof(*FUNCTIONS));
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char *FunctionName(unsigned int idx);

        CallUpdateDirect3D(void);

        inline CallUpdateDirect3D(const CallUpdateDirect3D& rhs) 
                : device(NULL) {
            *this = rhs;
        }

        virtual ~CallUpdateDirect3D(void);

        inline ID3D11Device *PeekDevice(void) {
            return this->device;
        }

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         */
        CallUpdateDirect3D& operator =(const CallUpdateDirect3D& rhs);

        /**
         * Set the device to be transported to the callee.
         *
         * @param device
         */
        virtual void SetDevice(ID3D11Device *device);

    protected:

        /** Super class typedef. */
        typedef Call Base;

        /** 
         * The map of function names for use in FunctionCount() and
         * FunctionName().
         */
        static const char *FUNCTIONS[1];

        ID3D11Device *device;

    };

    /** Description class typedef */
    typedef factories::CallAutoDescription<CallUpdateDirect3D> CallUpdateDirect3DDescription;

} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CALLUPDATEDIRECT3D_H_INCLUDED */