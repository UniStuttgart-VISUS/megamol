/*
 * mmvtkmDataCall.h (MultiParticleDataCall)
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_MMVTKM_MMVTKMDATACALL_H_INCLUDED
#define MEGAMOL_MMVTKM_MMVTKMDATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vtkm/cont/DataSet.h"

#include "mmcore/AbstractGetData3DCall.h"
#include "mmcore/factories/CallAutoDescription.h"
//#include "mmcore/moldyn/SimpleSphericalParticles.h"

namespace megamol {
namespace mmvtkm {

// MEGAMOLCORE_APIEXT template class MEGAMOLCORE_API AbstractParticleDataCall<SimpleSphericalParticles>;


/**
 * Call for multi-stream particle data.
 */
class mmvtkmDataCall : public core::AbstractGetData3DCall {
public:
    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) { return "vtkmDataCall"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description(void) { return "Transports vtkm data."; }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) { return AbstractGetData3DCall::FunctionCount(); }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) { return AbstractGetData3DCall::FunctionName(idx); }

    /** Ctor. */
    mmvtkmDataCall(void);


    /** Dtor. */
    virtual ~mmvtkmDataCall(void);

    /**
     * Assignment operator.
     * Makes a deep copy of all members. While for data these are only
     * pointers, the pointer to the unlocker object is also copied.
     *
     * @param rhs The right hand side operand
     *
     * @return A reference to this
     */
    mmvtkmDataCall& operator=(const mmvtkmDataCall& rhs);

    /**
     * Sets the mmvtkm data set file from the data source
     *
     * @param filename The file containing the mmvtkm data set
     */
    void SetDataSet(vtkm::cont::DataSet* data) { this->mData = data; }

    /**
     * Returns the mmvtkm data set file from the data source
     *
     * @return The file containing the mmvtkm data set
     */
    vtkm::cont::DataSet* GetDataSet() { return this->mData; }


private:
    vtkm::cont::DataSet* mData;
};


/** Description class typedef */
typedef core::factories::CallAutoDescription<mmvtkmDataCall> mmvtkmDataCallDescription;


} /* end namespace mmvtkm */
} /* end namespace megamol */

#endif /* MEGAMOL_MMVTKM_MMVTKMDATACALL_H_INCLUDED */
