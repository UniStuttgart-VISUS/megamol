/*
 * mmvtkmFileLoader.h
 *
 * Copyright (C) 2020-2021 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_MMVTKM_VTKMFILELOADER_H_INCLUDED
#define MEGAMOL_MMVTKM_VTKMFILELOADER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/AnimDataModule.h"

#include "mmvtkm/mmvtkmDataCall.h"


namespace megamol {
namespace mmvtkm {


/**
 * Data source module for mmvtkm files
 */
class mmvtkmFileLoader : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "vtkmFileLoader";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "File loader module for vtkm files.";
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
    mmvtkmFileLoader(void);

    /** Dtor. */
    virtual ~mmvtkmFileLoader(void);

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

    /**
     * Callback receiving the update of the file name parameter.
     *
     * @param slot The updated ParamSlot.
     *
     * @return Always 'true' to reset the dirty flag.
     */
    bool filenameChanged(core::param::ParamSlot& slot);

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getDataCallback(core::Call& caller);

    /**
     * Gets the meta data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getMetaDataCallback(core::Call& caller);

private:
    uint32_t version_;

    /** The file name  */
    core::param::ParamSlot filename_;

    /** The slot for requesting data */
    core::CalleeSlot getDataCalleeSlot_;

    /** The vtkm data holder */
    std::shared_ptr<VtkmData> vtkmData_;
    VtkmMetaData vtkmMetaData_;

    /** The vtkm data file name */
    std::string vtkmDataFile_;

    /** Used as flag if file has changed */
    bool fileChanged_;
};

} /* end namespace mmvtkm */
} /* end namespace megamol */

#endif /* MEGAMOL_MMVTKM_VTKMFILELOADER_H_INCLUDED */
