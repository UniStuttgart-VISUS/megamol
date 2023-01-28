/*
 * mmvtkmDataSource.h
 *
 * Copyright (C) 2020-2021 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_MMVTKM_VTKMDATASOURCE_H_INCLUDED
#define MEGAMOL_MMVTKM_VTKMDATASOURCE_H_INCLUDED
#pragma once


#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/data/AnimDataModule.h"

#include "mmvtkm/mmvtkmDataCall.h"


namespace megamol::mmvtkm {


/**
 * Data source module for VTKM files
 */
class mmvtkmDataSource : public core::view::AnimDataModule {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "vtkmDataSource";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Data source module for vtkm files.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor. */
    mmvtkmDataSource();

    /** Dtor. */
    ~mmvtkmDataSource() override;

protected:
    /**
     * Creates a frame to be used in the frame cache. This method will be
     * called from within 'initFrameCache'.
     *
     * @return The newly created frame object.
     */
    core::view::AnimDataModule::Frame* constructFrame() const override;

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Loads one frame of the data set into the given 'frame' object. This
     * method may be invoked from another thread. You must take
     * precausions in case you need synchronised access to shared
     * ressources.
     *
     * @param frame The frame to be loaded.
     * @param idx The index of the frame to be loaded.
     */
    void loadFrame(core::view::AnimDataModule::Frame* frame, unsigned int idx) override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

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

    /** The slot for requesting data */
    core::CalleeSlot getDataCalleeSlot_;

    /** caller slot */
    core::CallerSlot nodesAdiosCallerSlot_;
    core::CallerSlot labelAdiosCallerSlot_;

    /** Data file load id counter */
    size_t oldNodeDataHash_;
    size_t oldLabelDataHash_;

    /** The vtkm data holder */
    std::shared_ptr<VtkmData> vtkmData_;
    VtkmMetaData vtkmMetaData_;

    /** The vtkm data file name */
    std::string vtkmDataFile_;
};

} // namespace megamol::mmvtkm

#endif /* MEGAMOL_MMVTKM_VTKMDATASOURCE_H_INCLUDED */
