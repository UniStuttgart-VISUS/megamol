/*
 * mmvtkmDataCall.h
 *
 * Copyright (C) 2020-2021 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once


#include "mmcore/factories/CallAutoDescription.h"
#include "mmstd/generic/CallGeneric.h"
#include <memory>
//#include "mmstd/data/AbstractGetData3DCall.h"

#include "vtkm/cont/DataSet.h"


namespace megamol::mmvtkm {

struct VtkmData {
    /** Vtkm dataset storage */
    vtkm::cont::DataSet data;
};

struct VtkmMetaData {
    /** Min and max bounds of the dataset */
    vtkm::Bounds minMaxBounds = vtkm::Bounds(0.f, 1.f, 0.f, 1.f, 0.f, 1.f);
    std::vector<std::string> fieldNames;
};

class mmvtkmDataCall : public core::GenericVersionedCall<std::shared_ptr<VtkmData>, VtkmMetaData> {
public:
    inline mmvtkmDataCall() : GenericVersionedCall<std::shared_ptr<VtkmData>, VtkmMetaData>() {}
    ~mmvtkmDataCall() override = default;

    static const char* ClassName() {
        return "vtkmDataCall";
    }
    static const char* Description() {
        return "Transports vtkm data.";
    }
};

/**
 * Call for vtkm data.
 */
//class mmvtkmDataCall : public core::AbstractGetData3DCall {
//public:
//    /**
//     * Answer the name of the objects of this description.
//     *
//     * @return The name of the objects of this description.
//     */
//    static const char* ClassName(void) { return "vtkmDataCall"; }
//
//    /**
//     * Answer a human readable description of this module.
//     *
//     * @return A human readable description of this module.
//     */
//    static inline const char* Description(void) { return "Transports vtkm data."; }
//
//    /**
//     * Answer the number of functions used for this call.
//     *
//     * @return The number of functions used for this call.
//     */
//    static unsigned int FunctionCount(void) { return AbstractGetData3DCall::FunctionCount(); }
//
//    /**
//     * Answer the name of the function used for this call.
//     *
//     * @param idx The index of the function to return it's name.
//     *
//     * @return The name of the requested function.
//     */
//    static const char* FunctionName(unsigned int idx) { return AbstractGetData3DCall::FunctionName(idx); }
//
//    /** Ctor. */
//    mmvtkmDataCall(void);
//
//
//    /** Dtor. */
//    virtual ~mmvtkmDataCall(void);
//
//    /**
//     * Assignment operator.
//     * Makes a deep copy of all members. While for data these are only
//     * pointers, the pointer to the unlocker object is also copied.
//     *
//     * @param rhs The right hand side operand
//     *
//     * @return A reference to this
//     */
//    mmvtkmDataCall& operator=(const mmvtkmDataCall& rhs);
//
//    /**
//     * Sets the mmvtkm data set file from the data source
//     *
//     * @param filename The file containing the mmvtkm data set
//     */
//    void SetDataSet(const vtkm::cont::DataSet* data) { this->data_ = data; }
//
//    /**
//     * Returns the mmvtkm data set file from the data source
//     *
//     * @return The file containing the mmvtkm data set
//     */
//    const vtkm::cont::DataSet* GetDataSet() const { return this->data_; }
//
//    /**
//     * Sets the value if data has been changed within the vtkm data source
//     *
//     * @param true, if data has changed, false otherwise
//     */
//    void UpdateDataChanges(bool update) { this->dataChanges_ = update; }
//
//    /**
//     * Returns if the data within the vtkm data source has changed or not
//     */
//    bool HasUpdate() const { return this->dataChanges_; }
//
//    /**
//    * Sets lower and upper bounds of the current dataset
//    */
//    void SetBounds(const vtkm::Bounds& bounds) {
//        this->minMaxBounds_ = bounds;
//    }
//
//    /**
//    * Returns lower and upper bounds of the current dataset
//    */
//    const vtkm::Bounds GetBounds() const {
//        return this->minMaxBounds_;
//    }
//
//    /**
//    * Sets lower and upper bounds of the current dataset
//    */
//    void SetFieldNames(const std::vector<std::string>& fieldnames) {
//            this->fieldNames_ = fieldnames;
//    }
//
//    /**
//    * Returns lower and upper bounds of the current dataset
//    */
//    const std::vector<std::string> GetFieldNames() const {
//        return this->fieldNames_;
//    }
//
//
//private:
//    /** Vtkm dataset storage */
//    const vtkm::cont::DataSet* data_;
//
//    /** Min and max bounds of the dataset */
//    vtkm::Bounds minMaxBounds_;
//
//    /** Fieldnames available in the current dataset */
//    std::vector<std::string> fieldNames_;
//
//    /** True, if data was changed, false otherwise */
//    bool dataChanges_;
//};


/** Description class typedef */
typedef core::factories::CallAutoDescription<mmvtkmDataCall> vtkmDataCallDescription;


} // namespace megamol::mmvtkm
