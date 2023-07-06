/*
 * MeshTranslateRotateScale.h
 *
 * Copyright (C) 2018 MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once


#include "datatools/AbstractMeshManipulator.h"
#include "mmcore/param/ParamSlot.h"


namespace megamol::datatools_gl {

/**
 * Module thinning the number of particles
 *
 * Migrated from SGrottel particle's tool box
 */
class MeshTranslateRotateScale : public datatools::AbstractMeshManipulator {
public:
    /** Return module class name */
    static const char* ClassName() {
        return "MeshTranslateRotateScale";
    }

    /** Return module class description */
    static const char* Description() {
        return "Rotates, translates and scales the data";
    }

    /** Module is always available */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor */
    MeshTranslateRotateScale();

    /** Dtor */
    ~MeshTranslateRotateScale() override;
    bool InterfaceIsDirty() const;
    void InterfaceResetDirty();

protected:
    /**
     * Manipulates the particle data
     *
     * @remarks the default implementation does not changed the data
     *
     * @param outData The call receiving the manipulated data
     * @param inData The call holding the original data
     *
     * @return True on success
     */
    bool manipulateData(
        megamol::geocalls_gl::CallTriMeshDataGL& outData, megamol::geocalls_gl::CallTriMeshDataGL& inData) override;

private:
    core::param::ParamSlot translateSlot;
    core::param::ParamSlot quaternionSlot;
    core::param::ParamSlot scaleSlot;

    float** finalData;
    geocalls_gl::CallTriMeshDataGL::Mesh* mesh;
};

} // namespace megamol::datatools_gl
