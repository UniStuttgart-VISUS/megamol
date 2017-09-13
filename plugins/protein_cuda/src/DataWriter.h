/*
 * DataWriter.h
 *
 * Copyright (C) 2012 by University of Stuttgart (VISUS).
 * All rights reserved.
 *
 * $Id$
 */

#ifndef MMPROTEINCUDAPLUGIN_DATAWRITER_H_INCLUDED
#define MMPROTEINCUDAPLUGIN_DATAWRITER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/job/AbstractJob.h"
#include "mmcore/view/TimeControl.h"
#include "mmcore/Module.h"
#include "mmcore/CallerSlot.h"

#include "protein_calls/CrystalStructureDataCall.h"


namespace megamol {
namespace protein_cuda {

/**
 * TODO
 */
class DataWriter : public core::job::AbstractJob, public core::Module {

public:

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "DataWriter";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) {
        return "Job writing data to files of different formats.";
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
    DataWriter();

    /**
     * Dtor
     */
    virtual ~DataWriter();

    /**
     * Answers whether or not this job is still running.
     *
     * @return 'true' if this job is still running, 'false' if it has
     *         finished.
     */
    virtual bool IsRunning(void) const;

    /**
     * Starts the job thread.
     *
     * @return true if the job has been successfully started.
     */
    virtual bool Start(void);

    /**
     * Terminates the job thread.
     *
     * @return true to acknowledge that the job will finish as soon
     *         as possible, false if termination is not possible.
     */
    virtual bool Terminate(void);

    /**
     * Calculate 3d texture of the dipole moment.
     *
     * @param dc          The data call.
     * @param offset      The time window to be averaged
     * @param quality     TODO
     * @param radscale    Scaling for the atom radius.
     * @param gridspacing Spacing for the grid.
     * @param isoval      Isovalue for the extracted surface.
     * @return True, if the calculation was successful.
     */
	bool CalcMapDipoleAvg(protein_calls::CrystalStructureDataCall *dc,
            int offset,
            int quality,
            float radscale,
            float gridspacing,
            float isoval);


    /**
     * Calculate 3d texture of the dipole moment.
     *
     * @param dc          The data call.
     * @param offset      The time window to be averaged
     * @param quality     TODO
     * @param radscale    Scaling for the atom radius.
     * @param gridspacing Spacing for the grid.
     * @param isoval      Isovalue for the extracted surface.
     * @return True, if the calculation was successful.
     */
	bool CalcMapTiDisplAvg(protein_calls::CrystalStructureDataCall *dc,
            int offset,
            int quality,
            float radscale,
            float gridspacing,
            float isoval);


    bool PutStatistics(unsigned int frameIdx0, unsigned int frameIdx1,
            unsigned int avgOffs);


    /**
     * Write one frame to the VTI xml format.
     *
     * @param filePrefix Prefix to the filename.
     * TODO
     * @return True, if the file could be written.
     */
    bool WriteFrame2VTI(std::string filePrefix,
            vislib::TString dataIdentifier,
            float org[3],
            float step[3],
            int dim[3],
            int cycle,
            float *data);

    bool WriteDipoleToVTI(unsigned int frameIdx0, unsigned int frameIdx1,
            unsigned int avgOffs);

    bool WriteTiDisplVTI(unsigned int frameIdx0, unsigned int frameIdx1,
            unsigned int avgOffs);

	float GetNearestDistTi(protein_calls::CrystalStructureDataCall *dc, int idx);

    /**
     * Print cell length averaged over 'offs' frames.
     *
     * @param idxStart The index of the first frame
     * @param offs     The frame offset to calculate the average
     * @param dc       The data call
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool PutAvgCellLength(unsigned int idxStart, unsigned int offs,
			protein_calls::CrystalStructureDataCall *dc);

    bool PutAvgCellLengthAlt(unsigned int idxStart, unsigned int idxEnd,
			protein_calls::CrystalStructureDataCall *dc);

    /**
     * Print the sidelength/Volume of the data set.
     *
     * @param[in] idx0 The frame index of the first frame
     * @param[in] idx1 The frame idx of the second frame
     * @param[in] dc The data call
     * @return 'True' on success, 'false' otherwise.
     */
    bool PutCubeSize(unsigned int frIdx0, unsigned int frIdx1,
			protein_calls::CrystalStructureDataCall *dc);

	bool WriteTiDispl(protein_calls::CrystalStructureDataCall *dc);
	bool ReadTiDispl(protein_calls::CrystalStructureDataCall *dc);

    /**
     * Determine the maximum (absolute) values for x, y and z values of the
     * vector field.
     *
     * @param[in] dc The data call
     * @return 'True' on success, 'false' otherwise.
     */
	bool GetMaxCoords(protein_calls::CrystalStructureDataCall *dc);

	bool PutCubeVol(protein_calls::CrystalStructureDataCall *dc);

protected:


    /**
     * TODO
     */
    float CalcCellVolume(
            vislib::math::Vector<float, 3> A,
            vislib::math::Vector<float, 3> B,
            vislib::math::Vector<float, 3> C,
            vislib::math::Vector<float, 3> D,
            vislib::math::Vector<float, 3> E,
            vislib::math::Vector<float, 3> F,
            vislib::math::Vector<float, 3> G,
            vislib::math::Vector<float, 3> H);

    /**
     * TODO
     */
    float CalcVolTetrahedron(
            vislib::math::Vector<float, 3> A,
            vislib::math::Vector<float, 3> B,
            vislib::math::Vector<float, 3> C,
            vislib::math::Vector<float, 3> D);

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
     * Write frame data provided to paraview's VTI image data file format.
     *
     * @param frameIdx The index of the frame.
     * @param fileName The name of the outputfile.
     * @return True, if the file could be written.
     */
    bool writeFrame2VTKLegacy(unsigned int frameIdx,
            float gridspacing,
            vislib::TString fileName);

	bool WriteFrameFileBinAvg(protein_calls::CrystalStructureDataCall *dc);

    void sortByKey(unsigned int *idx, unsigned int n, float *pos);

	bool WriteTiODipole(protein_calls::CrystalStructureDataCall *dc);

	bool ReadTiODipole(protein_calls::CrystalStructureDataCall *dc);

    void PutVelocity();

	void PutDisplacement(protein_calls::CrystalStructureDataCall *dc);

private:

    /** Data caller slot */
    core::CallerSlot dataCallerSlot;

    /** The time control */
    core::view::TimeControl timeCtrl;

    /** Flag whether the job is done */
    bool jobDone;


    /// Arrays ///

    /* Array containing data of the current frame */
    float *frameData0;

    /* Array containing data of the next frame */
    float *frameData1;

    /* Array containing data of the displacement frame */
    float *frameDataDispl;

    float* addedPos;
    float* addedTiDispl;


    //// Quicksurf branch ///
    // TODO

    void *cudaqsurf;         ///< Pointer to CUDAQuickSurf object if it exists
    double pretime;          ///< Internal timer for performance instrumentation
    double voltime;          ///< Internal timer for performance instrumentation
    double gradtime;         ///< Internal timer for performance instrumentation
    double mctime;           ///< Internal timer for performance instrumentation
    double mcverttime;       ///< Internal timer for performance instrumentation
    double reptime;          ///< Internal timer for performance instrumentation
    float solidcolor[3];     ///< RGB color to use when not using per-atom colors
    int numvoxels[3];        ///< Number of voxels in each dimension
    float origin[3];         ///< Origin of the volumetric map
    float xaxis[3];          ///< X-axis of the volumetric map
    float yaxis[3];          ///< Y-axis of the volumetric map
    float zaxis[3];          ///< Z-axis of the volumetric map

    /** The grid spacing */
    float step[3];

};

} /* end namespace protein_cuda */
} /* end namespace megamol */

#endif /* MMPROTEINCUDAPLUGIN_DATAWRITERJOB_H_INCLUDED */
