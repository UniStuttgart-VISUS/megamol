/*
 * ImageViewer.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_IMAGEVIEWER_H_INCLUDED
#define MEGAMOLCORE_IMAGEVIEWER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/Renderer3DModule.h"
#include "vislib/Pair.h"
#include "vislib/SmartPtr.h"
#include "vislib/graphics/gl/OpenGLTexture2D.h"
#include "vislib/math/Rectangle.h"

#ifdef WITH_MPI
#include "mpi.h"
#endif /* WITH_MPI */

/*
 * Copyright (C) 2010 by Sebastian Grottel.
 */
#include "mmcore/CallerSlot.h"
#include "vislib/RawStorage.h"
#include "vislib/graphics/AbstractBitmapCodec.h"

using namespace megamol::core;

namespace megamol {
namespace imageviewer2 {

/**
 * Mesh-based renderer for bézier curve tubes
 */
class ImageViewer : public view::Renderer3DModule {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "ImageViewer"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "A litte less simple Image Viewer"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    ImageViewer(void);

    /** Dtor. */
    virtual ~ImageViewer(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtents(Call& call);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool Render(Call& call);

private:
    /**
     * Splits a line at the semicolon into a left and right part. If there
     * is no semicolon, defaultEye governs which one of the strings is set,
     * the other one is emptied.
     */
    void interpretLine(const vislib::TString source, vislib::TString& left, vislib::TString& right);

    /**
     * Callback invoked when the user pastes a line containing
     * <machine>[;<machine>]*. Sets blankMachines to machines that
     * should not try loading the images (for performance reasons).
     */
    bool onBlankMachineSet(param::ParamSlot& slot);

    /**
     * Callback invoked when the user pastes a line containing
     * <leftimg>[;<rightimg>]. The text is split using interpretLine
     * and assigned to (left|right)FilenameSlot.
     */
    bool onFilesPasted(param::ParamSlot& slot);

    /**
     * Callback invoked when the user pastes a text containing multiple
     * lines containing <leftimg>[;<rightimg>]. Splits text into single
     * lines and then behaves like onFilesPasted for each line.
     */
    bool onSlideshowPasted(param::ParamSlot& slot);

    /** Callback for going back to slide 0 */
    bool onFirstPressed(param::ParamSlot& slot);

    /** Callback for going back one slide */
    bool onPreviousPressed(param::ParamSlot& slot);

    /** Callback for going forward one slide */
    bool onNextPressed(param::ParamSlot& slot);

    /** Callback for going forward to the last slide */
    bool onLastPressed(param::ParamSlot& slot);

    /**
     * Callback that occurs on slide change. Copies file names from
     * leftFiles and rightFiles to leftFilenameSlot and
     * rightFilenameSlot respectively based on currentSlot.
     */
    bool onCurrentSet(param::ParamSlot& slot);

    /** makes sure the image for the respective eye is loaded. */
    bool assertImage(bool rightEye);

    bool initMPI();

    /** The image file path slot */
    param::ParamSlot leftFilenameSlot;

    /** The image file path slot */
    param::ParamSlot rightFilenameSlot;

    /** Slot to receive both file names at once */
    param::ParamSlot pasteFilenamesSlot;

    /** Slot to receive a whole slideshow at once */
    param::ParamSlot pasteSlideshowSlot;

    /** slot for going back to slide 0 */
    param::ParamSlot firstSlot;

    /** slot for going back one slide */
    param::ParamSlot previousSlot;

    /** slide for setting the current slide index */
    param::ParamSlot currentSlot;

    /** slot for going forward one slide */
    param::ParamSlot nextSlot;

    /** slot for going forward to the last slide */
    param::ParamSlot lastSlot;

    /** slot for inserting machine names that will not load the images */
    param::ParamSlot blankMachine;

    /** if only one image per pair is defined: where it should go */
    param::ParamSlot defaultEye;

    /** slot for MPIprovider */
    CallerSlot callRequestMpi;

    /** slot for image data */
    CallerSlot callRequestImage;

#ifdef WITH_MPI
    /** The communicator that the view uses. */
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm roleComm;
#endif /* WITH_MPI */

    int mpiRank = 0;

    int mpiSize = 0;

    enum mpiRole { IMG_BLANK = 10, IMG_LEFT = 11, IMG_RIGHT = 12};

    int roleRank = -1, roleSize = -1;
    int rank = -1;
    mpiRole myRole;
    bool remoteness = false;
    int roleImgcRank = -1;

    vislib::TString loadedFile = "";

    /** The width of the image */
    unsigned int width;

    /** The height of the image */
    unsigned int height;

    /** The image tiles */
    vislib::Array<vislib::Pair<vislib::math::Rectangle<float>, vislib::SmartPtr<vislib::graphics::gl::OpenGLTexture2D>>>
        tiles;

    /** the slide show files for the left eye */
    vislib::Array<vislib::TString> leftFiles;

    /** the slide show files for the right eye */
    vislib::Array<vislib::TString> rightFiles;

    /** semicolon-separated machines that should not try to load images */
    vislib::Array<vislib::TString> blankMachines;

    /** cache for the local machine name */
    vislib::TString machineName;

    /** hash to check whether image has changed */
    size_t datahash;
};

} /* end namespace imageviewer2 */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_IMAGEVIEWER_H_INCLUDED */
