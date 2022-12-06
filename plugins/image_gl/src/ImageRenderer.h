/**
 * MegaMol
 * Copyright (c) 2010, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <memory>

#include <glowl/GLSLProgram.hpp>

#include "mmcore/param/ParamSlot.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"
#include "vislib/Pair.h"
#include "vislib/SmartPtr.h"
#include "vislib/math/Rectangle.h"
#include "vislib_gl/graphics/gl/OpenGLTexture2D.h"

#ifdef MEGAMOL_USE_MPI
#include "mpi.h"
#endif /* MEGAMOL_USE_MPI */

/*
 * Copyright (C) 2010 by Sebastian Grottel.
 */
#include "mmcore/CallerSlot.h"
#include "vislib/RawStorage.h"
#include "vislib/graphics/AbstractBitmapCodec.h"

namespace megamol {
namespace image_gl {

/**
 * Mesh-based renderer for bezier curve tubes
 */
class ImageRenderer : public mmstd_gl::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "ImageRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "A litte less simple Image Renderer";
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
    ImageRenderer(void);

    /** Dtor. */
    virtual ~ImageRenderer(void);

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
    virtual bool GetExtents(mmstd_gl::CallRender3DGL& call);

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
    virtual bool Render(mmstd_gl::CallRender3DGL& call);

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
    bool onBlankMachineSet(core::param::ParamSlot& slot);

    /**
     * Callback invoked when the user pastes a line containing
     * <leftimg>[;<rightimg>]. The text is split using interpretLine
     * and assigned to (left|right)FilenameSlot.
     */
    bool onFilesPasted(core::param::ParamSlot& slot);

    /**
     * Callback invoked when the user pastes a text containing multiple
     * lines containing <leftimg>[;<rightimg>]. Splits text into single
     * lines and then behaves like onFilesPasted for each line.
     */
    bool onSlideshowPasted(core::param::ParamSlot& slot);

    /** Callback for going back to slide 0 */
    bool onFirstPressed(core::param::ParamSlot& slot);

    /** Callback for going back one slide */
    bool onPreviousPressed(core::param::ParamSlot& slot);

    /** Callback for going forward one slide */
    bool onNextPressed(core::param::ParamSlot& slot);

    /** Callback for going forward to the last slide */
    bool onLastPressed(core::param::ParamSlot& slot);

    /**
     * Callback that occurs on slide change. Copies file names from
     * leftFiles and rightFiles to leftFilenameSlot and
     * rightFilenameSlot respectively based on currentSlot.
     */
    bool onCurrentSet(core::param::ParamSlot& slot);

    /** makes sure the image for the respective eye is loaded. */
    bool assertImage(bool rightEye);

    bool initMPI();

    vislib::TString stdToTString(const std::string& str);

    /** The image file path slot */
    core::param::ParamSlot leftFilenameSlot;

    /** The image file path slot */
    core::param::ParamSlot rightFilenameSlot;

    /** Slot to receive both file names at once */
    core::param::ParamSlot pasteFilenamesSlot;

    /** Slot to receive a whole slideshow at once */
    core::param::ParamSlot pasteSlideshowSlot;

    /** slot for going back to slide 0 */
    core::param::ParamSlot firstSlot;

    /** slot for going back one slide */
    core::param::ParamSlot previousSlot;

    /** slide for setting the current slide index */
    core::param::ParamSlot currentSlot;

    /** slot for going forward one slide */
    core::param::ParamSlot nextSlot;

    /** slot for going forward to the last slide */
    core::param::ParamSlot lastSlot;

    /** slot for inserting machine names that will not load the images */
    core::param::ParamSlot blankMachine;

    /** if only one image per pair is defined: where it should go */
    core::param::ParamSlot defaultEye;

    /** Index of the shown image if the one from the call is taken */
    core::param::ParamSlot shownImage;

    /** slot for MPIprovider */
    core::CallerSlot callRequestMpi;

    /** slot for image data */
    core::CallerSlot callRequestImage;

#ifdef MEGAMOL_USE_MPI
    /** The communicator that the view uses. */
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm roleComm;
#endif /* MEGAMOL_USE_MPI */

    int mpiRank = 0;

    int mpiSize = 0;

    enum mpiRole { IMG_BLANK = 10, IMG_LEFT = 11, IMG_RIGHT = 12 };

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

    bool newImageNeeded;

    std::shared_ptr<glowl::GLSLProgram> theShader_;
    GLuint theVertBuffer;
    GLuint theTexCoordBuffer;
    GLuint theVAO;

    /** The image tiles */
    vislib::Array<
        vislib::Pair<vislib::math::Rectangle<float>, vislib::SmartPtr<vislib_gl::graphics::gl::OpenGLTexture2D>>>
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

    /** Flag determining whether the tiles have been recalculated */
    bool new_tiles_;
};

} // namespace image_gl
} /* end namespace megamol */
