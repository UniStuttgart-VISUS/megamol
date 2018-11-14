/*
 * PoreNetExtractor.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/view/Renderer3DModule.h"
#include "ArxelBuffer.h"
#include "AbstractQuartzModule.h"
#include "BufferMTPConnection.h"
#include "LoopBuffer.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/sys/File.h"
#include "vislib/graphics/gl/FramebufferObject.h"
#include "vislib/graphics/gl/glfunctions.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/sys/RunnableThread.h"
#include "mmcore/view/CallClipPlane.h"
#include "QuartzCrystalDataCall.h"
#include "QuartzParticleGridDataCall.h"
#include "PoreNetSliceProcessor.h"
#include "PoreMeshProcessor.h"


namespace megamol {
namespace demos {

    /**
     * Module for extracting and rendering PoreNetwork
     */
    class PoreNetExtractor : public core::view::Renderer3DModule, public AbstractQuartzModule {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "PoreNetExtractor";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Module managing and extracting a pore net";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable()
                && vislib::graphics::gl::FramebufferObject::AreExtensionsAvailable()
                && (vislib::graphics::gl::GLVersion() >= vislib::VersionNumber(2, 0));
        }

        /**
         * Ctor
         */
        PoreNetExtractor(void);

        /**
         * Dtor
         */
        virtual ~PoreNetExtractor(void);

    protected:

        /**
         * The get extents callback. The module should set the members of
         * 'call' to tell the caller the extents of its data (bounding boxes
         * and times).
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetExtents(core::Call& call);

        /**
         * The render callback.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool Render(core::Call& call);

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

    private:

        /** Possible extraction directions */
        enum ExtractionDir {
            EXTDIR_X = 0,
            EXTDIR_Y = 1,
            EXTDIR_Z = 2
        };

        /**
         * The extract button was clicked
         *
         * @param slot The calling slot
         *
         * @return True
         */
        bool onExtractBtnClicked(core::param::ParamSlot& slot);

        /**
         * The extract button was clicked
         *
         * @param slot The calling slot
         *
         * @return True
         */
        bool onLoadBtnClicked(core::param::ParamSlot& slot);

        /**
         * The extract button was clicked
         *
         * @param slot The calling slot
         *
         * @return True
         */
        bool onSaveBtnClicked(core::param::ParamSlot& slot);

        /**
         * Answer whether or not the extraction process is running
         *
         * @return True if the extraction process is running
         */
        bool isExtractionRunning(void);

        /**
         * Cancels the current extraction process
         */
        void abortExtraction(void);

        /**
         * Performs one synchrone step of the extraction process
         */
        void performExtraction(void);

        /**
         * Writes the file header
         *
         * @param file The file to write the file header to
         */
        void writeFileHeader(vislib::sys::File& file);

        /**
         * Finalizes and closes the data file
         *
         * @param file The data file written
         */
        void closeFile(vislib::sys::File& file);

        /** Removes all data */
        void clear(void);

        /**
         * Ensures the actuality of the type texture
         *
         * @param types The types
         */
        void assertTypeTexture(CrystalDataCall& types);

        /** Releases the type texture */
        void releaseTypeTexture(void);

        /**
         * Draws the particles on the clipping plane into the current output
         * buffer
         *
         * @param pgdc The particle data
         * @param tdc The crystalite data
         * @param ccp The clipping plane
         */
        void drawParticles(ParticleGridDataCall *pgdc, CrystalDataCall *tdc,
            core::view::CallClipPlane *ccp);

        /** The type texture */
        unsigned int typeTexture;

        /** The data bounding box */
        vislib::math::Cuboid<float> bbox;

        /** The data clipping box */
        vislib::math::Cuboid<float> cbox;

        /** The file name of the pore network data file */
        core::param::ParamSlot filenameSlot;

        /** Saves the data to the pore network data file while extracting */
        core::param::ParamSlot streamSaveSlot;

        /** The extraction direction */
        core::param::ParamSlot extractDirSlot;

        /** The size of the extraction volume */
        core::param::ParamSlot extractSizeSlot;

        /** The size of a single rendering tile used for extraction */
        core::param::ParamSlot extractTileSizeSlot;

        /** Saves the pore network data to the data file */
        core::param::ParamSlot saveBtnSlot;

        /** Loads the pore network data from the data file */
        core::param::ParamSlot loadBtnSlot;

        /** Extractes the pore network data from the connected data modules */
        core::param::ParamSlot extractBtnSlot;

        /** The size of the extraction volume */
        unsigned int sizeX, sizeY, sizeZ;

        /** The extract direction */
        ExtractionDir edir;

        /** The file to be used for stream-saving */
        vislib::sys::File *saveFile;

        /** The rendering tile */
        vislib::graphics::gl::FramebufferObject *tile;

        /** The tile buffer */
        ArxelBuffer::ArxelType *tileBuffer;

        /** the rendered slices */
        BufferMTPConnection<ArxelBuffer> slicesBuffers;

        /** The extracted loops per slice */
        BufferMTPConnection<LoopBuffer> loopBuffers;

        /** The number of the slice to render next */
        unsigned int slice;

        /** The crystalite shader */
        vislib::graphics::gl::GLSLShader cryShader;

        /** The pore-net slice processor thread */
        vislib::sys::RunnableThread<PoreNetSliceProcessor> sliceProcessor;

        /** The pore-mesh connector/calculator/thingy thread */
        vislib::sys::RunnableThread<PoreMeshProcessor> meshProcessor;

        /** Ignore me, please */
        PoreMeshProcessor::SliceLoops debugLoopDataEntryObject;

    };

} /* end namespace demos */
} /* end namespace megamol */
