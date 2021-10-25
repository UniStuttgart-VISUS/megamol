#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/moldyn/EllipsoidalDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_datatools/table/TableDataCall.h"
#include <map>
#include <array>

namespace megamol {
namespace stdplugin {
namespace datatools {

    /**
     * This module converts from a generic table to the MultiParticleDataCall.
     */
    class ParticlesToTable : public megamol::core::Module {

    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static inline const char *ClassName(void)  {
            return "ParticlesToTable";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static inline const char *Description(void) {
            return "Converts particles to generic tables.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static inline bool IsAvailable(void) {
            return true;
        }

        /**
         * Initialises a new instance.
         */
        ParticlesToTable(void);

        /**
         * Finalises an instance.
         */
        virtual ~ParticlesToTable(void);

    protected:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        bool getTableData(core::Call& call);

        bool getTableHash(core::Call& call);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

    private:

        bool assertMPDC(core::moldyn::MultiParticleDataCall *in, table::TableDataCall *tc);
        bool assertEPDC(core::moldyn::EllipsoidalParticleDataCall *c, table::TableDataCall *tc);


        //std::string cleanUpColumnHeader(const std::string& header) const;
        //std::string cleanUpColumnHeader(const vislib::TString& header) const;

        //bool pushColumnIndex(std::vector<uint32_t>& cols, const vislib::TString& colName);

        ///** Minimum coordinates of the bounding box. */
        //float bboxMin[3];

        ///** Maximum coordinates of the bounding box. */
        //float bboxMax[3];

        //float iMin, iMax;
        //
        /** The slot for retrieving the data as multi particle data. */
        core::CalleeSlot slotTableOut;

        /** The data callee slot. */
        core::CallerSlot slotParticlesIn;

        ///** The name of the float column holding the red colour channel. */
        //core::param::ParamSlot slotColumnB;

        ///** The name of the float column holding the green colour channel. */
        //core::param::ParamSlot slotColumnG;

        ///** The name of the float column holding the blue colour channel. */
        //core::param::ParamSlot slotColumnR;
        //    
        ///** The name of the float column holding the intensity channel. */
        //core::param::ParamSlot slotColumnI;
        //
        ///**
        //* The constant color of spheres if the data set does not provide
        //* one.
        //*/
        //core::param::ParamSlot slotGlobalColor;

        ///** The color mode: explicit rgb, intensity or constant */
        //core::param::ParamSlot slotColorMode;

        ///** The name of the float column holding the particle radius. */
        //core::param::ParamSlot slotColumnRadius;

        ///**
        //* The constant radius of spheres if the data set does not provide
        //* one.
        //*/
        //core::param::ParamSlot slotGlobalRadius;

        ///** The color mode: explicit rgb, intensity or constant */
        //core::param::ParamSlot slotRadiusMode;

        ///** The name of the float column holding the x-coordinate. */
        //core::param::ParamSlot slotColumnX;

        ///** The name of the float column holding the y-coordinate. */
        //core::param::ParamSlot slotColumnY;

        ///** The name of the float column holding the z-coordinate. */
        //core::param::ParamSlot slotColumnZ;

        ///** The name of the float column holding the vx-coordinate. */
        //core::param::ParamSlot slotColumnVX;

        ///** The name of the float column holding the vy-coordinate. */
        //core::param::ParamSlot slotColumnVY;

        ///** The name of the float column holding the vz-coordinate. */
        //core::param::ParamSlot slotColumnVZ;

        ///** given a tensor interpretable as an orthonormal coordinate system {X,Y,Z}^T, its 9 values {xx, xy, xz, ...} */
        //std::array<core::param::ParamSlot, 9> slotTensorColumn;

        ///** if the tensor contains normalized vectors, you can also supply a magnitude */
        //std::array<core::param::ParamSlot, 3> slotTensorMagnitudeColumn;

        /** The lastly calculated time step */
        unsigned int lastTimeStep;

        std::vector<float> everything;

        bool haveVelocities = false;
        bool haveTensor = false;
        bool haveTensorMagnitudes = false;

        SIZE_T inHash = SIZE_MAX;
        unsigned int inFrameID = std::numeric_limits<unsigned int>::max();
        std::vector<table::TableDataCall::ColumnInfo> column_infos;
        uint32_t total_particles = 0;

        SIZE_T myHash;
        SIZE_T myTime = -1;
        uint32_t stride;

    };

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */
