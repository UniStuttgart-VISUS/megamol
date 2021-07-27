/*
 * ParticleThermodyn.h
 *
 * Copyright (C) 2017 by MegaMol team
 * Alle Rechte vorbehalten.
 */

#ifndef MMSTD_DATATOOLS_PARTICLETHERMOMETER_H_INCLUDED
#define MMSTD_DATATOOLS_PARTICLETHERMOMETER_H_INCLUDED
#pragma once

#include <Eigen/Eigenvalues>
#include <nanoflann.hpp>
#include <vector>
#include "mmstd_datatools/PointcloudHelpers.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace stdplugin {
    namespace datatools {

        /**
         * Module overriding global attributes of particles
         */
        class ParticleThermodyn : public megamol::core::Module {
        public:
            enum searchTypeEnum { RADIUS, NUM_NEIGHBORS };

            enum metricsEnum {
                TEMPERATURE,
                FRACTIONAL_ANISOTROPY,
                DENSITY,
                PRESSURE,
                NEIGHBORS,
                NEAREST_DISTANCE,
                PHASE01,
                PHASE02
            };

            enum phaseEnum { FLUID = 0, GAS = 1 };

            /** Return module class name */
            static const char* ClassName(void) {
                return "ParticleThermodyn";
            }

            /** Return module class description */
            static const char* Description(void) {
                return "Computes an intensity from some properties of a particle (compared to its surroundings).";
            }

            /** Module is always available */
            static bool IsAvailable(void) {
                return true;
            }

            /** Ctor */
            ParticleThermodyn(void);

            /** Dtor */
            virtual ~ParticleThermodyn(void);

            /**
             * Called when the data is requested by this module
             *
             * @param c The incoming call
             *
             * @return True on success
             */
            bool getDataCallback(megamol::core::Call& c);

            /**
             * Called when the extend information is requested by this module
             *
             * @param c The incoming call
             *
             * @return True on success
             */
            bool getExtentCallback(megamol::core::Call& c);

        protected:
            /** Lazy initialization of the module */
            virtual bool create(void);

            /** Resource release */
            virtual void release(void);

        private:
            bool isDirty() const {
                return this->radiusSlot.IsDirty() || this->cyclXSlot.IsDirty() || this->cyclYSlot.IsDirty() ||
                       this->cyclZSlot.IsDirty() || this->numNeighborSlot.IsDirty() || this->searchTypeSlot.IsDirty() ||
                       this->metricsSlot.IsDirty() || this->removeSelfSlot.IsDirty() ||
                       this->findExtremesSlot.IsDirty() || this->extremeValueSlot.IsDirty() ||
                       this->fluidDensitySlot.IsDirty();
            }

            void resetDirty() {
                this->radiusSlot.ResetDirty();
                this->cyclXSlot.ResetDirty();
                this->cyclYSlot.ResetDirty();
                this->cyclZSlot.ResetDirty();
                this->numNeighborSlot.ResetDirty();
                this->searchTypeSlot.ResetDirty();
                this->metricsSlot.ResetDirty();
                this->removeSelfSlot.ResetDirty();
                this->findExtremesSlot.ResetDirty();
                this->extremeValueSlot.ResetDirty();
                this->fluidDensitySlot.ResetDirty();
            }

            bool assertData(core::moldyn::MultiParticleDataCall* in, core::moldyn::MultiParticleDataCall* outMPDC);

            bool computeCurrentFrame(unsigned int frameID);

            float computeTemperature(
                std::vector<std::pair<size_t, float>>& matches, size_t num_matches, float mass, float freedom);
            float computeFractionalAnisotropy(std::vector<std::pair<size_t, float>>& matches, size_t num_matches);
            float computeDensity(std::vector<std::pair<size_t, float>>& matches, size_t num_matches,
                float const curPoint[3], float radius, vislib::math::Cuboid<float> const& bbox);

            core::param::ParamSlot cyclXSlot;
            core::param::ParamSlot cyclYSlot;
            core::param::ParamSlot cyclZSlot;
            core::param::ParamSlot radiusSlot;
            core::param::ParamSlot numNeighborSlot;
            core::param::ParamSlot searchTypeSlot;
            core::param::ParamSlot minMetricSlot;
            core::param::ParamSlot maxMetricSlot;
            core::param::ParamSlot massSlot;
            core::param::ParamSlot freedomSlot;
            core::param::ParamSlot metricsSlot;
            core::param::ParamSlot removeSelfSlot;
            core::param::ParamSlot findExtremesSlot;
            core::param::ParamSlot extremeValueSlot;
            core::param::ParamSlot fluidDensitySlot;
            core::param::ParamSlot tcSlot;
            core::param::ParamSlot rhocSlot;
            core::param::ParamSlot timeSmoothSlot;

            size_t datahash;
            size_t myHash = 0;
            int lastTime;
            std::vector<float> newColors;
            std::vector<size_t> allParts;
            float maxDist;

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver;

            typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, simplePointcloud>,
                simplePointcloud, 3 /* dim */
                >
                my_kd_tree_t;

            std::unique_ptr<my_kd_tree_t> particleTree;
            simplePointcloud myPts;

            /** The slot providing access to the manipulated data */
            megamol::core::CalleeSlot outDataSlot;

            /** The slot accessing the original data */
            megamol::core::CallerSlot inDataSlot;
        };

    } /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MMSTD_DATATOOLS_PARTICLETHERMOMETER_H_INCLUDED */
