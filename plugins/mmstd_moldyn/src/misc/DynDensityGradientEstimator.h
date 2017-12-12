/*
 * DynDensityGradientEstimator.h
 *
 *  Created on: May 22, 2014
 *      Author: scharnkn@visus.uni-stuttgart.de
 */

#ifndef DYNDENSITYGRADIENTESTIMATOR_H_
#define DYNDENSITYGRADIENTESTIMATOR_H_

#include "mmcore/CallerSlot.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {

namespace stdplugin {
namespace moldyn {
namespace misc {

class DynDensityGradientEstimator : public core::Module {
public:
  /**
   * Answer the name of this module.
   *
   * @return The name of this module.
   */
  static const char *ClassName(void) { return "DynDensityGradientEstimator"; }

  /**
   * Answer a human readable description of this module.
   *
   * @return A human readable description of this module.
   */
  static const char *Description(void) { return "..."; }

  /**
   * Answers whether this module is available on the current system.
   *
   * @return 'true' if the module is available, 'false' otherwise.
   */
  static bool IsAvailable(void) { return true; }

  /** Ctor. */
  DynDensityGradientEstimator(void);

  /** Dtor. */
  virtual ~DynDensityGradientEstimator(void);

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

private:
  /**
   * Answer the extend of the data
   *
   * @param caller The calling CallVolumeData
   *
   * @return True on success
   */
  bool getExtent(core::Call &call);

  /**
   * Answer the data
   *
   * @param caller The calling CallVolumeData
   *
   * @return True on success
   */
  bool getData(core::Call &call);

  bool createVolumeCPU(class core::moldyn::MultiParticleDataCall &c2);

  core::CallerSlot getPartDataSlot;
  core::CalleeSlot putDirDataSlot;

  vislib::Array<float> dir, gridPos;
  float *dens;

  core::param::ParamSlot xResSlot;
  core::param::ParamSlot yResSlot;
  core::param::ParamSlot zResSlot;
};

} // namespace misc
} // namespace moldyn
} // namespace stdplugin
} // end namespace megamol
#endif /* DYNDENSITYGRADIENTESTIMATOR_H_ */
