/*
 * DirPartColModulate.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_DIRPARTCOLMODULATE_H_INCLUDED
#define MEGAMOLCORE_DIRPARTCOLMODULATE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/moldyn/DirectionalParticleDataCall.h"
#include "mmcore/CallVolumeData.h"
#include "vislib/RawStorage.h"
#include "vislib/types.h"


namespace megamol {

namespace stdplugin {
namespace moldyn {
namespace misc {

/**
 * Module to modulate the colour (saturation) of directional particles based on
 * a scalar volume.
 */
class DirPartColModulate : public core::Module {
public:
  /**
   * Answer the name of this module.
   *
   * @return The name of this module.
   */
  static const char *ClassName(void) { return "DirPartColModulate"; }

  /**
   * Answer a human readable description of this module.
   *
   * @return A human readable description of this module.
   */
  static const char *Description(void) {
    return "Module to modulate the colour (saturation) of directional "
           "particles based on a scalar volume";
  }

  /**
   * Answers whether this module is available on the current system.
   *
   * @return 'true' if the module is available, 'false' otherwise.
   */
  static bool IsAvailable(void) { return true; }

  /**
   * Disallow usage in quickstarts
   *
   * @return false
   */
  static bool SupportQuickstart(void) { return false; }

  /** Ctor. */
  DirPartColModulate(void);

  /** Dtor. */
  virtual ~DirPartColModulate(void);

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
   * Callback publishing the gridded data
   *
   * @param call The call requesting the gridded data
   *
   * @return 'true' on success, 'false' on failure
   */
  bool getData(core::Call &call);

  /**
   * Callback publishing the extend of the data
   *
   * @param call The call requesting the extend of the data
   *
   * @return 'true' on success, 'false' on failure
   */
  bool getExtend(core::Call &call);

  core::CallerSlot inParticlesDataSlot;

  core::CallerSlot inVolumeDataSlot;

  core::CalleeSlot outDataSlot;

  core::param::ParamSlot attributeSlot;

  core::param::ParamSlot attrMinValSlot;

  core::param::ParamSlot attrMaxValSlot;

  core::param::ParamSlot baseColourSlot;

  SIZE_T datahashOut;

  SIZE_T datahashParticlesIn;

  SIZE_T datahashVolumeIn;

  unsigned int frameID;

  vislib::RawStorage colData;
};

} // namespace misc
} // namespace moldyn
} // namespace stdplugin
} /* end namespace megamol */

#endif /* MEGAMOLCORE_DIRPARTCOLMODULATE_H_INCLUDED */
