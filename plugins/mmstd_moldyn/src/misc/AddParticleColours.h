/*
 * AddParticleColours.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ADDPARTICLECOLOURS_H_INCLUDED
#define MEGAMOLCORE_ADDPARTICLECOLOURS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Module.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "vislib/RawStorage.h"
#include "mmcore/view/CallGetTransferFunction.h"


namespace megamol {

namespace stdplugin {
namespace moldyn {
namespace misc {

/**
 * Renderer for gridded imposters
 */
class AddParticleColours : public core::Module {
public:
  /**
   * Answer the name of this module.
   *
   * @return The name of this module.
   */
  static const char *ClassName(void) { return "AddParticleColours"; }

  /**
   * Answer a human readable description of this module.
   *
   * @return A human readable description of this module.
   */
  static const char *Description(void) {
    return "Adds particle colours from a transfer function to the memory "
           "stored particles.";
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
  AddParticleColours(void);

  /** Dtor. */
  virtual ~AddParticleColours(void);

private:
  /**
   * Utility class used to unlock the additional colour data
   */
  class Unlocker : public core::moldyn::MultiParticleDataCall::Unlocker {
  public:
    /**
     * ctor.
     *
     * @param inner The inner unlocker object
     */
    Unlocker(core::moldyn::MultiParticleDataCall::Unlocker *inner);

    /** dtor. */
    virtual ~Unlocker(void);

    /** Unlocks the data */
    virtual void Unlock(void);

  private:
    /** the inner unlocker */
    core::moldyn::MultiParticleDataCall::Unlocker *inner;
  };

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
   * Gets the data from the source.
   *
   * @param caller The calling call.
   *
   * @return 'true' on success, 'false' on failure.
   */
  bool getDataCallback(core::Call &caller);

  /**
   * Gets the data from the source.
   *
   * @param caller The calling call.
   *
   * @return 'true' on success, 'false' on failure.
   */
  bool getExtentCallback(core::Call &caller);

  /** The call for the output data */
  core::CalleeSlot putDataSlot;

  /** The call for the input data */
  core::CallerSlot getDataSlot;

  /** The call for Transfer function */
  core::CallerSlot getTFSlot;

  /** Button to force rebuild of colour data */
  core::param::ParamSlot rebuildButtonSlot;

  /** The last frame */
  unsigned int lastFrame;

  /** The generated colour data */
  vislib::RawStorage colData;

  /** The colour data format */
  core::view::CallGetTransferFunction::TextureFormat colFormat;

  /** The update hash */
  vislib::RawStorage updateHash;
};

} // namespace misc
} // namespace moldyn
} // namespace stdplugin
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ADDPARTICLECOLOURS_H_INCLUDED */
