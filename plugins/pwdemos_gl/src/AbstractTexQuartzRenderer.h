/*
 * AbstractTexQuartzRenderer.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "AbstractQuartzRenderer.h"
#include "QuartzCrystalDataCall.h"


namespace megamol {
namespace demos_gl {

/**
 * Module rendering gridded quarts particle data
 */
class AbstractTexQuartzRenderer : public AbstractQuartzRenderer {
public:
    /** Ctor */
    AbstractTexQuartzRenderer(void);

    /** Dtor */
    virtual ~AbstractTexQuartzRenderer(void);

protected:
    /**
     * Ensures the actuality of the type texture
     *
     * @param types The types
     */
    void assertTypeTexture(CrystalDataCall& types);

    /** Releases the type texture */
    void releaseTypeTexture(void);

    /** The type texture */
    unsigned int typeTexture;
};

} // namespace demos_gl
} /* end namespace megamol */
