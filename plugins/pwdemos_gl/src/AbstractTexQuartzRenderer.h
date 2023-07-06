/*
 * AbstractTexQuartzRenderer.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "AbstractQuartzRenderer.h"
#include "QuartzCrystalDataCall.h"


namespace megamol::demos_gl {

/**
 * Module rendering gridded quarts particle data
 */
class AbstractTexQuartzRenderer : public AbstractQuartzRenderer {
public:
    /** Ctor */
    AbstractTexQuartzRenderer();

    /** Dtor */
    ~AbstractTexQuartzRenderer() override;

protected:
    /**
     * Ensures the actuality of the type texture
     *
     * @param types The types
     */
    void assertTypeTexture(CrystalDataCall& types);

    /** Releases the type texture */
    void releaseTypeTexture();

    /** The type texture */
    unsigned int typeTexture;
};

} // namespace megamol::demos_gl
