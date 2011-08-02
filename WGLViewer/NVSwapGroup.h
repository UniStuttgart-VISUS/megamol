/*
 * NVSwapGroup.h
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLVIEWER_NVSWAPGROUP_H_INCLUDED
#define MEGAMOLVIEWER_NVSWAPGROUP_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */



/**
    * Class to hold and apply the NVSwapGroup settings
    */
class NVSwapGroup {
public:

    /**
     * Returns the only singelton instance
     *
     * @return The only instance
     */
    static NVSwapGroup& Instance(void);

    bool JoinSwapGroup(unsigned int group);

    bool BindSwapBarrier(unsigned int group, unsigned int barrier);

    bool QuerySwapGroup(unsigned int &group, unsigned int &barrier);

    bool QueryMaxSwapGroups(unsigned int &maxGroups, unsigned int &maxBarriers);

    bool QueryFrameCount(unsigned int &count);

    bool ResetFrameCount(void);

private:

    /** Ctor */
    NVSwapGroup(void);

    /** Dtor */
    ~NVSwapGroup(void);

    /** initializes the extensions, if required */
    void assertExtensions(void);

    /** Flag whether or not the extensions have been initialized */
    bool extensionInitialized;

};


#endif /* MEGAMOLVIEWER_NVSWAPGROUP_H_INCLUDED */
