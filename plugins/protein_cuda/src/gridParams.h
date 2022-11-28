//
// gridParams.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Apr 18, 2013
//     Author: scharnkn
//

#ifndef MMPROTEINCUDAPLUGIN_GRIDPARAMS_H_INCLUDED
#define MMPROTEINCUDAPLUGIN_GRIDPARAMS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

struct gridParams {
    float minC[3];
    float maxC[3];
    float delta[3];
    unsigned int size[3];
};

#endif // MMPROTEINCUDAPLUGIN_GRIDPARAMS_H_INCLUDED
