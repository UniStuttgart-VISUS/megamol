/*
 * CallFrame.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "CallFrame.h"

using namespace megamol::core;


/**
 * protein::CallFrame::CallFrame CTOR
 */
protein::CallFrame::CallFrame(void): m_frameID(0), m_newRequest(false)
{

}

/** 
 * protein::CallFrame::~CallFrame DTOR
 */
protein::CallFrame::~CallFrame(void)
{

}

/** 
 * protein::CallFrame::NewRequest
 */
bool protein::CallFrame::NewRequest(void)
{
    return this->m_newRequest;
}

/**
 * protein::CallFrame::SetFrameRequest
 */
void protein::CallFrame::SetFrameRequest(unsigned int frmID)
{
    this->m_newRequest = true;
    this->m_frameID = frmID;
}

/**
 * protein::CallFrame::GetFrameRequest
 */
unsigned int protein::CallFrame::GetFrameRequest(void)
{
    this->m_newRequest = false;
    return this->m_frameID;
}

