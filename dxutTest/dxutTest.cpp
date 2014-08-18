s//--------------------------------------------------------------------------------------
// File: SimpleSample.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#include "DXUT.h"
#include "DXUTgui.h"
#include "DXUTmisc.h"
#include "DXUTCamera.h"
#include "DXUTSettingsDlg.h"
#include "SDKmisc.h"
#include "SDKmesh.h"

#include "AbstractTest.h"
#include "D3D9SimpleCameraTest.h"
#include "D3D9VisLogoTest.h"
#include "TestManager.h"


//#define DEBUG_VS   // Uncomment this line to debug D3D9 vertex shaders 
//#define DEBUG_PS   // Uncomment this line to debug D3D9 pixel shaders 


//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------
CModelViewerCamera      g_Camera;               // A model viewing camera
CDXUTDialogResourceManager dlgResMgr; // manager for shared resources of dialogs
CD3DSettingsDlg         dlg;          // Device settings dialog
CDXUTTextHelper*		g_pTxtHelper = NULL;
CDXUTDialog             g_HUD;                  // dialog for standard controls
CDXUTDialog             g_SampleUI;             // dialog for sample specific controls

// Direct3D 9 resources
ID3DXFont *font = NULL;        
ID3DXSprite*            g_pSprite9 = NULL;

TestManager testMgr;


//--------------------------------------------------------------------------------------
// UI control IDs
//--------------------------------------------------------------------------------------
#define IDC_TOGGLEFULLSCREEN    1
#define IDC_TOGGLEREF           2
#define IDC_CHANGEDEVICE        3
#define IDC_CHANGETEST (4)


//--------------------------------------------------------------------------------------
// Forward declarations 
//--------------------------------------------------------------------------------------
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing, void* pUserContext );
void    CALLBACK OnKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext );
void    CALLBACK OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext );
void    CALLBACK OnFrameMove( double fTime, float fElapsedTime, void* pUserContext );
bool    CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext );
void CALLBACK OnMouse(bool bLeftButtonDown, bool bRightButtonDown, 
        bool bMiddleButtonDown, bool bSideButton1Down, bool bSideButton2Down,
        INT nMouseWheelDelta, INT xPos, INT yPos, void *pUserContext);

bool    CALLBACK IsD3D9DeviceAcceptable( D3DCAPS9* pCaps, D3DFORMAT AdapterFormat, D3DFORMAT BackBufferFormat, bool bWindowed, void* pUserContext );
HRESULT CALLBACK OnD3D9CreateDevice( IDirect3DDevice9* pd3dDevice, const D3DSURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext );
HRESULT CALLBACK OnD3D9ResetDevice( IDirect3DDevice9* pd3dDevice, const D3DSURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext );
void    CALLBACK OnD3D9FrameRender( IDirect3DDevice9* pd3dDevice, double fTime, float fElapsedTime, void* pUserContext );
void    CALLBACK OnD3D9LostDevice( void* pUserContext );
void    CALLBACK OnD3D9DestroyDevice( void* pUserContext );

void    InitApp();
void    RenderText();


/**
 * Create any D3D9 resources that will live through a device reset 
 * (D3DPOOL_MANAGED) and aren't tied to the back buffer size
 */
HRESULT CALLBACK OnD3D9CreateDevice(PDIRECT3DDEVICE9 pd3dDevice, 
        const D3DSURFACE_DESC *pBackBufferSurfaceDesc, void *pUserContext) {
    TestManager *testMgr = static_cast<TestManager *>(pUserContext);
    HRESULT hr = D3D_OK;

    V_RETURN(::dlgResMgr.OnD3D9CreateDevice(pd3dDevice));
    V_RETURN(::dlg.OnD3D9CreateDevice(pd3dDevice));
    
    V_RETURN(::D3DXCreateFont(pd3dDevice, 15, 0, FW_BOLD, 1, FALSE, 
        DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, DEFAULT_QUALITY, 
        DEFAULT_PITCH | FF_DONTCARE, L"Verdana", &::font));

    testMgr->OnD3D9CreateDevice(pd3dDevice, pBackBufferSurfaceDesc);

    return hr;
}


/**
 * Release D3D9 resources created in the OnD3D9CreateDevice callback.
 */
void CALLBACK OnD3D9DestroyDevice(void *pUserContext) {
    TestManager *testMgr = static_cast<TestManager *>(pUserContext);
    testMgr->OnD3D9DestroyDevice();
    ::dlgResMgr.OnD3D9DestroyDevice();
    ::dlg.OnD3D9DestroyDevice();
    SAFE_RELEASE(::font);
}


/**
 * Render the scene using the D3D9 device
 */
void CALLBACK OnD3D9FrameRender(PDIRECT3DDEVICE9 pd3dDevice, double fTime,
        float fElapsedTime, void *pUserContext) {
    TestManager *testMgr = static_cast<TestManager *>(pUserContext);
    HRESULT hr = D3D_OK;
    //D3DXMATRIXA16 mWorld;
    //D3DXMATRIXA16 mView;
    //D3DXMATRIXA16 mProj;
    //D3DXMATRIXA16 mWorldViewProjection;
    
    // If the settings dialog is being shown, then render it instead of rendering the app's scene
    if (::dlg.IsActive()) {
        ::dlg.OnRender(fElapsedTime);
        return;
    }

    // Clear the render target and the zbuffer 
    V( pd3dDevice->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, D3DCOLOR_ARGB(0, 45, 50, 170), 1.0f, 0) );

    // Render the scene
    if (SUCCEEDED(pd3dDevice->BeginScene())) {
        // Get the projection & view matrix from the camera class
        //mWorld = *g_Camera.GetWorldMatrix();
        //mProj = *g_Camera.GetProjMatrix();
        //mView = *g_Camera.GetViewMatrix();

        //mWorldViewProjection = mWorld * mView * mProj;

        //pd3dDevice->SetTransform(D3DTS_WORLD, &mWorld);
        //pd3dDevice->SetTransform(D3DTS_PROJECTION, &mProj);
        //pd3dDevice->SetTransform(D3DTS_VIEW, &mView);

        testMgr->OnD3D9FrameRender(pd3dDevice, fTime, fElapsedTime);

        // Update the effect's variables.  Instead of using strings, it would 
        // be more efficient to cache a handle to the parameter by calling 
        // ID3DXEffect::GetParameterByName

        DXUT_BeginPerfEvent( DXUT_PERFEVENTCOLOR, L"HUD / Stats" ); // These events are to help PIX identify what the code is doing
        RenderText();
        V( g_HUD.OnRender( fElapsedTime ) );
        V( g_SampleUI.OnRender( fElapsedTime ) );
        DXUT_EndPerfEvent();

        V( pd3dDevice->EndScene() );
    }
}


/**
 * Release D3D9 resources created in the OnD3D9ResetDevice callback.
 */
void CALLBACK OnD3D9LostDevice(void *pUserContext) {
    TestManager *testMgr = static_cast<TestManager *>(pUserContext);
    testMgr->OnD3D9LostDevice();

    ::dlgResMgr.OnD3D9LostDevice();
    ::dlg.OnD3D9LostDevice();
    if (::font) {
        ::font->OnLostDevice();
    }

    SAFE_RELEASE( g_pSprite9 );
    SAFE_DELETE( g_pTxtHelper );
}



/**
 * Handle updates to the scene.  
 * This is called regardless of which D3D API is used.
 */
void CALLBACK OnFrameMove(double fTime, float fElapsedTime,
        void *pUserContext) {
    TestManager *testMgr = static_cast<TestManager *>(pUserContext);
    testMgr->OnFrameMove(fTime, fElapsedTime);
}


/**
 * Handles the GUI events.
 */
void CALLBACK OnGUIEvent(UINT nEvent, int nControlID, CDXUTControl *pControl, 
        void *pUserContext) {
    switch (nControlID) {
        case IDC_TOGGLEFULLSCREEN: DXUTToggleFullScreen(); break;
        case IDC_TOGGLEREF:        DXUTToggleREF(); break;
        case IDC_CHANGEDEVICE:     ::dlg.SetActive( !::dlg.IsActive() ); break;
        case IDC_CHANGETEST: {
            CDXUTComboBox *comboBox = dynamic_cast<CDXUTComboBox *>(pControl);
            ::testMgr.SetActiveTest(comboBox->GetSelectedIndex());
            // TODO: User ctx does not work ... somehow ...
            //::testMgr.SetActiveTest(reinterpret_cast<SIZE_T>(pUserContext));
            } break;
    }
}


/**
 * Handle key presses.
 */
void CALLBACK OnKeyboard(UINT nChar, bool bKeyDown, bool bAltDown,
        void *pUserContext) {
    TestManager *testMgr = static_cast<TestManager *>(pUserContext);
    testMgr->OnKeyboard(nChar, bKeyDown, bAltDown);
}


/** 
 * Handle mouse events.
 */
void CALLBACK OnMouse(bool bLeftButtonDown, bool bRightButtonDown, 
        bool bMiddleButtonDown, bool bSideButton1Down, bool bSideButton2Down,
        INT nMouseWheelDelta, INT xPos, INT yPos, void *pUserContext) {
    TestManager *testMgr = static_cast<TestManager *>(pUserContext);
    testMgr->OnMouse(bLeftButtonDown, bRightButtonDown, bMiddleButtonDown,
        bSideButton1Down, bSideButton2Down, nMouseWheelDelta, xPos, yPos);
}


/**
 * Entry point to the program. Initializes everything and goes into a message 
 * processing loop. Idle time is used to render the scene.
 */
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, 
        LPWSTR lpCmdLine, int nCmdShow) {
    // Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
    _CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
#endif

    // DXUT will create and use the best device (either D3D9 or D3D10) 
    // that is available on the system depending on which D3D callbacks are set below


    // Set DXUT callbacks
    ::DXUTSetCallbackMsgProc(MsgProc);
    ::DXUTSetCallbackKeyboard(OnKeyboard, &::testMgr);
    ::DXUTSetCallbackMouse(OnMouse, TRUE, &::testMgr);
    ::DXUTSetCallbackFrameMove(OnFrameMove, &::testMgr);
    ::DXUTSetCallbackDeviceChanging(ModifyDeviceSettings);

    ::DXUTSetCallbackD3D9DeviceAcceptable(IsD3D9DeviceAcceptable);
    ::DXUTSetCallbackD3D9DeviceCreated(OnD3D9CreateDevice, &::testMgr);
    ::DXUTSetCallbackD3D9DeviceReset(OnD3D9ResetDevice, &::testMgr);
    ::DXUTSetCallbackD3D9DeviceLost(OnD3D9LostDevice, &::testMgr);
    ::DXUTSetCallbackD3D9DeviceDestroyed(OnD3D9DestroyDevice, &::testMgr);
    ::DXUTSetCallbackD3D9FrameRender(OnD3D9FrameRender, &::testMgr);

    ::testMgr.AddTest(new D3D9VisLogoTest());
    ::testMgr.AddTest(new D3D9SimpleCameraTest());
    ::testMgr.SetActiveTest(0);

    ::InitApp();
    ::DXUTInit(true, true, NULL); // Parse the command line, show msgboxes on error, no extra command line params
    ::DXUTSetCursorSettings(true, true);
    ::DXUTCreateWindow(L"VISlib Direct3D Tests");
    ::DXUTCreateDevice(true, 640, 480);

    ::DXUTMainLoop();

    return ::DXUTGetExitCode();
}


//--------------------------------------------------------------------------------------
// Initialize the app 
//--------------------------------------------------------------------------------------
void InitApp()
{
    ::dlg.Init(&::dlgResMgr);
    g_HUD.Init( &::dlgResMgr );
    g_SampleUI.Init(&::dlgResMgr);

    g_HUD.SetCallback(::OnGUIEvent); int iY = 10; 
    g_HUD.AddButton(IDC_TOGGLEFULLSCREEN, L"Toggle full screen", 10, iY, 150, 22);
    g_HUD.AddButton(IDC_TOGGLEREF, L"Toggle REF (F3)", 10, iY += 24, 150, 22, VK_F3);
    g_HUD.AddButton(IDC_CHANGEDEVICE, L"Change device (F2)", 10, iY += 24, 150, 22, VK_F2);

    CDXUTComboBox *cb = NULL;
    g_HUD.AddComboBox(IDC_CHANGETEST, 10, iY += 24, 150, 22, 0, false, &cb);
    ::testMgr.RegisterTests(cb);

    g_SampleUI.SetCallback( OnGUIEvent ); iY = 10; 
}


//--------------------------------------------------------------------------------------
// Render the help and statistics text. This function uses the ID3DXFont interface for 
// efficient text rendering.
//--------------------------------------------------------------------------------------
void RenderText()
{
    g_pTxtHelper->Begin();
    g_pTxtHelper->SetInsertionPos( 5, 5 );
    g_pTxtHelper->SetForegroundColor( D3DXCOLOR( 1.0f, 1.0f, 0.0f, 1.0f ) );
    g_pTxtHelper->DrawTextLine( DXUTGetFrameStats( DXUTIsVsyncEnabled() ) );  
    g_pTxtHelper->DrawTextLine( DXUTGetDeviceStats() );
    g_pTxtHelper->End();
}


//--------------------------------------------------------------------------------------
// Rejects any D3D9 devices that aren't acceptable to the app by returning false
//--------------------------------------------------------------------------------------
bool CALLBACK IsD3D9DeviceAcceptable( D3DCAPS9* pCaps, D3DFORMAT AdapterFormat, 
                                      D3DFORMAT BackBufferFormat, bool bWindowed, void* pUserContext )
{
    // Skip backbuffer formats that don't support alpha blending
    IDirect3D9* pD3D = DXUTGetD3D9Object(); 
    if( FAILED( pD3D->CheckDeviceFormat( pCaps->AdapterOrdinal, pCaps->DeviceType,
                    AdapterFormat, D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING, 
                    D3DRTYPE_TEXTURE, BackBufferFormat ) ) )
        return false;

    // No fallback defined by this app, so reject any device that 
    // doesn't support at least ps2.0
    if( pCaps->PixelShaderVersion < D3DPS_VERSION(2,0) )
        return false;

    return true;
}


//--------------------------------------------------------------------------------------
// Called right before creating a D3D9 or D3D10 device, allowing the app to modify the device settings as needed
//--------------------------------------------------------------------------------------
bool CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext )
{
    if( pDeviceSettings->ver == DXUT_D3D9_DEVICE )
    {
        IDirect3D9 *pD3D = DXUTGetD3D9Object();
        D3DCAPS9 Caps;
        pD3D->GetDeviceCaps( pDeviceSettings->d3d9.AdapterOrdinal, pDeviceSettings->d3d9.DeviceType, &Caps );

        // If device doesn't support HW T&L or doesn't support 1.1 vertex shaders in HW 
        // then switch to SWVP.
        if( (Caps.DevCaps & D3DDEVCAPS_HWTRANSFORMANDLIGHT) == 0 ||
            Caps.VertexShaderVersion < D3DVS_VERSION(1,1) )
        {
            pDeviceSettings->d3d9.BehaviorFlags = D3DCREATE_SOFTWARE_VERTEXPROCESSING;
        }

        // Debugging vertex shaders requires either REF or software vertex processing 
        // and debugging pixel shaders requires REF.  
#ifdef DEBUG_VS
        if( pDeviceSettings->d3d9.DeviceType != D3DDEVTYPE_REF )
        {
            pDeviceSettings->d3d9.BehaviorFlags &= ~D3DCREATE_HARDWARE_VERTEXPROCESSING;
            pDeviceSettings->d3d9.BehaviorFlags &= ~D3DCREATE_PUREDEVICE;                            
            pDeviceSettings->d3d9.BehaviorFlags |= D3DCREATE_SOFTWARE_VERTEXPROCESSING;
        }
#endif
#ifdef DEBUG_PS
        pDeviceSettings->d3d9.DeviceType = D3DDEVTYPE_REF;
#endif
    }

    // For the first device created if its a REF device, optionally display a warning dialog box
    static bool s_bFirstTime = true;
    if( s_bFirstTime )
    {
        s_bFirstTime = false;
        if( (DXUT_D3D9_DEVICE == pDeviceSettings->ver && pDeviceSettings->d3d9.DeviceType == D3DDEVTYPE_REF) ||
            (DXUT_D3D10_DEVICE == pDeviceSettings->ver && pDeviceSettings->d3d10.DriverType == D3D10_DRIVER_TYPE_REFERENCE) )
            DXUTDisplaySwitchingToREFWarning( pDeviceSettings->ver );
    }

    return true;
}





//--------------------------------------------------------------------------------------
// Create any D3D9 resources that won't live through a device reset (D3DPOOL_DEFAULT) 
// or that are tied to the back buffer size 
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D9ResetDevice( IDirect3DDevice9* pd3dDevice, 
                                    const D3DSURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext )
{
    HRESULT hr;

    V_RETURN( ::dlgResMgr.OnD3D9ResetDevice() );
    V_RETURN( ::dlg.OnD3D9ResetDevice() );

    if( ::font ) V_RETURN( ::font->OnResetDevice() );

    V_RETURN( D3DXCreateSprite( pd3dDevice, &g_pSprite9 ) );
    g_pTxtHelper = new CDXUTTextHelper( ::font, g_pSprite9, NULL, NULL, 15 );

    // Setup the camera's projection parameters
    float fAspectRatio = pBackBufferSurfaceDesc->Width / (FLOAT)pBackBufferSurfaceDesc->Height;
    g_Camera.SetProjParams( D3DX_PI/4, fAspectRatio, 0.1f, 1000.0f );
    g_Camera.SetWindow( pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height );

    g_HUD.SetLocation( pBackBufferSurfaceDesc->Width-170, 0 );
    g_HUD.SetSize( 170, 170 );
    g_SampleUI.SetLocation( pBackBufferSurfaceDesc->Width-170, pBackBufferSurfaceDesc->Height-350 );
    g_SampleUI.SetSize( 170, 300 );

    return S_OK;
}








//--------------------------------------------------------------------------------------
// Handle messages to the application
//--------------------------------------------------------------------------------------
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing, void* pUserContext )
{
    // Pass messages to dialog resource manager calls so GUI state is updated correctly
    *pbNoFurtherProcessing = ::dlgResMgr.MsgProc( hWnd, uMsg, wParam, lParam );
    if( *pbNoFurtherProcessing )
        return 0;

    // Pass messages to settings dialog if its active
    if( ::dlg.IsActive() )
    {
        ::dlg.MsgProc( hWnd, uMsg, wParam, lParam );
        return 0;
    }

    // Give the dialogs a chance to handle the message first
    *pbNoFurtherProcessing = g_HUD.MsgProc( hWnd, uMsg, wParam, lParam );
    if( *pbNoFurtherProcessing )
        return 0;
    *pbNoFurtherProcessing = g_SampleUI.MsgProc( hWnd, uMsg, wParam, lParam );
    if( *pbNoFurtherProcessing )
        return 0;

    // Pass all remaining windows messages to camera so it can respond to user input
    g_Camera.HandleMessages( hWnd, uMsg, wParam, lParam );

    return 0;
}




