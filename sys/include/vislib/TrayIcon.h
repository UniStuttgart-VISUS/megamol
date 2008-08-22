/*
 * TrayIcon.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_TRAYICON_H_INCLUDED
#define VISLIB_TRAYICON_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#ifdef _WIN32

// Note: Must set _WIN32_IE before including shell api as it goes mad
// otherwise defining it itself in a very uncontrollen manner
#ifndef _WIN32_IE
#define _WIN32_IE 0x0500
#endif /* !_WIN32_IE */

#include <shlwapi.h>

// Must have NIIF_NODE for default parameters, even if _WIN32_IE is before
// Windows 2000.
#ifndef NIIF_NONE
#define NIIF_NONE 0
#endif /* !NIIF_NONE */


namespace vislib {
namespace sys {

    /**
     * This class encapsulates the functionality to create a Windows systray 
     * icon. It can be used in two ways: If a derived class is created, events
     * can be processed via overriding the onNotify method. If the class is
     * used directly, the user can specify a target window to receive 
     * notification messages from the tray icon.
     *
     * Note: You must have a message loop in order to get notifications from
     * the tray icon. A console application without such a message loop cannot
     * dispatch notification messages.
     */
    class TrayIcon {

    public:

        /** The possible states of the tray icon. */
        enum IconState {
            ICON_NOT_INSTALLED = 0,     // Icon has not been added.
            ICON_HIDDEN,                // Icon is added, but hidden.
            ICON_VISIBLE                // Icon is visible.
        };

        /**
         * Create a new instance using the default instance handle returned by
         * GetModuleHandle(NULL).
         */
        TrayIcon(void);

        /**
         * Create a new instance using the specified instance handle.
         *
         * @param hInstance The instance handle.
         */
        TrayIcon(HINSTANCE hInstance);

        /**
         * Dtor.
         *
         * The dtor deletes the tray icon, if not yet deleted, and destroys the 
         * hidden control window, if one has been created.
         */
        virtual ~TrayIcon(void);

        /**
         * Creates a new tray icon. If this tray icon has already been created,
         * it is destroyed and recreated using the specified parameters.
         *
         * @param targetWnd       Handle to the window that receives 
         *                        notification messages associated with an icon 
         *                        in the taskbar status area. The Shell uses 
         *                        'targetWnd' and 'id' to identify which icon 
         *                        to operate on when Shell_NotifyIcon is 
         *                        invoked.
         *                        If 'targetWnd' is NULL, the object will 
         *                        created an own hidden window to handle
         *                        messages.
         * @param callbackMessage Application-defined message identifier. The 
         *                        system uses this identifier to send 
         *                        notifications to the window identified in 
         *                        'targetWnd'. These notifications are sent 
         *                        when a mouse event occurs in the bounding 
         *                        rectangle of the icon, or when the icon is 
         *                        selected or activated with the keyboard. 
         *                        The wParam parameter of the message contains
         *                        the identifier of the taskbar icon in which 
         *                        the event occurred. The lParam parameter holds
         *                        the mouse or keyboard message associated with 
         *                        the event.
         * @param id              Application-defined identifier of the taskbar
         *                        icon. The Shell uses 'targetWnd' and 'id' to 
         *                        identify which icon to operate on when 
         *                        Shell_NotifyIcon is invoked. You can have 
         *                        multiple icons associated with a single 
         *                        'targetWnd' by assigning each a different 
         *                        'id'.
         * @param toolTip         Pointer to a null-terminated string with the 
         *                        text for a standard ToolTip. It can have a 
         *                        maximum of 64 characters including the 
         *                        terminating NULL.
         * @param icon            Handle to the icon to be added, modified, 
         *                        or deleted. 
         * @param initiallyHidden Controls whether the tray icon is shown or
         *                        initially hidden when the method returns
         *                        successfully.
         * @param balloonText     Pointer to a null-terminated string with the 
         *                        text for a balloon ToolTip. It can have a 
         *                        maximum of 256 characters including the
         *                        terminating NULL. 
         * @param balloonTitle    Pointer to a null-terminated string 
         *                        containing a title for a balloon ToolTip. 
         *                        This title appears in boldface above the text.
         *                        It can have a maximum of 64 characters 
         *                        including the terminating NULL.
         * @param balloonIcon     The icon of the balloon ToolTip. This must be
         *                        one of NIIF_NONE, NIIF_INFO, NIIF_WARNING or 
         *                        NIIF_ERROR.
         * @param balloonTimeout  The timeout value, in milliseconds, for a 
         *                        balloon ToolTip. The system minimum and 
         *                        maximum timeout values are currently set 
         *                        at 10 seconds and 30 seconds, respectively.
         *
         * @throws SystemException If the tray icon could not be created.
         */
        void Create(HWND targetWnd, const UINT callbackMessage, const UINT id,
            const wchar_t *toolTip, const HICON icon,
            const bool initiallyHidden = false, 
            const wchar_t *balloonText = NULL, 
            const wchar_t *balloonTitle = NULL, 
            const DWORD balloonIcon = NIIF_NONE, 
            const UINT balloonTimeout = 10);

        /**
         * Creates a new tray icon. If this tray icon has already been created,
         * it is destroyed and recreated using the specified parameters.
         *
         * @param targetWnd       Handle to the window that receives 
         *                        notification messages associated with an icon 
         *                        in the taskbar status area. The Shell uses 
         *                        'targetWnd' and 'id' to identify which icon 
         *                        to operate on when Shell_NotifyIcon is 
         *                        invoked.
         *                        If 'targetWnd' is NULL, the object will 
         *                        created an own hidden window to handle
         *                        messages.
         * @param callbackMessage Application-defined message identifier. The 
         *                        system uses this identifier to send 
         *                        notifications to the window identified in 
         *                        'targetWnd'. These notifications are sent 
         *                        when a mouse event occurs in the bounding 
         *                        rectangle of the icon, or when the icon is 
         *                        selected or activated with the keyboard. 
         *                        The wParam parameter of the message contains
         *                        the identifier of the taskbar icon in which 
         *                        the event occurred. The lParam parameter holds
         *                        the mouse or keyboard message associated with 
         *                        the event.
         * @param id              Application-defined identifier of the taskbar
         *                        icon. The Shell uses 'targetWnd' and 'id' to 
         *                        identify which icon to operate on when 
         *                        Shell_NotifyIcon is invoked. You can have 
         *                        multiple icons associated with a single 
         *                        'targetWnd' by assigning each a different 
         *                        'id'.
         * @param toolTip         Pointer to a null-terminated string with the 
         *                        text for a standard ToolTip. It can have a 
         *                        maximum of 64 characters including the 
         *                        terminating NULL.
         * @param icon            Handle to the icon to be added, modified, 
         *                        or deleted. 
         * @param initiallyHidden Controls whether the tray icon is shown or
         *                        initially hidden when the method returns
         *                        successfully.
         * @param balloonText     Pointer to a null-terminated string with the 
         *                        text for a balloon ToolTip. It can have a 
         *                        maximum of 256 characters including the
         *                        terminating NULL. 
         * @param balloonTitle    Pointer to a null-terminated string 
         *                        containing a title for a balloon ToolTip. 
         *                        This title appears in boldface above the text.
         *                        It can have a maximum of 64 characters 
         *                        including the terminating NULL.
         * @param balloonIcon     The icon of the balloon ToolTip. This must be
         *                        one of NIIF_NONE, NIIF_INFO, NIIF_WARNING or 
         *                        NIIF_ERROR.
         * @param balloonTimeout  The timeout value, in milliseconds, for a 
         *                        balloon ToolTip. The system minimum and 
         *                        maximum timeout values are currently set 
         *                        at 10 seconds and 30 seconds, respectively.
         *
         * @throws SystemException If the tray icon could not be created.
         */
        void Create(HWND targetWnd, const UINT callbackMessage, const UINT id,
            const char *toolTip, const HICON icon,
            const bool initiallyHidden = false, 
            const char *balloonText = NULL, 
            const char *balloonTitle = NULL, 
            const DWORD balloonIcon = NIIF_NONE, 
            const UINT balloonTimeout = 10);

        /**
         * Creates a new tray icon. If this tray icon has already been created,
         * it is destroyed and recreated using the specified parameters.
         *
         * @param targetWnd       Handle to the window that receives 
         *                        notification messages associated with an icon 
         *                        in the taskbar status area. The Shell uses 
         *                        'targetWnd' and 'id' to identify which icon 
         *                        to operate on when Shell_NotifyIcon is 
         *                        invoked.
         *                        If 'targetWnd' is NULL, the object will 
         *                        created an own hidden window to handle
         *                        messages.
         * @param callbackMessage Application-defined message identifier. The 
         *                        system uses this identifier to send 
         *                        notifications to the window identified in 
         *                        'targetWnd'. These notifications are sent 
         *                        when a mouse event occurs in the bounding 
         *                        rectangle of the icon, or when the icon is 
         *                        selected or activated with the keyboard. 
         *                        The wParam parameter of the message contains
         *                        the identifier of the taskbar icon in which 
         *                        the event occurred. The lParam parameter holds
         *                        the mouse or keyboard message associated with 
         *                        the event.
         * @param id              Application-defined identifier of the taskbar
         *                        icon. The Shell uses 'targetWnd' and 'id' to 
         *                        identify which icon to operate on when 
         *                        Shell_NotifyIcon is invoked. You can have 
         *                        multiple icons associated with a single 
         *                        'targetWnd' by assigning each a different 
         *                        'id'.
         * @param hResourceModule A handle to the module to look for resources
         *                        in.
         * @param toolTip         Pointer to a null-terminated string with the 
         *                        text for a standard ToolTip. It can have a 
         *                        maximum of 64 characters including the 
         *                        terminating NULL.
         * @param iconID          The resource ID of the icon to be added, 
         *                        modified, or deleted. 
         * @param initiallyHidden Controls whether the tray icon is shown or
         *                        initially hidden when the method returns
         *                        successfully.
         * @param balloonText     Pointer to a null-terminated string with the 
         *                        text for a balloon ToolTip. It can have a 
         *                        maximum of 256 characters including the
         *                        terminating NULL. 
         * @param balloonTitle    Pointer to a null-terminated string 
         *                        containing a title for a balloon ToolTip. 
         *                        This title appears in boldface above the text.
         *                        It can have a maximum of 64 characters 
         *                        including the terminating NULL.
         * @param balloonIcon     The icon of the balloon ToolTip. This must be
         *                        one of NIIF_NONE, NIIF_INFO, NIIF_WARNING or 
         *                        NIIF_ERROR.
         * @param balloonTimeout  The timeout value, in milliseconds, for a 
         *                        balloon ToolTip. The system minimum and 
         *                        maximum timeout values are currently set 
         *                        at 10 seconds and 30 seconds, respectively.
         *
         * @throws SystemException If the tray icon could not be created.
         */
        void Create(HWND targetWnd, const UINT callbackMessage, const UINT id,
            HINSTANCE hResourceModule, const wchar_t *toolTip, 
            const UINT iconID, const bool initiallyHidden = false, 
            const wchar_t *balloonText = NULL, 
            const wchar_t *balloonTitle = NULL, 
            const DWORD balloonIcon = NIIF_NONE, 
            const UINT balloonTimeout = 10);

        /**
         * Creates a new tray icon. If this tray icon has already been created,
         * it is destroyed and recreated using the specified parameters.
         *
         * @param targetWnd       Handle to the window that receives 
         *                        notification messages associated with an icon 
         *                        in the taskbar status area. The Shell uses 
         *                        'targetWnd' and 'id' to identify which icon 
         *                        to operate on when Shell_NotifyIcon is 
         *                        invoked.
         *                        If 'targetWnd' is NULL, the object will 
         *                        created an own hidden window to handle
         *                        messages.
         * @param callbackMessage Application-defined message identifier. The 
         *                        system uses this identifier to send 
         *                        notifications to the window identified in 
         *                        'targetWnd'. These notifications are sent 
         *                        when a mouse event occurs in the bounding 
         *                        rectangle of the icon, or when the icon is 
         *                        selected or activated with the keyboard. 
         *                        The wParam parameter of the message contains
         *                        the identifier of the taskbar icon in which 
         *                        the event occurred. The lParam parameter holds
         *                        the mouse or keyboard message associated with 
         *                        the event.
         * @param id              Application-defined identifier of the taskbar
         *                        icon. The Shell uses 'targetWnd' and 'id' to 
         *                        identify which icon to operate on when 
         *                        Shell_NotifyIcon is invoked. You can have 
         *                        multiple icons associated with a single 
         *                        'targetWnd' by assigning each a different 
         *                        'id'.
         * @param hResourceModule A handle to the module to look for resources
         *                        in.
         * @param toolTip         Pointer to a null-terminated string with the 
         *                        text for a standard ToolTip. It can have a 
         *                        maximum of 64 characters including the 
         *                        terminating NULL.
         * @param iconID          The resource ID of the icon to be added, 
         *                        modified, or deleted. 
         * @param initiallyHidden Controls whether the tray icon is shown or
         *                        initially hidden when the method returns
         *                        successfully.
         * @param balloonText     Pointer to a null-terminated string with the 
         *                        text for a balloon ToolTip. It can have a 
         *                        maximum of 256 characters including the
         *                        terminating NULL. 
         * @param balloonTitle    Pointer to a null-terminated string 
         *                        containing a title for a balloon ToolTip. 
         *                        This title appears in boldface above the text.
         *                        It can have a maximum of 64 characters 
         *                        including the terminating NULL.
         * @param balloonIcon     The icon of the balloon ToolTip. This must be
         *                        one of NIIF_NONE, NIIF_INFO, NIIF_WARNING or 
         *                        NIIF_ERROR.
         * @param balloonTimeout  The timeout value, in milliseconds, for a 
         *                        balloon ToolTip. The system minimum and 
         *                        maximum timeout values are currently set 
         *                        at 10 seconds and 30 seconds, respectively.
         *
         * @throws SystemException If the tray icon could not be created.
         */
        void Create(HWND targetWnd, const UINT callbackMessage, const UINT id,
            HINSTANCE hResourceModule, const char *toolTip, 
            const UINT iconID, const bool initiallyHidden = false, 
            const char *balloonText = NULL, 
            const char *balloonTitle = NULL, 
            const DWORD balloonIcon = NIIF_NONE, 
            const UINT balloonTimeout = 10);

        /**
         * Creates a new tray icon. If this tray icon has already been created,
         * it is destroyed and recreated using the specified parameters.
         *
         * @param targetWnd       Handle to the window that receives 
         *                        notification messages associated with an icon 
         *                        in the taskbar status area. The Shell uses 
         *                        'targetWnd' and 'id' to identify which icon 
         *                        to operate on when Shell_NotifyIcon is 
         *                        invoked.
         *                        If 'targetWnd' is NULL, the object will 
         *                        created an own hidden window to handle
         *                        messages.
         * @param callbackMessage Application-defined message identifier. The 
         *                        system uses this identifier to send 
         *                        notifications to the window identified in 
         *                        'targetWnd'. These notifications are sent 
         *                        when a mouse event occurs in the bounding 
         *                        rectangle of the icon, or when the icon is 
         *                        selected or activated with the keyboard. 
         *                        The wParam parameter of the message contains
         *                        the identifier of the taskbar icon in which 
         *                        the event occurred. The lParam parameter holds
         *                        the mouse or keyboard message associated with 
         *                        the event.
         * @param id              Application-defined identifier of the taskbar
         *                        icon. The Shell uses 'targetWnd' and 'id' to 
         *                        identify which icon to operate on when 
         *                        Shell_NotifyIcon is invoked. You can have 
         *                        multiple icons associated with a single 
         *                        'targetWnd' by assigning each a different 
         *                        'id'.
         * @param hResourceModule A handle to the module to look for resources
         *                        in.
         * @param toolTipID       Resource ID of a null-terminated string with 
         *                        the text for a standard ToolTip. It can have a
         *                        maximum of 64 characters including the 
         *                        terminating NULL.
         * @param iconID          The resource ID of the icon to be added, 
         *                        modified, or deleted. 
         * @param initiallyHidden Controls whether the tray icon is shown or
         *                        initially hidden when the method returns
         *                        successfully.
         * @param balloonTextID   Resource ID of a null-terminated string with 
         *                        the text for a balloon ToolTip. It can have a
         *                        maximum of 256 characters including the
         *                        terminating NULL. 
         * @param balloonTitleID  Resource ID of a null-terminated string 
         *                        containing a title for a balloon ToolTip. 
         *                        This title appears in boldface above the text.
         *                        It can have a maximum of 64 characters 
         *                        including the terminating NULL.
         * @param balloonIcon     The icon of the balloon ToolTip. This must be
         *                        one of NIIF_NONE, NIIF_INFO, NIIF_WARNING or 
         *                        NIIF_ERROR.
         * @param balloonTimeout  The timeout value, in milliseconds, for a 
         *                        balloon ToolTip. The system minimum and 
         *                        maximum timeout values are currently set 
         *                        at 10 seconds and 30 seconds, respectively.
         *
         * @throws SystemException If the tray icon could not be created.
         */
        void Create(HWND targetWnd, const UINT callbackMessage, const UINT id,
            HINSTANCE hResourceModule, const UINT toolTipID, 
            const UINT iconID, const bool initiallyHidden = false, 
            const UINT balloonTextID = 0, 
            const UINT balloonTitleID = 0, 
            const DWORD balloonIcon = NIIF_NONE, 
            const UINT balloonTimeout = 10);

        /**
         * Delete the tray icon.
         *
         * This method has no effect, if the icon has not been installed.
         *
         * @throw SystemExcption If the icon could not be deleted.
         */
        void Destroy(void);

        /**
         * Hide the tray icon.
         *
         * This method has no effect, if the tray icon is not visible.
         *
         * @throws SystemException If the icon could not be hidden.
         */
        void Hide(void);

        /**
         * Answer the current state of the tray icon.
         *
         * @return The current visiblity state.
         */
        inline IconState GetIconState(void) const {
            return this->iconState;
        }

		/**
		 * Change the tray icon to 'icon'.
		 *
		 * @param icon the handle of the new icon. The caller remains owner
		 *             of the icon and can destroy it right afterwards.
		 *
		 * @throws SystemException in case of error.
		 */
		inline void SetIcon(const HICON icon) {
			this->configureIcon(icon);
			this->notify(NIM_MODIFY);
		}

		/**
		 * Change the tray icon to the icon pointed to by 'iconID'.
		 * 
         * @param hResourceModule A handle to the module to look for resources
         *                        in.
         * @param iconID          The resource ID of the icon to be added, 
         *                        modified, or deleted. 
		 *
		 * @throws SystemException in case of error.
		 */
		void SetIcon(HINSTANCE hResourceModule, const UINT iconID);

        /**
         * Show the tray icon.
         *
         * This method has no effect, if the tray icon is already visible.
         *
         * @throws SystemException If the icon could not be made visible.
         */
        void Show(void);

        /**
         * Displays a balloon tool tip. 
         *
         * See documentation of Create for more details about the balloon help
         * parameters.
         *
         * @param balloonText    The tool tip text. If empty, an active balloon 
         *                       help will be hidden.
         * @param balloonTitle   The title above the tool tip text in bold face.
         * @param balloonIcon    The icon to be shown.
         * @param balloonTimeout The timeout.
         *
         * @return true, if the balloon help was shown, false otherwise (e. g. 
         *         on systems which do not support balloon help).
         */
        bool ShowBalloonHelp(const wchar_t *balloonText, 
            const wchar_t *balloonTitle = NULL, 
            const DWORD balloonIcon = NIIF_NONE, 
            const UINT balloonTimeout = 10);

        /**
         * Displays a balloon tool tip.
         *
         * See documentation of Create for more details about the balloon help
         * parameters.
         *
         * @param balloonText    The tool tip text. If empty, an active balloon 
         *                       help will be hidden.
         * @param balloonTitle   The title above the tool tip text in bold face.
         * @param balloonIcon    The icon to be shown.
         * @param balloonTimeout The timeout.
         *
         * @return true, if the balloon help was shown, false otherwise (e. g. 
         *         on systems which do not support balloon help).
         */
        bool ShowBalloonHelp(const char *balloonText, 
            const char *balloonTitle = NULL, 
            const DWORD balloonIcon = NIIF_NONE, 
            const UINT balloonTimeout = 10);

        /**
         * Displays a balloon tool tip.
         *
         * See documentation of Create for more details about the balloon help
         * parameters.
         *
         * @param hResourceModule The module to look for the strings in.
         * @param balloonTextID   The tool tip text. If empty, an active
         *                        balloon help will be hidden.
         * @param balloonTitleID  The title above the tool tip text in bold 
         *                        face.
         * @param balloonIcon     The icon to be shown.
         * @param balloonTimeout  The timeout.
         *
         * @return true, if the balloon help was shown, false otherwise (e. g. 
         *         on systems which do not support balloon help).
         */
        bool ShowBalloonHelp(HINSTANCE hResourceModule, 
            const UINT balloonTextID, const UINT balloonTitleID = 0, 
            const DWORD balloonIcon = NIIF_NONE, 
            const UINT balloonTimeout = 10);

    protected:

        /**
         * If the internal window is used for handling notifications, the
         * tray icon object will call this method once a notification on
         * the icon occurred. Subclasses can override this method in order
         * to handle events.
         *
         * @param wParam The identifier of the taskbar icon in which the event 
         *               occurred. This should always be 'this->nid.uID'.
         * @param lParam The mouse or keyboard message associated with the 
         *               event.
         *
         * @return You should return 0 in case of success.
         */
        virtual LRESULT onNotify(WPARAM wParam, LPARAM lParam);

    private:

        /** The capabilities of the shell regarding tray icons. */
        enum Capabilities {
            CAPABILITIES_NONE = 0,
            CAPABILITIES_V1 = 1,
            CAPABILITIES_V2 = 2
        };

        static LRESULT WINAPI wndProc(HWND hWnd, UINT msg, WPARAM wParam,
            LPARAM lParam);

        /** The maximum length of a balloon help text. */
        static const UINT MAX_BALLOON_LEN;

        /** The maximum length of a balloon title text. */
        static const UINT MAX_BALLOON_TITLE_LEN;

        /** The maximum length of a tray icon tooltip. */
        static const UINT MAX_TOOLTIP_LEN;

        /** The name of the window class used for the controlling window. */
        static const wchar_t *WNDCLASSNAME;

        /**
         * Configure the balloon help properties in 'this->nid'. The properties
         * and the appropriate flag are set.
         *
         * @param balloonText    The actual text on the balloon.
         * @param balloonTitle   An optional title text.
         * @param balloonIcon    The icon to be shown. This must be one of 
         *                       NIIF_NONE, NIIF_INFO, NIIF_WARNING or 
         *                       NIIF_ERROR.
         * @param balloonTimeout The timeout in seconds. This must be a value
         *                       within [10, 30].
         *
         * @return true, if balloon help is supported and has been configured,
         *         false otherwise.
         */
        bool configureBalloonHelp(const wchar_t *balloonText, 
            const wchar_t *balloonTitle, const DWORD balloonIcon, 
            const UINT balloonTimeout);

        /**
         * Configure the 'uCallbackMessage' member in 'this->nid'. The property
         * and the appropriate flag are set.
         *
         * @return true.
         */
        bool configureCallbackMessage(const UINT callbackMessage);

        /**
         * Configure the 'hIcon' member in 'this->nid'. The property
         * and the appropriate flag are set.
         *
         * @return true.
         */
        bool configureIcon(const HICON icon);

        /**
         * Configure the visibility member 'this->nid'. The properties
         * and the appropriate flags are set.
         *
         * @param isHidden true for hiding the icon, false for showing it.
         *
         * @return true, if hiding is supported and has been configured,
         *         false otherwise.
         */
        bool configureHidden(const bool isHidden);

        /**
         * Configure the 'szTip' member in 'this->nid'. The property
         * and the appropriate flag are set.
         *
         * @return true.
         */
        bool configureToolTip(const wchar_t *toolTip);

        /**
         * In-ctor initialisations.
         */
        bool init(HINSTANCE hInstance);

        /**
         * Notify the shell icon with the current data 'this->nid' and use
         * 'message' as message.
         *
         * @param message The message to send. Note that 'this->nid' must be
         *                configured appropriately.
         *
         * @throws SystemException If ::Shell_NotifyIcon fails.
         */
        void notify(const DWORD message);

        /**
         * Register the window class for the hidden window that is used for
         * controlling the tray icon. The window class has the name
         * TrayIcon::WNDCLASSNAME. If this window class has already been 
         * registered, the method does nothing.
         *
         * @return true, if the window class has been registered or has already
         *         been registered before,
         *         false otherwise.
         */
        bool registerWndClass(void);

        /** The capabilities of the notify icon. */
        Capabilities capabilities;

        /** Instance handle. */
        HINSTANCE hInstance;

        /**
         * The hidden window that receives notifications about events occurring 
         * on the tray icon. If the user provided an own window to be notified,
         * this member remains NULL.
         */
        HWND hWnd;

        /** The current state of the tray icon. */
        IconState iconState;

        /** The configuration data of the tray icon. */
        NOTIFYICONDATAW nid;
    };

} /* end namespace sys */
} /* end namespace vislib */


#endif /* _WIN32 */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_TRAYICON_H_INCLUDED */
