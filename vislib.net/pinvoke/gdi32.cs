using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;

namespace vislib.pinvoke {

    /// <summary>
    /// class holding gdi32 p/invoke definitions
    /// </summary>
    public static class gdi32 {

        /// <summary>
        /// The library file name
        /// </summary>
        public const string LIBNAME = "gdi32.dll";

        #region structs

        [StructLayout(LayoutKind.Explicit)]
        public class PIXELFORMATDESCRIPTOR {
            [FieldOffset(0)]
            public UInt16 nSize;
            [FieldOffset(2)]
            public UInt16 nVersion;
            [FieldOffset(4)]
            public UInt32 dwFlags;
            [FieldOffset(8)]
            public Byte iPixelType;
            [FieldOffset(9)]
            public Byte cColorBits;
            [FieldOffset(10)]
            public Byte cRedBits;
            [FieldOffset(11)]
            public Byte cRedShift;
            [FieldOffset(12)]
            public Byte cGreenBits;
            [FieldOffset(13)]
            public Byte cGreenShift;
            [FieldOffset(14)]
            public Byte cBlueBits;
            [FieldOffset(15)]
            public Byte cBlueShift;
            [FieldOffset(16)]
            public Byte cAlphaBits;
            [FieldOffset(17)]
            public Byte cAlphaShift;
            [FieldOffset(18)]
            public Byte cAccumBits;
            [FieldOffset(19)]
            public Byte cAccumRedBits;
            [FieldOffset(20)]
            public Byte cAccumGreenBits;
            [FieldOffset(21)]
            public Byte cAccumBlueBits;
            [FieldOffset(22)]
            public Byte cAccumAlphaBits;
            [FieldOffset(23)]
            public Byte cDepthBits;
            [FieldOffset(24)]
            public Byte cStencilBits;
            [FieldOffset(25)]
            public Byte cAuxBuffers;
            [FieldOffset(26)]
            public SByte iLayerType;
            [FieldOffset(27)]
            public Byte bReserved;
            [FieldOffset(28)]
            public UInt32 dwLayerMask;
            [FieldOffset(32)]
            public UInt32 dwVisibleMask;
            [FieldOffset(36)]
            public UInt32 dwDamageMask;
        }

        public const byte PFD_TYPE_RGBA = 0;
        public const byte PFD_TYPE_COLORINDEX = 1;

        public const uint PFD_DOUBLEBUFFER = 1;
        public const uint PFD_STEREO = 2;
        public const uint PFD_DRAW_TO_WINDOW = 4;
        public const uint PFD_DRAW_TO_BITMAP = 8;
        public const uint PFD_SUPPORT_GDI = 16;
        public const uint PFD_SUPPORT_OPENGL = 32;
        public const uint PFD_GENERIC_FORMAT = 64;
        public const uint PFD_NEED_PALETTE = 128;
        public const uint PFD_NEED_SYSTEM_PALETTE = 256;
        public const uint PFD_SWAP_EXCHANGE = 512;
        public const uint PFD_SWAP_COPY = 1024;
        public const uint PFD_SWAP_LAYER_BUFFERS = 2048;
        public const uint PFD_GENERIC_ACCELERATED = 4096;
        public const uint PFD_SUPPORT_DIRECTDRAW = 8192;

        public const sbyte PFD_MAIN_PLANE = 0;
        public const sbyte PFD_OVERLAY_PLANE = 1;
        public const sbyte PFD_UNDERLAY_PLANE = -1;

        #endregion

        #region functions

        [DllImport(LIBNAME, SetLastError = true)]
        public static extern int ChoosePixelFormat(IntPtr hDC,
            [In, MarshalAs(UnmanagedType.LPStruct)] PIXELFORMATDESCRIPTOR ppfd);

        [DllImport(LIBNAME, SetLastError = true)]
        public static extern int SetPixelFormat(IntPtr hDC, int iPixelFormat,
            [In, MarshalAs(UnmanagedType.LPStruct)] PIXELFORMATDESCRIPTOR ppfd);

        [DllImport(LIBNAME)]
        public static extern int SwapBuffers(IntPtr hDC);

        //[DllImport(LIBNAME, SetLastError = true)]
        //public static extern IntPtr CreateDIBSection(
        //    IntPtr hdc,
        //    [In, MarshalAs(UnmanagedType.LPStruct)] BITMAPINFO pbmi,
        //    uint iUsage,
        //    out IntPtr ppvBits,
        //    IntPtr hSection,
        //    uint dwOffset);

        //[DllImport(LIBNAME)]
        //public static extern bool DeleteObject(IntPtr hObject);

        //[DllImport(LIBNAME, SetLastError = true)]
        //public static extern IntPtr CreateCompatibleDC(IntPtr hDC);

        //[DllImport(LIBNAME)]
        //public static extern bool DeleteDC(IntPtr hDC);

        //[DllImport(LIBNAME)]
        //public static extern IntPtr SelectObject(IntPtr hDC, IntPtr hObject);

        #endregion

    }

}
