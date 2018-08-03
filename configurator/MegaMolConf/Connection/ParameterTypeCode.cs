﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MegaMolConf.Communication {

    /// <summary>
    /// Known parameter types in MegaMol.
    /// </summary>
    /// <remark>
    /// Please make sure that the number values match the parameter type 
    /// memory footprint (AKA the 6CC interpreted as uint64_t). 
    /// Example: struct.unpack('q', b'MMCOLO\0\0') => (87189165919565,)
    /// </remark>
    public enum ParameterTypeCode : ulong {
        /// <summary>
        /// ButtonParam
        /// </summary>
        MMBUTN = 86124114627917,
        /// <summary>
        /// BoolParam
        /// </summary>
        MMBOOL = 83903515872589,
        /// <summary>
        /// EnumParam
        /// </summary>
        MMENUM = 85028780723533,
        /// <summary>
        /// FilePathParam
        /// </summary>
        MMFILW = 95985158475085,
        /// <summary>
        /// FilePathParam
        /// </summary>
        MMFILA = 71795902664013,
        /// <summary>
        /// FloatParam
        /// </summary>
        MMFLOT = 92699558825293,
        /// <summary>
        /// IntParam
        /// </summary>
        MMINTR = 90522044157261,
        /// <summary>
        /// StringParam
        /// </summary>
        MMSTRW = 96011113680205,
        /// <summary>
        /// StringParam
        /// </summary>
        MMSTRA = 71821857869133,
        /// <summary>
        /// TernaryParam
        /// </summary>
        MMTRRY = 98210103446861,
        /// <summary>
        /// Vector2fParam
        /// </summary>
        MMVC2F = 77181692038477,
        /// <summary>
        /// Vector3fParam
        /// </summary>
        MMVC3F = 77185987005773,
        /// <summary>
        /// Vector4fParam
        /// </summary>
        MMVC4F = 77190281973069,
        /// <summary>
        /// FlexEnumParam
        /// </summary>
        MMFENU = 93794658045261,
        /// <summary>
        /// TransferFunc1DParam
        /// </summary>
        MMTF1W = 95869144943949,
        /// <summary>
        /// TransferFunc1DParam
        /// </summary>
        MMTF1A = 71679889132877,
	    /// <summary>
        /// ColorParam
        /// </summary>
        MMCOLO = 87189165919565
    }

}
