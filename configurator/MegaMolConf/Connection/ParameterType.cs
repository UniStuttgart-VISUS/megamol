using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MegaMolConf.Communication {

    /// <summary>
    /// Possible MegaMol parameter Types
    /// </summary>
    public enum ParameterType {
        Unknown,
        ButtonParam,
        BoolParam,
        EnumParam,
        FilePathParam,
        FloatParam,
        IntParam,
        StringParam,
        TernaryParam,
        Vector2fParam,
        Vector3fParam,
        Vector4fParam,
        FlexEnumParam,
        TransferFunc1DParam
    }

}
