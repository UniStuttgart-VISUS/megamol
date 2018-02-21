using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml.Serialization;

namespace MegaMolConf.Data {

    [XmlInclude(typeof(ParamType.Bool)),
    XmlInclude(typeof(ParamType.Button)),
    XmlInclude(typeof(ParamType.Enum)),
    XmlInclude(typeof(ParamType.FlexEnum)),
    XmlInclude(typeof(ParamType.Float)),
    XmlInclude(typeof(ParamType.Int)),
    XmlInclude(typeof(ParamType.FilePath)),
    XmlInclude(typeof(ParamType.TransferFunc1D)),
    XmlInclude(typeof(ParamType.String))]
    public abstract class ParamTypeBase {
        public string TypeName { get; set; }
        abstract public bool ValuesEqual(string a, string b);
    }

}
