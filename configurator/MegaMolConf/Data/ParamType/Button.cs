using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MegaMolConf.Data.ParamType {

    public sealed class Button : ParamTypeBase {
        public override bool ValuesEqual(string a, string b) {
            return true;
        }
    }

}
