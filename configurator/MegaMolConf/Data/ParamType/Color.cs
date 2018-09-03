using SD = System.Drawing;

namespace MegaMolConf.Data.ParamType {
    public sealed class Color : ParamTypeValueBase {
        public string DefaultValue { get; set; }

        public override void ParseDefaultValue(string v) {
            this.DefaultValue = v;
        }

        public override string DefaultValueString() {
            return this.DefaultValue;
        }

        public override bool ValuesEqual(string a, string b) {
            return a.Equals(b);
        }

        public static string ToString(SD.Color c) {
            return $"#{c.R:X2}{c.G:X2}{c.B:X2}{c.A:X2}";
        }

        public static SD.Color FromString(string s) {
            return SD.ColorTranslator.FromHtml(s);
        }
    }
}
