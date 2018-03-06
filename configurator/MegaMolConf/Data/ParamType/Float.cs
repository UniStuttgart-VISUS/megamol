namespace MegaMolConf.Data.ParamType {

    public sealed class Float : ParamTypeValueBase {
        public float DefaultValue { get; set; }
        public float MinValue { get; set; }
        public float MaxValue { get; set; }
        public override void ParseDefaultValue(string v) {
            DefaultValue = float.Parse(v, System.Globalization.CultureInfo.InvariantCulture);
        }
        public override string DefaultValueString() {
            return DefaultValue.ToString(System.Globalization.CultureInfo.InvariantCulture);
        }
        public override bool ValuesEqual(string a, string b) {
            return float.Parse(a, System.Globalization.CultureInfo.InvariantCulture).Equals(float.Parse(b, System.Globalization.CultureInfo.InvariantCulture));
        }
    }

}
