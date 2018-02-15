using System;
using System.ComponentModel;
using System.Drawing.Design;
using System.Globalization;
using System.Windows;
using System.Windows.Forms;
using System.Windows.Forms.Design;

namespace MegaMolConf {

    internal class TransferFunc1DEditor : UITypeEditor {

        //internal class ElementTypeConverter : TypeConverter {
        //    public override bool CanConvertTo(ITypeDescriptorContext context, Type destinationType) {
        //        return typeof(string) == destinationType;
        //    }
        //    public override object ConvertTo(ITypeDescriptorContext context, CultureInfo culture, object value, Type destinationType) {
        //        if (typeof(string) == destinationType) {
        //            return ((Element)value).DisplayName;
        //        }
        //        return string.Empty;
        //    }
        //}

        //[TypeConverter(typeof(ElementTypeConverter))]
        //internal class Element {
        //    public string Value { get; set; }
        //    public Element(string n) {
        //        this.Value = n;
        //    }
        //    public string DisplayName {
        //        get { return this.Value; }
        //    }
        //    public override string ToString() {
        //        return this.Value;
        //    }
        //};

        private Data.ParamType.TransferFunc1D e;

        private IWindowsFormsEditorService _editorService;

        public TransferFunc1DEditor(Data.ParamType.TransferFunc1D e) {
            this.e = e;
        }

        public override UITypeEditorEditStyle GetEditStyle(ITypeDescriptorContext context) {
            // drop down mode (we'll host a listbox in the drop down)
            return UITypeEditorEditStyle.DropDown;
        }

        public override object EditValue(ITypeDescriptorContext context, IServiceProvider provider, object value) {
            _editorService = (IWindowsFormsEditorService)provider.GetService(typeof(IWindowsFormsEditorService));

            TransferFunc1DDialog tf1c = new TransferFunc1DDialog();

            // show this model stuff
            //_editorService.DropDownControl(tf1c);
            _editorService.ShowDialog(tf1c);

            return tf1c.GetSerializedTransferFunction();
        }

        private void OnListBoxSelectedValueChanged(object sender, EventArgs e) {
            // close the drop down as soon as something is clicked
            _editorService.CloseDropDown();
        }

    }

}