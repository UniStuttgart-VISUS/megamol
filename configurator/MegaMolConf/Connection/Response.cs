using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MegaMolConf.Communication {

    /// <summary>
    /// The response sent by MegaMol to answer a request
    /// </summary>
    public class Response {

        private Request req = null;

        /// <summary>
        /// The original request this response was replied to
        /// </summary>
        public Request Request {
            get { return req; }
            internal set {
                if (value != null) {
                    req = value;
                    Command = req.Command;
                }
            }
        }

        /// <summary>
        /// The original request command this response was replied to
        /// </summary>
        public String Command { get; private set; }

        /// <summary>
        /// The error state of this response
        /// </summary>
        public string Error { get; private set; }

        /// <summary>
        /// The answer objects
        /// </summary>
        public string Answer { get; private set; }

        internal void fromZFrameString(string v) {
            if ((v != null) && (v.StartsWith("Error: "))) {
                Error = v.Substring(7);
                Answer = null;
                return;
            }
            Error = null;
            Answer = v;
        }

        /// <summary>
        /// Human readable string of the answer hold in this request
        /// </summary>
        /// <returns>A string or null if the request is empty</returns>
        public override string ToString() {
            if (Error != null) return "Error: " + Error;
            if (Answer == null) return null;
            return Answer.ToString();
        }

    }

}
