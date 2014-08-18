using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace vislib.gl {

    /// <summary>
    /// Class counting frames per second
    /// </summary>
    internal class FPSCounter {

        /// <summary>
        /// The size of the time buffer
        /// Must be a multiple of 2 (odd)
        /// </summary>
        private const int timeBufferSize = 50;

        /// <summary>
        /// The time buffer
        /// </summary>
        private long[] timeBuffer = new long[timeBufferSize];

        /// <summary>
        /// The position in the time buffer
        /// </summary>
        private int timeBufferPos = 0;

        /// <summary>
        /// Answer the currently counted FPS
        /// </summary>
        public float FPS {
            get {
                float time = 0.0f;
                float cnt = 0.0f;
                for (int i = 2; i < timeBufferSize; i += 2) {
                    long diff = this.timeBuffer[i] - this.timeBuffer[i - 2];
                    if (diff > 0.0f) {
                        time += (float)diff;
                        cnt += 1.0f;
                    }
                }
                time /= TimeSpan.TicksPerSecond;
                return (time > 0.0f) ? (cnt / time) : 0.0f;
            }
        }

        /// <summary>
        /// Marks the begin of a frame
        /// </summary>
        public void BeginFrame() {
            this.timeBuffer[this.timeBufferPos] = DateTime.Now.Ticks;
            this.timeBufferPos = (this.timeBufferPos + 1) % timeBufferSize;
        }

        /// <summary>
        /// Clears the fps counter
        /// </summary>
        public void ClearCounter() {
            for (int i = 0; i < timeBufferSize; i++) {
                this.timeBuffer[i] = 0;
            }
            this.timeBufferPos = 0;
        }

        /// <summary>
        /// Marks the end of a frame
        /// </summary>
        public void EndFrame() {
            this.timeBuffer[this.timeBufferPos] = DateTime.Now.Ticks;
            this.timeBufferPos = (this.timeBufferPos + 1) % timeBufferSize;
        }

    }

}
