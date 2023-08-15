acq = oscilloscope_single_acquisition:new():points(points):count(count):segmented(true)
trigger = oscilloscope_trigger:new("EXT", "EDGE")
trigger:level(5, oscilloscope_quantity:new(2000.0, "mV")):slope(oscilloscope_trigger_slope.rising):mode(oscilloscope_trigger_mode.normal)
quant = oscilloscope_quantity:new(range, "ms")
rtb02 = rtx_instrument_configuration:new(quant, acq, trigger, timeout);
chan_1 = oscilloscope_channel:new(1)
chan_1:state(true):attenuation(oscilloscope_quantity:new(10, "A")):range(oscilloscope_quantity:new(40, "A")):label(oscilloscope_label:new("current1", true))
chan_2 = oscilloscope_channel:new(2)
chan_2:state(true):attenuation(oscilloscope_quantity:new(10, "A")):range(oscilloscope_quantity:new(40, "A")):label(oscilloscope_label:new("current2", true))
rtb02:channel(chan_1)
rtb02:channel(chan_2)
rtb02 = as_slave(rtb02)