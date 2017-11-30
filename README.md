### Snow Events Picker
The functions implemented in this script can automatically pick-out 
solid snow precipitation events and melting events from the 
wireless-sensor network's daily time-series data.

The user need to prepare a pandas data frame, the index of which should
be date/datetime, and each column of the table should be daily time-series
snow-depth data of each sensor. Please make sure all sensors/columns are
representing sensor nodes within the same WSN site.

When using the tool, the user need to call `get_precipitation_events` for
picking out all precipitation events that are in solid phase, or call
`get_melting_events` for selecting all melting periods.

Please be aware this tool is semi-automated. The user needs to specify a few
parameters so the expected results can be calculated.