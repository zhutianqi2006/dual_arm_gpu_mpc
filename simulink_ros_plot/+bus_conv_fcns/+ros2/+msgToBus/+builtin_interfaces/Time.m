function slBusOut = Time(msgIn, slBusOut, varargin)
%#codegen
%   Copyright 2021-2022 The MathWorks, Inc.
    slBusOut.sec = int32(msgIn.sec);
    slBusOut.nanosec = uint32(msgIn.nanosec);
end
