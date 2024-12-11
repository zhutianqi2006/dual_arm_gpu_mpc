function slBusOut = String(msgIn, slBusOut, varargin)
%#codegen
%   Copyright 2021-2022 The MathWorks, Inc.
    slBusOut.data_SL_Info.ReceivedLength = uint32(strlength(msgIn.data));
    currlen  = min(slBusOut.data_SL_Info.ReceivedLength, length(slBusOut.data));
    slBusOut.data_SL_Info.CurrentLength = uint32(currlen);
    slBusOut.data(1:currlen) = uint8(char(msgIn.data(1:currlen))).';
end
