function slBusOut = MultiArrayLayout(msgIn, slBusOut, varargin)
%#codegen
%   Copyright 2021-2022 The MathWorks, Inc.
    maxlength = length(slBusOut.dim);
    recvdlength = length(msgIn.dim);
    currentlength = min(maxlength, recvdlength);
    if (max(recvdlength) > maxlength) && ...
            isequal(varargin{1}{1},ros.slros.internal.bus.VarLenArrayTruncationAction.EmitWarning)
        diag = MSLDiagnostic([], ...
                             message('ros:slros:busconvert:TruncatedArray', ...
                                     'dim', msgIn.MessageType, maxlength, max(recvdlength), maxlength, varargin{2}));
        reportAsWarning(diag);
    end
    slBusOut.dim_SL_Info.ReceivedLength = uint32(recvdlength);
    slBusOut.dim_SL_Info.CurrentLength = uint32(currentlength);
    for iter=1:currentlength
        slBusOut.dim(iter) = bus_conv_fcns.ros2.msgToBus.std_msgs.MultiArrayDimension(msgIn.dim(iter),slBusOut(1).dim(iter),varargin{:});
    end
    slBusOut.data_offset = uint32(msgIn.data_offset);
end
