function data = record_figures_terminate( data )
%RECORD_FIGURES_TERMINATE Summary of this function goes here
%   Detailed explanation goes here


for f=data.vidObj
    try
        close(f{1});
    end
end

end

