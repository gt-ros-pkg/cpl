function [v tag] = nx_fromxmlstr(s)
%FROMXMLSTR Summary of this function goes here
%   Detailed explanation goes here

    xdoc  = xmlreadstring(s);
    xroot = xdoc.getDocumentElement;
    [tag v] = read_node(xroot);
end

function [nodename nodevalue] = read_node(n)
    
    nodename = char(n.getNodeName());
    
    if n.getLength() == 1 & strcmp('#text', n.getChildNodes.item(0).getNodeName())
        
        nodevalue = char(n.getChildNodes.item(0).getTextContent());
        
        if n.getAttributes.getLength() == 2
            nodevalue = str2num(nodevalue);
            [rows cols] = size(nodevalue);
            
            for i=1:n.getAttributes.getLength()
                a = n.getAttributes.item(i-1);
                if strcmp(a.getName(), 'rows')
                    rows = str2num(char(a.getTextContent()));
                elseif strcmp(a.getName(), 'cols')
                    cols = str2num(char(a.getTextContent()));
                end
            end
            
            if rows * cols == length(nodevalue)
                nodevalue = reshape(nodevalue, [rows cols]);
            end
        end
        
    else
        nodevalue = struct();
        for i=1:n.getLength()
            [f v] = read_node(n.getChildNodes.item(i-1));
            
            if ~isfield(nodevalue, f)
                nodevalue.(f) = v;
            else
                nodevalue.(f)(end+1) = v;
            end
        end
        
    end
end