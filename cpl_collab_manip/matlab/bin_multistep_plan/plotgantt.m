function plotgantt(in_struct,st_day,st_mon,st_yr,en_day,en_mon,en_yr)
%PLOTGANTT creates a Gantt chart from a structure of data
%
%  POLTGANTT(DATA,ST_DAY,ST_MON,ST_YR,EN_DAY,EN_MON,EN_YR)
%
%  DATA    A structure array of vectors. The field names of the structure
%                will be printed as the Y-axis labels in the Gantt plot. In the
%                vector, use NaNs where there will be no data on the plot, and  
%                 1s where there will be data on the plot.
%  ST_DAY, ST_MON, ST_YR    scalars showing the start date, start month 
%                                                     and start year of the plot
%  EN_DAY, EN_MON, EN_YR  scalars showing the end date, end month 
%                                                     and end year of the plot
%

%  Adam Leadbetter (alead@bodc.ac.uk) - 2009May13

%
%  Invocation checks
%
  if(nargin  ~=  7)
    error('Error: PLOTGANTT requires 7 (seven) inputs...');
  end
  start_date  =  datenum(st_yr,st_mon,st_day);
  end_date  =  datenum(en_yr,en_mon,en_day);
  if(end_date  <=  start_date)
    error('Error: End date must be after start date...');
  end
%
%  Create a cell array of strings containing the dates
%
  Dates  =  cell(1,1);
  doneOne  =  false;
  doneAll  =  false;
  while(~doneAll)
    t_day  =  num2str(st_day);
    if(length(t_day)  ==  1)
      t_day(2)  =  t_day(1);
      t_day(1)  =  '0';
    end
    t_mon  =  num2str(st_mon);
    if(length(t_mon)  ==  1)
      t_mon(2)  =  t_mon(1);
      t_mon(1)  =  '0';
    end
    t_day  =  cat(2,t_day,'/',t_mon);
    if(~doneOne)
      Dates{1}  =  t_day;
      doneOne  =  true;
    else
      Dates{end+1}   =  t_day;
    end
    st_day  =  st_day  +  1;
    if(st_mon  ==  9  ||  st_mon  ==  4  ...
      ||  st_mon  == 6  ||  st_mon  ==  11)
        if(st_day  ==  31)
          st_day  =  1;
          st_mon  =  st_mon  +  1;
        end
    elseif(st_mon  ==  2)
      if(st_day  ==  29  &&  ~isleapyear(st_yr))
        st_day  =  1;
        st_mon  =  3;
      elseif(st_day  ==  30  &&  isleapyear(st_yr))
        st_day  =  1;
        st_mon  =  3;
      end
    elseif(st_mon  ==  12)
      if(st_day  ==  32)
        st_day  =  1;
        st_mon  =  1;
        st_yr   =  st_yr + 1;
      end
    else
      if(st_day  ==  32)
        st_day  =  1;
        st_mon  =  st_mon  +  1;
      end
    end
    start_date  =  datenum(st_yr,st_mon,st_day);
    if(start_date  >  end_date)
      doneAll  =  true;
    end
  end
%
%  Get the field names and then loop through, plotting them one at a time
%
  hold on;
  Fields  =  fieldnames(in_struct);
  kk  =  1;
  max_field_name  =  0;
  while(kk  <=  length(Fields))
    t_data  =  eval(['in_struct.' Fields{kk} ';']);
    plot(t_data + length(Fields)-kk,'LineWidth',100/length(Fields));
    if(length(Fields{kk}(:))  >  max_field_name)
      max_field_name  =  length(Fields{kk}(:));
    end
    kk  =  kk + 1;
  end
  axis([1 length(t_data) 0 length(Fields)+1]);
%
%  Get the field names into a format we can use for labelling
%
  YLabels  =  repmat(' ',[max_field_name,length(Fields)+2]);
  kk  =  1;
  while(kk  <=  length(Fields))
    YLabels(end-(length(Fields{kk}(:))-1):end,kk+1)  =  Fields{kk}(:);
    kk  =  kk + 1;
  end
  YLabels  =  flipud(YLabels');
%
%  Make the plot look pretty
%
  AX  =  gca;
  set(AX,'XTick',0:10:length(t_data));
  XTicks  =  get(AX,'XTick');
  Dates  =  Dates(XTicks(2:end));
  Dates(2:end+1)  =  Dates(1:end);
  set(AX,'XTickLabel',Dates);
  set(AX,'YTick',0:1:length(Fields)+1);
  set(AX,'YTickLabel',YLabels);
  set(AX,'FontSize',6);
  box on;
  grid on;
  hold off;