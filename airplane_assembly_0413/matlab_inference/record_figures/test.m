
close all

data = record_figures_init();

for i=1:3:100
    
    nx_figure(10);
    imshow(ones(100) * i / 100);
    
    nx_figure(20);
    imshow(ones(100) * i / 100);
    
    nx_figure(99);
    imshow(ones(100) * i / 100);

    data = record_figures_process(data);
end

data = record_figures_terminate(data);











