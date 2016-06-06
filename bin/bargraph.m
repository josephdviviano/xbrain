function bargraph(data, titles, maximum);
    dims = size(data);
    numplts = dims(1);

    figure;

    for plt = 1:numplts;
        subplot(1, numplts, plt);
        bar(data(plt, :), 'FaceColor',[0 0 0],'EdgeColor',[0 0 0]);
        ylim([-abs(maximum) abs(maximum)])
        title(titles{plt});
    end
end


