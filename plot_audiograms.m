audiograms = [
              0 0 0 55 80 90;
              0 15 30 60 80 85;
              14 14 11 14 24 39;
              24 24 25 31 46 60;
              40 40 50 60 65 65;
              45 55 50 30 20 20  
              ];
%set(gcf, 'Position',  [100, 100, 500, 350])
set(0,'defaulttextinterpreter','latex')
%'defaultUicontrolFontName'
%'defaultUitableFontName'
set(0, 'defaultAxesFontName', 'latex')
set(0, 'defaultTextFontName', 'latex')
set(0, 'defaultUipanelFontName', 'latex')

%set(gca,'FontName','cmr12')
%set(0,'DefaultTextFontname', 'CMU Serif')
%set(0,'DefaultAxesFontName', 'CMU Serif')
font = 'Times';
size = 12;
%figure('DefaultTextFontName', font, 'DefaultAxesFontName', font);
h1 = subplot(1,1,1);
h = plot([1,2,3,4,5,5.5], audiograms', '-o', 'linewidth', 2);
set(h1, 'Ydir', 'reverse');
set(h1, 'Ydir', 'reverse');
xlim([0.75 5.75]);
ylim([-20 120]);
set(gca, 'XTick' ,[1 2 3 4 5 5.5], 'FontName', 'Times','FontSize',size);
set(gca, 'XTickLabel',{'0.25'; '0.5'; '1'; '2'; '4'; '6'}, 'FontSize',size,'FontName', 'Times');
grid on
%set(axes1,'FontName',font,'FontSize',fntsz);
ylabel('Threshold (dB HL)','FontSize',size+2);
xlabel('Frequency (kHz)','FontSize',size+2);
%set(findall(gcf,'-property','FontSize'),'FontSize',12)
