% %%%%%%%%%%%%%%%%%%%%%
% % Visualization
set(0,'defaulttextinterpreter','latex')
set(0,'defaultfigurecolor',[1 1 1])
set(0,'defaultaxesfontsize',11);
set(0,{'DefaultAxesXColor','DefaultAxesYColor','DefaultAxesZColor','DefaultTextColor'},...
    {'k','k','k','k'});
set(gca,'FontSize',30) %gca returns the current axes in the current figure

clear all;
load drawData
Pgpmp = interp1(t_pmp,Pg_pmp,t);
err_dp = abs(Pg_sp-Pg_dp);
err_pmp = abs(Pg_sp-Pgpmp);


figure(1);
%subplot(3,1,[1 2]);
plot(t,Pl,'-r','linewidth',2);
hold on;
plot(t_pmp,Pg_pmp,'k-x','linewidth',1.5,'MarkerSize',6,'MarkerFaceColor',[1 1 1],'MarkerIndices',1:500:length(Pg_pmp));
plot(t,Pg_dp,'b-o','linewidth',1.5,'MarkerSize',4,'MarkerFaceColor',[1 1 1],'MarkerIndices',1:80:length(Pg_dp));
plot(t,Pg_sp,'--k','Color',[0 0.8 0],'linewidth',2);
xlim([0 45]);
ylim([15 43]);
ylabel('Power flow', 'FontSize',30);
grid on
lgd = legend('Load','Dynamic Programming', 'Pontryagins Minimum Principal', 'Shortest path', 'Location', 'NorthWest');
set(lgd,'Interpreter','latex', 'FontSize',25)
ax = gca;
ax.FontSize = 25;  % Font Size of 15

% subplot(3,1,3);
% plot(t,abs(Pg_sp-Pgpmp),'k-x','linewidth',2,'MarkerSize',6,'MarkerFaceColor',[1 1 1],'MarkerIndices',1:100:length(Pg_sp));
% hold on;
% plot(t,abs(Pg_sp-Pg_dp),'b-o','linewidth',2,'MarkerSize',4,'MarkerFaceColor',[1 1 1],'MarkerIndices',1:80:length(Pg_dp));
% xlim([0 45]);
% grid on
% ylim([0 7]);
% ylabel('Error');
%  xlabel('Time [s]', 'FontSize',30);
% % lgd = legend('$|E_{g,SP}-E_{g,PMP}|$', '$|E_{g,SP}-E_{g,DP}|$', 'Location', 'northwest');
% % set(lgd,'Interpreter','latex', 'FontSize',9)
% ax = gca;
% ax.FontSize = 25;  % Font Size of 15





axesHandles = findall(0,'type','axes');
set(axesHandles,'TickLabelInterpreter', 'latex')