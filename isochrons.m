% isochrons code taken directly from izhikevich dsn 2009
% for iapp=55, run isochrons('F',0:0.1:64.81,[32.2198;0.229432])
% use ml.py to extract initial conditions and period given iappe

function isochrons(F,phases,x0)
    % plot isochrons of a planar dynamical system x’=F(t,x)
    % at points given by the vector ’phases’.
    % ’x0’ is a point on the limit cycle (2x1-vector)
    T= phases(end); % is the period of the cycle
    tau = T/600; % time step of integration
    m=200; % spatial grid
    k=1; % the number of skipped cycles
    [t,lc] = ode23s(F,0:tau:T,x0); % forward integration

    figure(1)
    
    dx=(max(lc)-min(lc))'/m; % spatial resolution
    center = (max(lc)+min(lc))'/2; % center of the limit cycle
    iso=[x0-m^0.5*dx, x0+m^0.5*dx] % isochron’s initial segment
    
    n = ceil(T/tau)+1;
    H = zeros(n,3);
    H(:,1) = linspace(0,1,n); % hue
    H(:,2) = 1; % contrast
    H(:,3) = 1; % brightness
    M = colormap(hsv2rgb(H));
    kk=1;
    for t=0:-tau:-(k+1)*T % backward integration
        
        for i=1:size(iso,2)
            iso(:,i)=iso(:,i)-tau*feval(F,t,iso(:,i)); % move one step
        end;
        i=1;
        while i<=size(iso,2) % remove infinite solutions
            if any(abs(iso(:,i)-center)>1.5*m*dx) % check boundaries
                iso = [iso(:,1:i-1), iso(:,i+1:end)]; % remove
            else
                i=i+1;
            end;
        end;
        i=1;
        while i<=size(iso,2)-1
            d=sqrt(sum(((iso(:,i)-iso(:,i+1))./dx).^2)); % normalized distance
            if d > 2 % add a point in the middle
                iso = [iso(:,1:i), (iso(:,i)+iso(:,i+1))/2 ,iso(:,i+1:end)];
            end;
            if d < 0.5 % remove the point
                iso = [iso(:,1:i), iso(:,i+2:end)];
            else
                i=i+1;
            end;
        end;
        %if (mod(-t,T)<=tau/2) & (-t<k*T+tau) % refresh the screen
        %    cla;plot(lc(:,1),lc(:,2),'r'); hold on; % plot the limit cycle
        %end;
        if min(abs(mod(-t,T)-phases))<tau/2 % plot the isochrons
            plot(lc(:,1),lc(:,2),'k','LineWidth',5); hold on;
            %plot(iso(1,:),iso(2,:),'k-'); drawnow;
        end
        
        if t <= -T
            %if mod(t,10*tau)
            if mod(k,50)==1
                plot(iso(1,:),iso(2,:),'Color',M(kk,:)); hold on;
            end
            %end
            kk = kk + 1;
        end

        %hold off;
        %hue = hue + (tau/T);
        
    end;
    xlim([-60 55])
    ylim([-.1 .5])
end
