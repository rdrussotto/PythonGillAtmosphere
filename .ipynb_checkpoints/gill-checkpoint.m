% Script to solve Gill problem, exact, in WTG limit, and no-dissipation 
% comparison, for a specified zonally compensated mass source.  We use an FFT 
% with periodic BC in x and finite differencing with BC v = 0 at N, S walls.
% The main control parameters are:
%   nodiss (switch for nondim Rayleigh friction- 0: a= 0.15; 1: a = 0.0001)
%   zonalcomp (switch to turn on zonal compensation of mass source)
% Other control parameters controlling domain shape, etc. are near top of 
% script. 
% If nodiss = 0, three two-panel figures are plotted:
%  figure(1)  Gill solution div and vort/u/v (b = a, b = Newtonian cooling)
%   ...and if zonalcomp = 1...
%  figure(2)  WTG solution div and vort/u/v (b = 0), zonally-compensated.
%  figure(3)  Gill and WTG geopotls.
% If nodiss = 1, one two-panel figure is plotted:
%  figure(1)  a = b = 0 vort/u/v and phi.
%
% This script is used to generate the following figures in the WTG_gill paper:
% Setting nodiss, zonalcomp, y0 = 0 (original Gill model):
%  figure(1)  gill-nocomp-uv.eps  
% Setting nodiss = 0, zonalcomp = 1, y0 = 0 (zonally compensated Gill model):
%  figure(1)  gill-uv.eps
%  figure(2)  WTG-uv.eps
%  figure(3)  comp-phi.eps
% We also get rms relative error stats uerr, verr, phierr, zetaerr, etc.
% Setting nodiss = 1, zonalcomp = 1, y0 = 0 (no dissipation):
%  figure(1)  nodiss.eps

%Modified by Rick Russotto (starting October 2018) to save plots as PDFs
%(since can't run Matlab as gui) and print test output to find discrepancies
%with Python version I'm developing. 

%Printing test output didn't work so well...
%(matrices take up too much space) maybe better to just run
%and look at the variables I need within command line,
%comparing to the Python version with printed test output there.

%-------------User-defined parameters follow...--------------

  zonalcomp = 0; % No zonal compensation of heating (Original Gill problem)
%  zonalcomp = 1; % Gill problem with zonally compensated heating
%  nodiss = 1; % Inviscid, no thermal diffusion
  nodiss = 0; % Thermal diffusivity = Rayleigh damping (Original Gill problem)

%  Set Rayleigh friction (nondimensionalized in units of sqrt(beta*c))

  if(nodiss)
    a = 0.0001;  % a = 0 would produce divide by zero.
  else
    a = 0.15;  
  end

%  Define thermal diffusivity b
b = a;

%  Define other physical parameters

  H = 1;     % Layer depth
  g = 1;     % gravity
  beta = 1;  % df/dy

%  Domain size and number of mesh points

  lx = 20; % periodic domain width -lx/2 < x < lx/2
  ly = 20; % Rigid walls at +/- ly/2
  nx = 128; % Number of gridpoints in x
  ny = 120; % Number of y gridpoints in -ly/2 <= y < ly/2

%  Parameters defining plot window in x,y and stride (in gridpoints) for
%  velocity vector plotting

  xmin = -lx/2;
  xmax = lx/2;
  ymin = -3.5;
  ymax = 3.5;
  stride = 4;

%----------End of user-defined parameters---------  

  dx = lx/nx;
  x = -lx/2 + dx*(0:(nx-1));
  dy = ly/ny;
  y = -ly/2 + dy*(0:ny);
  [X,Y] = meshgrid(x,y);
  
%     %Russotto: indent test output
%         x
%         y

% Shallow water phase speed

  c = sqrt(g*H);

%   Define mass source M(x,y)

  sx = 2;    % Mass source half-width in x
  sy = 1;    % Mass source half-width in y
  x0 = 0;    % Central x of mass source
  y0 = 0;    % Central y of mass source.

  kh = pi/(2*sx);
  phase = kh*(X-x0);
  phase(X-x0>sx) = pi/2;
  phase(X-x0<-sx) = pi/2;
  F = cos(phase);
  M =  F.*exp(-(Y-y0).^2/(2*sy^2));
  M(1,:) = 0;                  % Zero the heating at top and bottom bdries.
  M(ny+1,:) = 0;
  if (zonalcomp)
    M = M - (mean(M.')).'*ones(1,nx); % Correct zonal avg heating to be zero
  end
  dMdy = -(Y-y0).*M/sy^2;
  d2Mdy2 = ((Y-y0).^2/sy^2 - 1).*M/sy^2;
  Mhat = (fft(M.')).';
  dMdyhat = (fft(dMdy.')).';
  
%     %Test output
%         g
%         kh
%         phase
%         F
%         M
%         dMdy
%         d2Mdy2
%         Mhat
%         dMdyhat
        

%  Define waveno. matrix 

  kx = (2*pi/lx)*[0:(nx/2 - 1) (-nx/2):(-1)];
  KX = ones(ny+1,1)*kx;
  
%         kx
%         KX

%-------------------Gill computations-----------------------------------------


%
%  Define v source term Sv = (a*d/dy - beta*y*d/dx)M/H
  
  Svhat = (a*dMdyhat - beta*1i*KX.*Y.*Mhat)/H;
        
%        Svhat

%
%  Solve (-b(a^2 + beta^2y^2)/c^2 + a*del^2 + beta*d/dx)v = Sv, or
%
%  a*d2vhat/dy2 + (-b(a^2 + beta^2y^2)/c^2 -a*k^2 + i*k*beta)vhat = Svhat
%
%  where Sv = (a*d/dy - beta*y*d/dx)M/H is the same source term
%  as in WTG (but this time we don't remove wavenumber zero).

%  This is done as a loop over wavenumbers, using only the interior 
%  y-gridpoints 2:ny (since v = 0 at the boundaries)

  vhat = zeros(ny+1,nx);
  d1 = a/dy^2;
  for i = 1:nx
    k = kx(i);
    d0 = -2*d1 - a*k^2 + 1i*k*beta;
    e = ones(ny-1,1);
    Av = spdiags([d1*e d0*e-b*(a^2 + (beta*(y(2:ny))').^2)/c^2 d1*e],...
                 -1:1,ny-1,ny-1);
    r = Svhat(2:ny,i);
    vhat(2:ny,i) = Av\r;
    %disp(i)
    %disp(vhat(41,i))
  end
  v = real((ifft(vhat.')).');
  
%         d1
%         vhat
%         v
        

%  Calculate phi from
%
%     (b/c^2)phi + du/dx = M/H - dv/dy 
%  and
%     a*u = -dphi/dx + beta*y*v
%  Eliminating u between these equations,
%     (a*b/c^2 - d2/dx2)phi = aM/H - a dv/dy - beta*y*dv/dx
%  whose FFT in x diagnoses phi

  dvdyhat = zeros(ny+1,nx);
  dvdyhat(2:ny,:) = (vhat(3:(ny+1),:) - vhat(1:(ny-1),:))/(2*dy);
  dvdyhat(1,:) = (vhat(2,:) - vhat(1,:))/dy;
  dvdyhat(ny+1,:) = (vhat(ny+1,:) - vhat(ny,:))/dy;
  phihat = (a*Mhat/H - a*dvdyhat - 1i*beta*Y.*KX.*vhat)./(a*b/c^2 + KX.^2);
  phi = real((ifft(phihat.')).');

  D = M/H - b*phi/c^2;
  
%         dvdyhat
%         phihat
%         phi
%         D


%  Calculate vorticity zeta from divergence and v:
%
%    a*zeta + beta*(y*D + v) = 0

  zeta = (beta/a)*(-Y.*D - v);
  
%         zeta

%  u calculated using div eqn: du/dx + dv/dy = M/H-b*phi/(c^2)

  dvdy = zeros(ny+1,nx);
  dvdy(2:ny,:) = (v(3:ny+1,:)  - v(1:(ny-1),:))/(2*dy);
  
  uhat = (Mhat/H - b*phihat/c^2-(fft(dvdy.')).');
%   uhat=(Mhat/H-(fft(dvdy.')).');				  

  uhat(:,2:nx) = uhat(:,2:nx)./(1i*KX(:,2:nx));

%  The k=0 components are indeterminate; for these go to zonally
%  averaged vorticity equation dudyhat(:,1) = -zetahat(:,1), with BC that
%  the meridional average of uhat(:,1) should equal zero. 

  zetahat = (fft(zeta.')).';
  dudyhath = -0.5*(zetahat(1:ny,1) + zetahat(2:(ny+1),1)); 
  uhat(:,1) = [0; dy*cumsum(dudyhath)];
  uhatmean = mean([uhat(2:ny,1);0.5*(uhat(1,1)+uhat(ny+1,1))]);
  uhat(:,1) = uhat(:,1) - uhatmean;

  ucompare = real((ifft(uhat.')).');
  
%         dvdy
%         uhat
%         uhatmean
%         zetahat
%         dudyhath
%         ucompare
	    

%  u calculated using x momentum eqn: -beta.y.v=-d(phi)/dx-a.u 
		   dphidx=zeros(ny+1,nx);  

dphidx(:,2:nx-1)=(phi(:,3:nx)-phi(:,1:nx-2))/(2*dx);
  dphidx(:,1)=(phi(:,2)-phi(:,nx))/(2*dx);
  dphidx(:,nx)=(phi(:,1)-phi(:,nx-1))/(2*dx);
  u=(-beta*Y.*v+dphidx)/(-a);


%         dphidx
%         u








  if (~nodiss)

%    figure(1): Plot Gill div and vel/vort
%    If mass source is compensated (zonalcomp = 1), plot 2 additional figs:
%     figure(2): WTG div and vel/vort
%     figure(3): Gill phi and WTG phi
%    If mass source is compensated and off-equatorial, plot
%     figure(1): Gill vel/vort and WTG vel/vort

%-------Gill plots--------
    figure(1)
    subplot(2,1,1)

%    Plot Gill divergence

    cint = -0.9:0.2:-0.1;
    contour(x,y,D,cint,'k-')
    hold on
    cint = [-0.06 -0.02];
    %contour(x,y,D,cint,'k--')
    contour(x,y,D,cint,'k:')
    cint = [0.02:0.04:0.1];
    contour(x,y,D,cint,'k-.')
    axis equal
    axis([xmin xmax ymin ymax])
    xlabel('x/R_{eq}')
    ylabel('y/R_{eq}')
    text(0.75*xmax+0.25*xmin,0.85*ymax+0.15*ymin,...
         ['a = b = ' num2str(a) 'c/R_{eq}'])
    title('Gill convergence')
    hold off

%    Plot Gill velocity vectors and vorticity
 
    subplot(2,1,2)

%    czetamax = max(max(zeta))
    czetamax = 3;     % fixed contours
    cpos = (0.1:0.2:1.9)*czetamax;
    contour(x,y,zeta,cpos,'k-')
    hold on
%    contour(x,y,zeta,-cpos,'k--')
    contour(x,y,zeta,-cpos,'k:')
    axis equal
    axis([xmin xmax ymin ymax])
    xlabel('x/R_{eq}')
    ylabel('y/R_{eq}')
    text(0.75*xmax+0.25*xmin,0.85*ymax+0.15*ymin,['a = b = ' num2str(a) 'c/R_{eq}'])
    title('Gill Velocity and Vorticity')
  
%    Plot velocity vectors

    quiver(x(1:stride:nx),y(1:stride:ny),...
           u(1:stride:ny,1:stride:nx),v(1:stride:ny,1:stride:nx))

    hold off
    
    % Russotto: save figure to pdf
        saveas(gcf, 'gill01.pdf')
        %saveas(gcf, 'gill01.png')

%    Plot Gill phi

    figure(3)
    subplot(2,1,1)
%    cphimax = max(max(abs(phi)))
    cphimax = 2;      % fixed contours
    cpos = (0.1:0.2:1.9)*cphimax;

    contour(x,y,phi,cpos,'k-')
    hold on
    %contour(x,y,phi,-cpos,'k--')
    contour(x,y,phi,-cpos,'k:')
    axis equal
    axis([xmin xmax ymin ymax])
    xlabel('x/R_{eq}')
    ylabel('y/R_{eq}')
    title('Gill geopotential')

	   

    hold off
    
    % Russotto: save figure to pdf
        saveas(gcf, 'gill03.pdf')
        %saveas(gcf, 'gill03.png')

    if (zonalcomp)
  
      ugill = u;
      vgill = v;
      Dgill = D;  
      phigill = phi;
      zetagill = zeta;  
  
%  -----------WTG computations (require zonally-compensated heating)--------
%  
%      Define v source term Sv = (a*d/dy - beta*y*d/dx)M/H
  
      Svhat = (a*dMdyhat - beta*1i*KX.*Y.*Mhat)/H;
  
%      Solve a*del^2 v + beta*dv/dx = Sv, or
%  
%      a*d2vhat/dy2 + (-a*k^2 + i*k*beta)vhat = Svhat
%  
%      This is done as a loop over wavenumbers, using only the interior 
%      y-gridpoints 2:ny (since v = 0 at the boundaries)
%      By skipping i = 1 (k=0) the solution automatically removes the zonal
%      mean (k=0) contribution to the heating, as required by WTG.
  
      vhat = zeros(ny+1,nx);
      d1 = a/dy^2;
      for i = 2:nx
        k = kx(i);
        d0 = -2*d1 - a*k^2 + 1i*k*beta;
        e = ones(ny-1,1); 
        Av = spdiags([d1*e d0*e d1*e],-1:1,ny-1,ny-1);
        r = Svhat(2:ny,i);
        vhat(2:ny,i) = Av\r;
      end
      v = real((ifft(vhat.')).');
  
      D = M/H;
    
      figure(2)
  
%      Plot WTG divergence
  
      subplot(2,1,1)
      cint = -0.9:0.2:-0.1;
      contour(x,y,D,cint,'k-')
      hold on
      cint = [-0.06 -0.02];
      contour(x,y,D,cint,'k--')
      cint = [0.02:0.04:0.1];
      contour(x,y,D,cint,'k-.')
      axis equal
      axis([xmin xmax ymin ymax])
      xlabel('x/R_{eq}')
      ylabel('y/R_{eq}')
      text(0.75*xmax+0.25*xmin,0.85*ymax+0.15*ymin,['a = ' num2str(a) 'c/R_{eq}'])
      title('WTG convergence')
      hold off
  
%      Plot WTG velocity vectors and vorticity

      subplot(2,1,2)
  
%      Calculate vorticity zeta from divergence and v:
%
%        a*zeta + beta*(y*D + v) = 0
  
      zeta = (beta/a)*(-Y.*D - v);

%      czetamax = max(max(zeta))
      czetamax = 3;    % fixed contours
      cpos = (0.1:0.2:1.9)*czetamax;
      contour(x,y,zeta,cpos,'k-')
      hold on
      contour(x,y,zeta,-cpos,'k--')
      axis equal
      axis([xmin xmax ymin ymax])
      xlabel('x/R_{eq}')
      ylabel('y/R_{eq}')
      title('WTG Velocity and Vorticity')
  
%      u calculated using div eqn du/dx + dv/dy = M/H
  
      dvdy = zeros(ny+1,nx);
      dvdy(2:ny,:) = (v(3:ny+1,:)  - v(1:(ny-1),:))/(2*dy);
      uhat = (Mhat/H - (fft(dvdy.')).');
      uhat(:,2:nx) = uhat(:,2:nx)./(1i*KX(:,2:nx));
  
%      The k=0 components are indeterminate; for these go to zonally
%      averaged (bar) vorticity equation dudybar = -zetabar 
%      For WTG, ubar = 0 since no waveno 0 heat source.
  
      uhat(:,1) = 0;
  
      u = real((ifft(uhat.')).');
  
%      Plot velocity vectors

      quiver(x(1:stride:nx),y(1:stride:ny),...
             u(1:stride:ny,1:stride:nx),v(1:stride:ny,1:stride:nx))

      hold off
      
      % Russotto: save as pdf
      saveas(gcf, 'gill02.pdf')

%      Calculate WTG phi from -del^2 phi = a*D - beta*y*zeta + beta*u
%      We use the BC i*k*beta*y*phihat = a*dphihat/dy at the boundaries,
%      derived from the momentum equations.

      phihat = zeros(ny+1,nx);
      yzetahat = (fft((Y.*zeta).')).';
      Sphihat = a*Mhat/H - beta*yzetahat + beta*uhat;
      Sphihat(:,1) = 0.;

      d1 = 1/dy^2;
      for i = 2:nx   % Skip wavenumber zero to account for zero-mean heat source
        k = kx(i);
        e = ones(ny+1,1) ;
        Ap = spdiags([-d1*e (2*d1+k^2)*e -d1*e],-1:1,ny+1,ny+1);
        Ap(1,2) = a/dy;
        Ap(1,1) = -a/dy - i*k*beta*(-ly/2);
        Ap(ny+1,ny+1) = a/dy - i*k*beta*(ly/2);
        Ap(ny+1,ny) = -a/dy;
        r = [0; Sphihat(2:ny,i); 0];
        phihat(:,i) = Ap\r;
      end % i
      phi = real((ifft(phihat.')).');

%      Plot WTG phi

      figure(3)
      subplot(2,1,2)
%      cphimax = max(max(abs(phi)))
      cphimax = 2;
      cpos = (0.1:0.2:1.9)*cphimax;
      contour(x,y,phi,cpos,'k-')
      hold on
      contour(x,y,phi,-cpos,'k--')
      axis equal
      axis([xmin xmax ymin ymax])
      xlabel('x/R_{eq}')
      ylabel('y/R_{eq}')
      title('WTG geopotential')
      hold off
      
      % Russotto: save figure as pdf
      saveas(gcf, 'gill103.pdf')

%      Relative error statistics

      uerr = var(u(:)-ugill(:))/var(ugill(:));
      verr = var(v(:)-vgill(:))/var(vgill(:));
      Derr = var(D(:)-Dgill(:))/var(Dgill(:));
      phierr = var(phi(:)-phigill(:))/var(phigill(:));
      zetaerr = var(zeta(:)-zetagill(:))/var(zetagill(:));
  
    end  % if(zonalcomp)  

  else 

%    nodiss = 1 (true). Plot only one figure: 
%      (top) undamped velocity and vorticity
%      (bot) undamped phi

    figure(1)
    subplot(2,1,1)

%    czetamax = max(max(zeta))
    czetamax = 3;     % fixed contours
    cpos = (0.1:0.2:1.9)*czetamax;
    contour(x,y,zeta,cpos,'k-')
    hold on
    contour(x,y,zeta,-cpos,'k--')
    axis equal
    axis([xmin xmax ymin ymax])
    xlabel('x/R_{eq}')
    ylabel('y/R_{eq}')
    text(0.75*xmax+0.25*xmin,0.85*ymax+0.15*ymin,'a = b = 0')
    title('Undamped Velocity and Vorticity')
    stride = 4;
    quiver(x(1:stride:nx),y(1:stride:ny),...
           u(1:stride:ny,1:stride:nx),v(1:stride:ny,1:stride:nx))

    subplot(2,1,2)
%    cphimax = max(max(abs(phi)))
    cphimax = 2;      % fixed contours
    cpos = (0.1:0.2:1.9)*cphimax;

    contour(x,y,phi,cpos,'k-')
    hold on
    contour(x,y,phi,-cpos,'k--')
    axis equal
    axis([xmin xmax ymin ymax])
    xlabel('x/R_{eq}')
    ylabel('y/R_{eq}')
    title('Undamped Geopotential')
    hold off

    subplot(2,1,2)
%    cphimax = max(max(abs(phi)))
    cphimax = 2;      % fixed contours
    cpos = (0.1:0.2:1.9)*cphimax;

    contour(x,y,phi,cpos,'k-')
    hold on
    contour(x,y,phi,-cpos,'k--')
    axis equal
    axis([xmin xmax ymin ymax])
    xlabel('x/R_{eq}')
    ylabel('y/R_{eq}')
    title('Undamped geopotential')
    hold off
    
    % Russotto: save figure as pdf
    saveas(gcf, 'gill101.pdf')


%    Plot exact undamped solution in same format as FFT soln if desired.

    exactnodiss = 1; % Switch to plot exact undamped solution .
    if(exactnodiss)
      figure(2)
      v_ud = -Y.*M/H;
      dudx_ud = (2*M + Y.*dMdy)/H;
      dudxhat_ud = fft(dudx_ud.').';
      uhat_ud(:,2:nx) = dudxhat_ud(:,2:nx)./(1i*KX(:,2:nx));
      uhat_ud(:,1) = 0;
      u_ud = real((ifft(uhat_ud.')).');

      dphidx_ud = -beta.*Y.*Y.*M/H;
      dphidxhat_ud = fft(dphidx_ud.').';
      phihat_ud(:,2:nx) = dphidxhat_ud(:,2:nx)./(1i*KX(:,2:nx));
      phihat_ud(:,1) = 0;
      phi_ud = real((ifft(phihat_ud.')).');

      dvdxhat_ud = 1i*KX.*(fft(v_ud.').');
      d2udxdy_ud = (3*dMdy + Y.*d2Mdy2)/H;
      d2udxdyhat_ud = fft(d2udxdy_ud.').';
      dudyhat_ud(:,2:nx) = d2udxdyhat_ud(:,2:nx)./(1i*KX(:,2:nx));
      dudyhat_ud(:,1) = 0;
      zeta_ud = real((ifft((dvdxhat_ud - dudyhat_ud).')).');
  
      subplot(2,1,1)
      czetamax = 3;
      cpos = (0.1:0.2:1.9)*czetamax;
      contour(x,y,zeta_ud,cpos,'k-')
      hold on
      contour(x,y,zeta_ud,-cpos,'k--')
      axis equal
      axis([xmin xmax ymin ymax])
      xlabel('x/R_{eq}')
      ylabel('y/R_{eq}')
      text(0.75*xmax+0.25*xmin,0.85*ymax+0.15*ymin,'a = b = 0')
      title('Exact undamped velocity and vorticity')
      stride = 4;
      quiver(x(1:stride:nx),y(1:stride:ny),...
             u_ud(1:stride:ny,1:stride:nx),v_ud(1:stride:ny,1:stride:nx))

      subplot(2,1,2)
      cphimax = 2;
      cpos = (0.1:0.2:1.9)*cphimax;
      contour(x,y,phi_ud,cpos,'k-')
      hold on
      contour(x,y,phi_ud,-cpos,'k--')
      axis equal
      axis([xmin xmax ymin ymax])
      xlabel('x/R_{eq}')
      ylabel('y/R_{eq}')
      title('Exact undamped geopotential')
      hold off
      
      % Russotto: save figure as pdf
      saveas(gcf, 'gill102.pdf')
      
    end % exactnodiss
  end % if(nodiss)

