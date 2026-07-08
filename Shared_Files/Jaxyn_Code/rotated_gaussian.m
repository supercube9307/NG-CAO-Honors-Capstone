function [y, fitted_gaussian] = rotated_gaussian (par)
global frame_to_fit xp yp ifit iter
% global frame_to_fit indr indc ifit deltax deltay fitEval

%fit_function = par(1) + par(2) .* cos( par(3)*(xpixnom+deltax) +  par(4)*(ypixnom+deltay) + par(5) );
fit_function = par(1) + par(2)*exp(-(cos(par(3))*(xp-par(4))-sin(par(3))*(yp-par(5))).^2/(2*par(6)^2) - (sin(par(3))*(xp-par(4))+cos(par(3))*(yp-par(5))).^2/(2*par(7)^2));
 

fitEval =  fit_function - ifit * frame_to_fit;

y = fitEval(:);
fitted_gaussian = fit_function;

iter = iter+1;
disp(['rotated_gaussian:  iter=', num2str(iter),'   ',num2str(norm(y))])

